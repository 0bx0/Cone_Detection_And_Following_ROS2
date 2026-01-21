#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo
from cone_msgs.msg import ConeDetectionArray
from std_msgs.msg import String, Float64

from cone_msgs.action import FollowCone

# =========================================================================================
#                                 TUNING CONFIGURATION
# =========================================================================================
CONFIG = {
    # --- SPEEDS ---
    'forward_speed':       0.2,   
    'turn_speed':          0.3,   
    'backup_speed':       -0.2,   

    # --- P-CONTROLLER FOR TURNING ---
    'turn_kp':             0.02,   
    'max_turn_speed':      0.3,    
    'min_turn_speed':      0.1,     

    # --- DISTANCES ---
    'stop_distance':       1.9,    # Stop at this distance from cone
    'max_detection_dist':  50.0,   
    
    # --- 180 TURN CONFIG ---
    'heading_tolerance':   4.0,    # Degrees tolerance for 180 turn completion
    'settle_time':         1.0,    
    'turn_timeout':        20.0,   
    
    # --- ALIGNMENT ---
    'kp':                  0.003,  
    'ki':                  0.0001, 
    'ki_max':              0.1,    
    'alignment_tolerance': 100.0,   # Pixels tolerance for alignment
    'inner_band':          200.0,   
    'outer_band':          360.0,  
    'final_align_tol':     20.0,   
    
    # --- TIMERS ---
    'lost_timeout':        5.0,    
    'grace_period':        1.0,    
    'backup_duration':     2.5,    # Backup for 2-3 seconds (using 2.5s)
    'catchbox_delay':      3.0,    
    'retry_backoff_dur':   4.0,    
    'stop_settle_time':    0.5,    # Time to wait after stopping before next action
}
# =========================================================================================

class ConeFollowerNode(Node):
    """
    Cone Follower Node implementing complete control loop:
    1. SEARCH: Rotate to find cone
    2. STOP_ON_DETECT: Stop when cone detected  
    3. ALIGN: Align ZED center (width/4) to cone center (stop-and-turn)
    4. APPROACH: Move forward toward cone
    5. STOP_AT_CONE: Stop when reached stop distance
    6. TURN_180: Use compass heading to perform 180° turn
    7. BACKUP: Move backward for 2-3 seconds
    8. MISSION_COMPLETE: Send goal completed signal
    """
    
    def __init__(self):
        super().__init__('cone_follower_node')
        print("Cone Follower [SAFETY CRITICAL EDITION] - Started")

        self.declare_parameter('detection_topic', '/cone_detector/detections')
        self.declare_parameter('camera_info_topic', '/zed/zed_node/left/camera_info') 
        self.declare_parameter('compass_topic', '/mavros/global_position/compass_hdg')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('fallback_image_width', 1280.0)

        for key, value in CONFIG.items():
            self.declare_parameter(key, value)

        self.detection_topic = self.get_parameter('detection_topic').value
        self.cam_info_topic = self.get_parameter('camera_info_topic').value
        self.compass_topic = self.get_parameter('compass_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.fallback_width = self.get_parameter('fallback_image_width').value
        self.image_center_x = self.fallback_width / 2.0  # Center of left camera image
        self.camera_info_received = False

        # Load Params
        self.fwd_speed = self.get_parameter('forward_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.backup_speed = self.get_parameter('backup_speed').value
        self.stop_dist = self.get_parameter('stop_distance').value
        self.max_dist = self.get_parameter('max_detection_dist').value
        
        self.heading_tol = self.get_parameter('heading_tolerance').value
        self.settle_time = self.get_parameter('settle_time').value
        self.turn_timeout = self.get_parameter('turn_timeout').value
        
        self.turn_kp = self.get_parameter('turn_kp').value
        self.max_turn_speed = self.get_parameter('max_turn_speed').value
        self.min_turn_speed = self.get_parameter('min_turn_speed').value

        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.ki_max = self.get_parameter('ki_max').value
        self.alignment_tol = self.get_parameter('alignment_tolerance').value
        
        self.inner_band = self.get_parameter('inner_band').value
        self.outer_band = self.get_parameter('outer_band').value
        self.final_tol = self.get_parameter('final_align_tol').value
        
        self.lost_timeout = self.get_parameter('lost_timeout').value
        self.grace_period = self.get_parameter('grace_period').value
        self.backup_dur = self.get_parameter('backup_duration').value
        self.catchbox_wait = self.get_parameter('catchbox_delay').value
        self.retry_backoff = self.get_parameter('retry_backoff_dur').value
        self.stop_settle = self.get_parameter('stop_settle_time').value

        # State machine - start IDLE, wait for action goal
        self.state = 'IDLE'
        self.last_detection = None
        self.last_valid_time = self.get_clock().now()
        self.state_start_time = self.get_clock().now()
        self.integral_error = 0.0
        
        self.current_heading = None
        self.target_heading = None
        self.turn_direction = None  # Lock in turn direction for 180-degree turn (+1 or -1)
        self.backup_start_time = None
        
        # Action Server State
        self.goal_handle = None
        self.goal_color = None
        self.goal_type = None
        self.action_active = False

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, '/auto/cone_follow/status', 10)
        
        self.create_subscription(ConeDetectionArray, self.detection_topic, self.detections_cb, 10)
        self.create_subscription(CameraInfo, self.cam_info_topic, self.camera_info_cb, 10)
        # MAVROS uses BEST_EFFORT reliability - must match QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(Float64, self.compass_topic, self.compass_cb, sensor_qos)
        
        # Action Server
        self.action_server = ActionServer(
            self,
            FollowCone,
            'cone_follow',
            self.execute_cone_following
        )

        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Cone Follower Action Server initialized')
        self.get_logger().info('⏳ Waiting for action goal (cone_follow)...')

    # =========================================================================================
    #                               SAFETY & HELPER FUNCTIONS
    # =========================================================================================
    
    def stop_all_motion(self):
        """
        CRITICAL SAFETY FUNCTION.
        Spams 0.0 velocity to ensuring the rover stops dead.
        """
        self.get_logger().warn("!!! EMERGENCY STOP TRIGGERED !!!")
        t = Twist()
        t.linear.x = 0.0
        t.angular.z = 0.0
        # Send 20 stop commands to flood the buffer and guarantee stop
        for _ in range(20): 
            self.cmd_pub.publish(t)
            time.sleep(0.02)

    def stop_motion(self):
        """Send a single stop command (non-emergency)."""
        t = Twist()
        t.linear.x = 0.0
        t.angular.z = 0.0
        self.cmd_pub.publish(t)

    def normalize_heading(self, heading):
        while heading < 0: heading += 360.0
        while heading >= 360: heading -= 360.0
        return heading

    def heading_diff(self, current, target):
        """Returns shortest path from current to target (-180 to +180)"""
        diff = target - current
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        return diff

    def get_steering_cmd(self, error, limit_speed=True, use_integral=True):
        # DEAD-ZONE: If error is small, don't apply any turning (prevents oscillation)
        if abs(error) < self.final_tol:
            self.get_logger().info(f"  PID | Err: {error:6.1f} | IN DEAD-ZONE, no turn")
            return 0.0
        
        p_out = error * self.kp
        if use_integral:
            self.integral_error += error
            clamp = self.ki_max / self.ki
            if self.integral_error > clamp: self.integral_error = clamp
            if self.integral_error < -clamp: self.integral_error = -clamp
            i_out = self.integral_error * self.ki
        else:
            i_out = 0.0
        angular_vel = p_out + i_out
        if limit_speed:
            angular_vel = max(-self.turn_speed, min(self.turn_speed, angular_vel))
        
        # Apply minimum turn speed (rover won't respond to very small values)
        if angular_vel > 0 and angular_vel < self.min_turn_speed:
            angular_vel = self.min_turn_speed
        elif angular_vel < 0 and angular_vel > -self.min_turn_speed:
            angular_vel = -self.min_turn_speed
        
        # DEBUG: Control Output
        self.get_logger().info(f"  PID | Err: {error:6.1f} | P: {p_out:6.3f} | I: {i_out:6.3f} | CMD: {angular_vel:6.3f}")
        return angular_vel

    def set_state(self, new_state: str, reason: str = ""):
        if self.state != new_state:
            self.get_logger().warn(f"============== STATE CHANGE ==============")
            self.get_logger().warn(f"  {self.state}  --->  {new_state}")
            self.get_logger().warn(f"  Reason: {reason}")
            self.get_logger().warn(f"==========================================")
            self.state = new_state
            self.state_start_time = self.get_clock().now()
            self.integral_error = 0.0

    # =========================================================================================
    #                               CALLBACKS
    # =========================================================================================
    def camera_info_cb(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.image_center_x = msg.width / 2.0  # Center of left camera image
            self.camera_info_received = True
            self.get_logger().info(f" ✅ Camera Info Received. Width: {msg.width} (Center: {self.image_center_x})")

    def compass_cb(self, msg: Float64):
        self.current_heading = msg.data
        self.get_logger().info(f"🧭 Compass: {self.current_heading:5.1f} deg", throttle_duration_sec=1.0)

    def detections_cb(self, msg: ConeDetectionArray):
        if not msg.detections: 
            self.get_logger().info(" . No Detections", throttle_duration_sec=1.0)
            return
        
        # If no action is active, ignore all detections
        if not self.action_active:
            return
        
        best = min(msg.detections, key=lambda d: d.distance)
        count = len(msg.detections)
        
        self.get_logger().info(f"👁️ SEEN {count} cones. Best: {best.distance:.2f}m @ {best.cx:.0f}px (Color: {best.color})")

        if best.distance > self.max_dist: 
            self.get_logger().info(f"   -> IGNORED (Too Far > {self.max_dist}m)")
            return
        
        # CHECK COLOR MATCH: Only accept cones matching the goal color
        detected_color = best.color.lower().strip()
        if detected_color != self.goal_color:
            self.get_logger().info(
                f"   -> COLOR MISMATCH: Detected '{detected_color}' but looking for '{self.goal_color}'. Ignoring.",
                throttle_duration_sec=2.0
            )
            return
        
        # Color matches! Update detection and state
        self.last_detection = best
        self.last_valid_time = self.get_clock().now()
        
        self.get_logger().info(
            f"✅ TARGET CONE DETECTED: {self.goal_color.upper()} @ {best.distance:.2f}m"
        )
        
        # Only allow state interruption if we are in SEARCH mode
        if self.state == 'SEARCH':
            self.set_state('STOP_ON_DETECT', reason='target cone found')

    def send_cmd(self, linear, angular):
        t = Twist()
        t.linear.x = float(linear)
        t.angular.z = float(angular)
        self.cmd_pub.publish(t)
        self.get_logger().info(f"🚗 CMD | Lin: {linear:5.2f} | Ang: {angular:5.2f} | State: {self.state}")

    async def execute_cone_following(self, goal_handle):
        """
        Execute cone following action
        Goal contains: color (str), type (str - 'pickup'/'dropoff')
        Returns: success (bool), message (str)
        """
        self.goal_handle = goal_handle
        self.goal_color = goal_handle.request.color.lower()
        self.goal_type = goal_handle.request.type.lower()
        self.action_active = True
        
        self.get_logger().info(
            f"Cone Following Action Started: {self.goal_type.upper()} {self.goal_color.upper()}"
        )
        
        # Reset state machine for this action - START in SEARCH
        self.set_state('SEARCH', reason='Action started')
        
        # Create feedback message
        feedback_msg = FollowCone.Feedback()
        
        # Run the action until completion or cancellation
        while rclpy.ok() and self.action_active:
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Cone following action canceled")
                self.stop_all_motion()
                self.action_active = False
                self.state = 'IDLE'
                goal_handle.canceled()
                return FollowCone.Result(success=False, message="Canceled")
            
            # Publish feedback (distance to cone if detected)
            if self.last_detection:
                feedback_msg.distance_to_cone = self.last_detection.distance
            else:
                feedback_msg.distance_to_cone = float('inf')
            feedback_msg.status = self.state
            goal_handle.publish_feedback(feedback_msg)
            
            # Check for mission completion
            if self.state == 'MISSION_COMPLETE':
                self.action_active = False
                self.state = 'IDLE'
                goal_handle.succeed()
                self.get_logger().info(
                    f"✅ Cone following action succeeded: {self.goal_type.upper()} {self.goal_color.upper()}"
                )
                return FollowCone.Result(
                    success=True,
                    message=f"{self.goal_type} completed successfully"
                )
            
            # Small sleep to avoid busy-waiting
            time.sleep(0.1)
        
        # If we exit the loop, abort
        self.action_active = False
        self.state = 'IDLE'
        goal_handle.abort()
        return FollowCone.Result(success=False, message="Action aborted")

    # =========================================================================================
    #                               MAIN CONTROL LOOP
    # =========================================================================================
    def control_loop(self):
        """
        Main control loop implementing the state machine:
        
        SEARCH -> STOP_ON_DETECT -> ALIGN -> APPROACH -> STOP_AT_CONE -> TURN_180 -> BACKUP -> MISSION_COMPLETE
        
        Key behaviors:
        - SEARCH: Rotate in place to find cone
        - STOP_ON_DETECT: Stop immediately when cone detected
        - ALIGN: Stop and turn to align ZED center (width/4) to cone center
        - APPROACH: Move forward toward cone (re-align if needed by stopping first)
        - STOP_AT_CONE: Stop when within stop_distance
        - TURN_180: Subscribe to compass, perform 180° turn
        - BACKUP: Move backward for 2-3 seconds
        - MISSION_COMPLETE: Signal goal completed
        """
        if not self.action_active:
            # If no action is active, stay stopped
            self.stop_motion()
            return
        
        now = self.get_clock().now()
        time_in_state = (now - self.state_start_time).nanoseconds / 1e9
        time_since_last_detection = (now - self.last_valid_time).nanoseconds / 1e9
        
        # Publish status
        status_msg = String()
        status_msg.data = self.state
        self.status_pub.publish(status_msg)
        
        # ========================== STATE: SEARCH ==========================
        if self.state == 'SEARCH':
            # Rotate in place to search for cone
            self.send_cmd(0.0, self.turn_speed)
            # Transition handled in detections_cb when cone found
        
        # ========================== STATE: STOP_ON_DETECT ==========================
        elif self.state == 'STOP_ON_DETECT':
            # Stop immediately upon detection, let rover settle
            self.send_cmd(0.0, 0.0)
            if time_in_state > self.stop_settle:
                self.set_state('ALIGN', reason='stopped, now aligning to cone')
        
        # ========================== STATE: ALIGN ==========================
        elif self.state == 'ALIGN':
            # Stop and turn to align ZED center to cone center
            # Use grace_period for ALIGN (shorter tolerance for cone visibility)
            if self.last_detection is None or time_since_last_detection > self.grace_period:
                self.set_state('SEARCH', reason=f'lost cone during alignment ({time_since_last_detection:.1f}s > {self.grace_period}s)')
                return
            
            error_x = self.last_detection.cx - self.image_center_x
            self.get_logger().info(f"📐 ALIGN | Cone CX: {self.last_detection.cx:.1f} | Target: {self.image_center_x:.1f} | Error: {error_x:.1f}px | Age: {time_since_last_detection:.2f}s")
            
            if abs(error_x) <= self.alignment_tol:
                # Aligned! Transition to approach
                self.send_cmd(0.0, 0.0)
                self.set_state('APPROACH', reason=f'aligned (error={error_x:.1f}px)')
            else:
                # Stop and turn to align (do NOT move forward while aligning)
                angular_cmd = self.get_steering_cmd(error_x, use_integral=False)
                self.send_cmd(0.0, angular_cmd)
        
        # ========================== STATE: APPROACH ==========================
        elif self.state == 'APPROACH':
            # Move forward toward cone with angular corrections
            # Use grace_period for APPROACH (shorter tolerance for cone visibility)
            if self.last_detection is None or time_since_last_detection > self.grace_period:
                self.set_state('SEARCH', reason=f'lost cone during approach ({time_since_last_detection:.1f}s > {self.grace_period}s)')
                return
            
            distance = self.last_detection.distance
            error_x = self.last_detection.cx - self.image_center_x
            
            self.get_logger().info(f"🎯 APPROACH | Dist: {distance:.2f}m | Error: {error_x:.1f}px | Age: {time_since_last_detection:.2f}s")
            
            # Check if we've reached the cone
            if distance <= self.stop_dist:
                self.send_cmd(0.0, 0.0)
                self.set_state('STOP_AT_CONE', reason=f'reached stop distance ({distance:.2f}m <= {self.stop_dist}m)')
                return
            
            # Only transition to ALIGN if SEVERELY off-center (beyond outer_band)
            if abs(error_x) > self.outer_band:
                # Too far off, stop immediately and transition to ALIGN state
                self.send_cmd(0.0, 0.0)
                self.set_state('ALIGN', reason=f'off-center during approach (error={error_x:.1f}px > {self.outer_band}px)')
            else:
                # Move forward with proportional angular corrections
                # This prevents the stop-rotate-move oscillation
                angular_cmd = 0.0
                if abs(error_x) > self.alignment_tol:
                    # Apply proportional correction while moving
                    angular_cmd = error_x * self.kp
                    # Clamp angular command
                    angular_cmd = max(-self.turn_speed, min(self.turn_speed, angular_cmd))
                
                self.send_cmd(self.fwd_speed, angular_cmd)
        
        # ========================== STATE: STOP_AT_CONE ==========================
        elif self.state == 'STOP_AT_CONE':
            # Stop and prepare for 180° turn
            self.send_cmd(0.0, 0.0)
            
            if time_in_state > self.stop_settle:
                # Record current heading and calculate target (180° opposite)
                if self.current_heading is not None:
                    self.target_heading = self.normalize_heading(self.current_heading + 180.0)
                    # Lock in turn direction: always turn RIGHT (negative angular velocity)
                    # This prevents the rover from oscillating on which way to turn
                    self.turn_direction = -1  # -1 = clockwise/right, +1 = counter-clockwise/left
                    self.get_logger().info(f"🔄 Starting 180° turn: Current={self.current_heading:.1f}° -> Target={self.target_heading:.1f}° (Direction: RIGHT)")
                    self.set_state('TURN_180', reason=f'starting 180° turn to {self.target_heading:.1f}°')
                else:
                    self.get_logger().warn("⚠️ No compass heading available! Waiting for compass data...")
        
        # ========================== STATE: TURN_180 ==========================
        elif self.state == 'TURN_180':
            # Perform 180° turn using compass heading
            if self.current_heading is None:
                self.get_logger().warn("⚠️ Lost compass heading! Stopping...")
                self.send_cmd(0.0, 0.0)
                return
            
            if self.target_heading is None:
                self.get_logger().error("❌ No target heading set! This should not happen.")
                self.send_cmd(0.0, 0.0)
                return
            
            heading_error = self.heading_diff(self.current_heading, self.target_heading)
            self.get_logger().info(f"🔄 TURN_180 | Current: {self.current_heading:.1f}° | Target: {self.target_heading:.1f}° | Error: {heading_error:.1f}°")
            
            # Check if turn is complete
            if abs(heading_error) <= self.heading_tol:
                self.send_cmd(0.0, 0.0)
                self.get_logger().info(f"✅ 180° turn complete! Final heading: {self.current_heading:.1f}°")
                self.backup_start_time = self.get_clock().now()
                self.set_state('BACKUP', reason='180° turn completed')
            elif time_in_state > self.turn_timeout:
                # Timeout, move on anyway
                self.send_cmd(0.0, 0.0)
                self.get_logger().warn(f"⚠️ 180° turn timeout! Current heading: {self.current_heading:.1f}° (target was {self.target_heading:.1f}°)")
                self.backup_start_time = self.get_clock().now()
                self.set_state('BACKUP', reason='180° turn timeout')
            else:
                # Use locked turn direction to prevent direction flip-flopping
                # Speed is proportional to how far we are from target, but direction is fixed
                turn_speed = min(abs(heading_error) * self.turn_kp * 2.0, self.max_turn_speed)
                turn_speed = max(turn_speed, self.min_turn_speed)  # Apply minimum
                
                # Apply the locked direction
                turn_cmd = self.turn_direction * turn_speed
                
                self.get_logger().info(f"  TURN | Error: {heading_error:.1f}° | Dir: {self.turn_direction} | Speed: {turn_speed:.2f} | CMD: {turn_cmd:.2f}")
                
                # Stop and turn (no forward motion)
                self.send_cmd(0.0, turn_cmd)
        
        # ========================== STATE: BACKUP ==========================
        elif self.state == 'BACKUP':
            # Move backward for 2-3 seconds
            if self.backup_start_time is None:
                self.backup_start_time = self.get_clock().now()
            
            backup_elapsed = (now - self.backup_start_time).nanoseconds / 1e9
            self.get_logger().info(f"⬅️ BACKUP | Elapsed: {backup_elapsed:.1f}s / {self.backup_dur}s")
            
            if backup_elapsed >= self.backup_dur:
                self.send_cmd(0.0, 0.0)
                self.set_state('MISSION_COMPLETE', reason=f'backup complete ({backup_elapsed:.1f}s)')
            else:
                # Move backward (negative speed)
                self.send_cmd(self.backup_speed, 0.0)
        
        # ========================== STATE: MISSION_COMPLETE ==========================
        elif self.state == 'MISSION_COMPLETE':
            # Stop all motion, action server will handle the rest
            self.send_cmd(0.0, 0.0)
            self.get_logger().info("🎉 MISSION COMPLETE! Goal achieved.")
            # Action completion is handled in execute_cone_following


def main(args=None):
    rclpy.init(args=args)
    node = ConeFollowerNode()
    
    # Use multi-threaded executor for action server
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n\n!!! KEYBOARD INTERRUPT DETECTED !!!")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
    finally:
        # EXECUTE SAFETY STOP NO MATTER WHAT
        print("INITIATING SAFETY STOP SEQUENCE...")
        node.stop_all_motion()
        node.destroy_node()
        rclpy.shutdown()
        print("SAFETY STOP COMPLETE. SHUTDOWN.")

if __name__ == '__main__':
    main()
