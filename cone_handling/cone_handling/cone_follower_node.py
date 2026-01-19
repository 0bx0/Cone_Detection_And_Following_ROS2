#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
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
    'forward_speed':       0.25,   
    'turn_speed':          0.35,   
    'backup_speed':       -0.15,   

    # --- P-CONTROLLER FOR TURNING ---
    # kp = 0.02 means for every 1 degree of error, we turn at 0.02 rad/s
    # 180 deg error = 3.6 rad/s (capped at max)
    # 10 deg error = 0.2 rad/s
    'turn_kp':             0.02,   
    'max_turn_speed':      0.5,    
    'min_turn_speed':      0.15,   

    # --- DISTANCES ---
    'stop_distance':       1.2,    # UPDATED: Stops exactly at 2.2m
    'max_detection_dist':  50.0,   
    
    # --- 180 TURN CONFIG ---
    'heading_tolerance':   4.0,    # Slightly relaxed to prevent infinite hunting
    'settle_time':         1.0,    
    'turn_timeout':        20.0,   # Increased to ensure it finishes turning
    
    # --- ALIGNMENT ---
    'kp':                  0.003,  
    'ki':                  0.0001, 
    'ki_max':              0.1,    
    'inner_band':          20.0,   
    'outer_band':          120.0,  
    'final_align_tol':     20.0,   
    
    # --- TIMERS ---
    'lost_timeout':        2.0,    
    'grace_period':        0.5,    
    'backup_duration':     2.0,    
    'catchbox_delay':      3.0,    
    'retry_backoff_dur':   4.0,    
}
# =========================================================================================

class ConeFollowerNode(Node):
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
        self.image_center_x = self.fallback_width / 4.0
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
        
        self.inner_band = self.get_parameter('inner_band').value
        self.outer_band = self.get_parameter('outer_band').value
        self.final_tol = self.get_parameter('final_align_tol').value
        
        self.lost_timeout = self.get_parameter('lost_timeout').value
        self.grace_period = self.get_parameter('grace_period').value
        self.backup_dur = self.get_parameter('backup_duration').value
        self.catchbox_wait = self.get_parameter('catchbox_delay').value
        self.retry_backoff = self.get_parameter('retry_backoff_dur').value

        # State machine - start IDLE, wait for action goal
        self.state = 'IDLE'
        self.last_detection = None
        self.last_valid_time = self.get_clock().now()
        self.state_start_time = self.get_clock().now()
        self.integral_error = 0.0
        
        self.current_heading = None
        self.target_heading = None
        
        # Action Server State
        self.goal_handle = None
        self.goal_color = None
        self.goal_type = None
        self.action_active = False

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, '/auto/cone_follow/status', 10)
        
        self.create_subscription(ConeDetectionArray, self.detection_topic, self.detections_cb, 10)
        self.create_subscription(CameraInfo, self.cam_info_topic, self.camera_info_cb, 10)
        self.create_subscription(Float64, self.compass_topic, self.compass_cb, 10)
        
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
        
        # DEBUG: Control Output (High Frequency)
        self.get_logger().info(f"  PID | Err: {error:6.1f} | P: {p_out:6.3f} | I: {i_out:6.3f} | Integ: {self.integral_error:6.1f} | CMD: {angular_vel:6.3f}")
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
            self.image_center_x = msg.width / 4.0
            self.camera_info_received = True
            self.get_logger().info(f" ✅ Camera Info Received. Width: {msg.width} (Center: {self.image_center_x})")

    def compass_cb(self, msg: Float64):
        self.current_heading = msg.data
        # DEBUG: Reduced throttle to 1.0s to confirm connectivity
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
        
        # DEBUG: Very verbose detection info
        self.get_logger().info(f"👁️ SEEN {count} cones. Best: {best.distance:.2f}m @ {best.cx:.0f}px (Color: {best.color}) (Conf: {best.confidence:.2f})")

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
        # DEBUG: Actuator Output
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
    #                               MAIN LOOP
    # =========================================================================================
    def control_loop(self):
        # Only run control loop if an action goal is active
        if not self.action_active:
            return
        
        now = self.get_clock().now()

        # --- PHASE DEFINITIONS ---
        # VISUAL_PHASE: We need the camera. If we lose the cone, we search.
        VISUAL_PHASE = ['SEARCH', 'STOP_ON_DETECT', 'ALIGN_TO_INNER', 'DRIVE_FORWARD', 'FINAL_ALIGN']
        
        # BLIND_PHASE: We are Turning or Backing up. 
        # IGNORE CAMERA. DO NOT RESET. DO NOT SEARCH.
        BLIND_PHASE = ['TURN_AROUND', 'POST_TURN_SETTLE', 'BACK_UP', 'CATCHBOX', 'MISSION_COMPLETE', 'RETRY_BACKOFF']

        # --- SAFETY: COMPLETE STOP ---
        if self.state == 'MISSION_COMPLETE':
            self.send_cmd(0.0, 0.0)
            # Mission complete - action will handle state transition to IDLE
            return

        # --- PERSISTENCE CHECK (VISUAL ONLY) ---
        # We only care about losing the cone if we are actively driving towards it.
        if self.state in VISUAL_PHASE:
            elapsed_since_det = (now - self.last_valid_time).nanoseconds / 1e9
            is_blind = elapsed_since_det > self.grace_period
            is_lost = elapsed_since_det > self.lost_timeout
            
            if is_blind and not is_lost:
                self.send_cmd(0.0, 0.0)
                return
                
            if is_lost and self.state != 'SEARCH':
                self.get_logger().warn(f"Tracking Lost in Visual Phase ({elapsed_since_det:.1f}s). Resetting.")
                self.set_state('SEARCH', reason='timeout')
                self.last_detection = None

        # --- STATE MACHINE --- pls fix

        if self.state == 'SEARCH':
            self.send_cmd(0.0, self.turn_speed)

        elif self.state == 'STOP_ON_DETECT':
            self.send_cmd(0.0, 0.0)
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed > 1.0:
                self.set_state('ALIGN_TO_INNER')

        elif self.state == 'ALIGN_TO_INNER':
            if not self.last_detection: return
            err = self.image_center_x - self.last_detection.cx
            if abs(err) < self.inner_band:
                self.get_logger().info(f"ALIGN_SUCCESS | Err: {err:.1f} < Band: {self.inner_band}")
                self.set_state('DRIVE_FORWARD')
                return
            rot = self.get_steering_cmd(err, use_integral=True)
            self.send_cmd(0.0, rot)

        elif self.state == 'DRIVE_FORWARD':
            if not self.last_detection:
                self.send_cmd(0.0, 0.0)
                return
            
            # --- STOP DISTANCE TRIGGER ---
            if self.last_detection.distance <= self.stop_dist:
                self.get_logger().info(f"TARGET_REACHED | Dist: {self.last_detection.distance:.2f}m <= Stop: {self.stop_dist}m")
                self.send_cmd(0.0, 0.0)
                # TRANSITION TO BLIND PHASE -> Camera logic effectively ends here
                self.set_state('FINAL_ALIGN', reason='Target Reached')
                return
            
            err = self.image_center_x - self.last_detection.cx
            if abs(err) > self.outer_band:
                self.get_logger().warn(f"ALIGN_DRIFT | Err: {err:.1f} > Band: {self.outer_band}")
                self.set_state('ALIGN_TO_INNER', reason='drifted out')
                return
            rot = self.get_steering_cmd(err, use_integral=True)
            self.send_cmd(self.fwd_speed, rot)

        elif self.state == 'FINAL_ALIGN':
            if not self.last_detection: return
            err = self.image_center_x - self.last_detection.cx
            if abs(err) < self.final_tol:
                # Calculate target heading before transitioning
                if self.current_heading is not None:
                    self.target_heading = self.normalize_heading(self.current_heading + 180.0)
                    self.get_logger().info(f"Targeting {self.target_heading:.1f} (Current: {self.current_heading:.1f})")
                self.set_state('TURN_AROUND', reason='Perfectly Aligned, Starting 180')
                return
            rot = self.get_steering_cmd(err * 0.5, use_integral=False)
            rot = max(-0.1, min(0.1, rot)) 
            self.send_cmd(0.0, rot)

        # ========================================================
        #  BLIND PHASE: 180 Turn & Backup (Camera ignored here)
        # ========================================================

        elif self.state == 'TURN_AROUND':
            # Safety check - need compass data for turn
            if self.current_heading is None or self.target_heading is None:
                self.get_logger().warn('WAITING FOR COMPASS...', throttle_duration_sec=1.0)
                self.send_cmd(0.0, 0.0)
                return
            
            #logging the compass heading
            self.get_logger().info(f"🧭 TURNING | Current Heading: {self.current_heading:.1f} | Target Heading: {self.target_heading:.1f}")
            # TIMEOUT HANDLING
            # if elapsed > self.turn_timeout:
            #     self.get_logger().error("TURN FAILED (TIMEOUT). RETRYING...")
            #     self.set_state('RETRY_BACKOFF')
            #     return

            # P-CONTROLLER LOGIC
            diff = self.heading_diff(self.current_heading, self.target_heading)
            
            if abs(diff) <= self.heading_tol:
                self.get_logger().info(f"TURN COMPLETE | Diff: {diff:.2f} <= Tol: {self.heading_tol}")
                self.send_cmd(0.0, 0.0)
                self.set_state('POST_TURN_SETTLE', reason='180 Complete')
                return

            # Calc Speed: Error * Gain
            raw_rotation = diff * self.turn_kp
            
            # Clamp Speed
            abs_rot = abs(raw_rotation)
            if abs_rot > self.max_turn_speed: abs_rot = self.max_turn_speed
            if abs_rot < self.min_turn_speed: abs_rot = self.min_turn_speed
            
            # Direction Logic
            # 
            # Mavros (Compass) increases Clockwise. ROS (Twist) turns Counter-Clockwise.
            # We must INVERT the command to match frames.
            final_z_cmd = abs_rot * (1.0 if raw_rotation > 0 else -1.0)
            final_z_cmd = -final_z_cmd 
            
            self.get_logger().info(f"TURNING | Cur: {self.current_heading:.1f} | Tgt: {self.target_heading:.1f} | Diff: {diff:.1f} | Cmd: {final_z_cmd:.2f}", throttle_duration_sec=0.2)
            self.send_cmd(0.0, final_z_cmd)

        elif self.state == 'POST_TURN_SETTLE':
            self.send_cmd(0.0, 0.0)
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= self.settle_time:
                self.set_state('BACK_UP')

        elif self.state == 'BACK_UP':
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= self.backup_dur:
                self.send_cmd(0.0, 0.0)
                self.set_state('CATCHBOX')
                return
            # Simple blind reverse. No camera checks here.
            self.send_cmd(self.backup_speed, 0.0)

        elif self.state == 'CATCHBOX':
            self.send_cmd(0.0, 0.0)
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= self.catchbox_wait:
                self.stop_all_motion()
                self.status_pub.publish(String(data="SUCCESS"))
                self.get_logger().info("MISSION COMPLETE. HOLDING POSITION.")
                self.set_state('MISSION_COMPLETE')

        # Fallback for RETRY (only if turn times out)
        elif self.state == 'RETRY_BACKOFF':
            elapsed = (now - self.state_start_time).nanoseconds / 1e9
            if elapsed >= self.retry_backoff:
                self.send_cmd(0.0, 0.0)
                self.set_state('SEARCH', reason='Retry Reset')
                return
            self.send_cmd(-0.2, 0.0)

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
