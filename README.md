# Cone following and detection script I made for IRC'26 

## Dependencies? 
You'll need ultralytics, opencv-python and numpy<2 to run this,
> Note! This script is made to run with ZED 2i, if you're using a ZED then download the ZED Wrapper
```bash
pip install ultralytics opencv-python "numpy<2"
```

## How do I run this?

First you'll need to make two edits, 
1. In cone_detector_node.py change
```python
  self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
```
To whatever your image topic is

2. If your color encoding is not bgra8, change these lines to whatever encoding you're using
```python
  cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgra8')
  ...
  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
```

3. To run the node:
```bash
# in your package root, run:
colcon build --packages-select cone_detector cone_handling
source install/setup.bash

# Run these commands in seperate terminals
ros2 run cone_detector cone_detector_node

# In another terminal
ros2 run cone_handling cone_follower_node
