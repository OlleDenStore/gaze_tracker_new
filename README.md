# gaze

ROS node for gaze tracking in ros_stuff/gaze_track.py. Subscribes to aligned rgb and depth images on /camera/color/image_raw and /camera/aligned_depth_to_color/image_raw topics and needs camera information from /camera/aligned_depth_to_color/camera_info topic.

Publishes gaze direction as sensor_msgs/Imu message (Vector3, but easily visualized in Rviz) to /gaze topic (along with the head and eye gaze individually to /gazeH and /gazeE.

gaze tracker uses python 3. Depends on cv2, which when installed with apt is not compatible with python3 - you need to build it yourself, e.g:
```
cd src
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
```
```
catkin config -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
```
```
catkin build
```
