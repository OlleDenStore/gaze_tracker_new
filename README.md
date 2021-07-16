# gaze-srv

ROS service for simple robot control based on gaze direction. run service client in src/gaze_services/gaze_movebase_srv.py
calling
```
rosservice call /goto_gaze "empty: {}" 
```
will make make rbot using ROS navigation stack turn based on gaze direction
