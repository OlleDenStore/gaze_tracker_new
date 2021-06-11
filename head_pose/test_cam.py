# First import the library
import pyrealsense2 as rs
import cv2
import numpy as np

# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

colorizer = rs.colorizer()

while True:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    colorizer.set_option(rs.option.color_scheme, 2);
    colorized_depth = np.asanyarray(colorizer.colorize(depth).get_data())
    cv2.imshow('depth', colorized_depth)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all the windows 
cv2.destroyAllWindows()