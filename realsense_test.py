#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path


try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    rs.config.enable_device_from_file(config, 'eye_gaze/Data/videos/eval.bag')


    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    # Align objects
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)
        

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.


    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution

    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming from file


    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    #colorizer = rs.colorizer()

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Get depth frame
        #depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        #depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        #depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        
        color_frame = np.asanyarray(color_frame.get_data())
        cv2.imshow("Stream", cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB))
        #cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pass