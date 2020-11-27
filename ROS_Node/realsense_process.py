import sys

sys.path.insert(0, '/usr/local/lib/python3.6')
print(sys.path)

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv

import pyrealsense2.pyrealsense2 as rs

import Pyro4
import base64

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

orb = cv.ORB_create()

# Recognize target Pyro4 server
realsenseROSNode = Pyro4.Proxy('PYRONAME:realsense_ROSNode')

try:
    while True:
        frames= pipeline.wait_for_frames()
        frame = frames.get_color_frame()

        print('Frame Ready')
        color_img = np.asanyarray(frame.get_data())

        # Encode realsense real-time data into base64 data
        retval, buffer = cv.imencode('.jpg', color_img)
        TX_data = base64.b64encode(buffer)

        # Send base64 encoded string of Image to target Pyro4 server
        realsenseROSNode.response(TX_data.decode('utf-8'))

finally:

	print('Realsense test complete')
