import pyrealsense2 as rs
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

import Pyro4

import base64
import cv2 as cv

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

### Connection between Python processes of different versions
uri = input('Python2 Server URI?').strip()
python2Server = Pyro4.Proxy(uri)

while True:

    frames = pipeline.wait_for_frames()
    frame = frames.get_color_frame()

    print('Frame Ready')

    color_img = np.asanyarray(frame.get_data())

    print(color_img.shape)
    print(type(color_img))
    print(color_img.dtype)

    ### Send base64 compression results of Realsense Image
    retval, buffer = cv.imencode('.jpg', color_img)     # Encode image data using jpg codec
    TX_data = base64.b64encode(buffer)                  # Encode color image of realsense into base64

    print(python2Server.response(TX_data.decode('utf-8')))      # Send encoded image over Pyro4