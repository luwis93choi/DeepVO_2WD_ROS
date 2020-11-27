import Pyro4
import base64

import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv

import time

@Pyro4.expose
class Python2_Server(object):
    def response(self, data):

        decoded_string = np.fromstring(base64.b64decode(data), np.uint8)    # Decode base64 encoding of RX data into unsigned int array
        decoded_img = cv.imdecode(decoded_string, cv.IMREAD_COLOR)          # Restore decoded array into image

        cv.namedWindow('Python2 Server Realsene Img', cv.WINDOW_AUTOSIZE)
        cv.imshow('Python2 Server Realsene Img', decoded_img)
        cv.waitKey(1)

        return '[Server] RX Time : {}'.format(time.time())

daemon = Pyro4.Daemon()
uri = daemon.register(Python2_Server)

print('Ready Python2 Server : {}'.format(uri))

daemon.requestLoop()