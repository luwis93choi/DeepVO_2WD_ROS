#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge

import Pyro4
import base64

import numpy as np
import cv2 as cv

import time

realsense_img_msgs = None
deepVO_Node = None

bridge = None
rx_img = None

def callback(realsense_img_msgs):

    global bridge
    global rx_img

    global deepVO_Node

    rospy.loginfo('Realsense Image Received')

    rx_img_np_arr = np.fromstring(realsense_img_msgs.data, np.uint8)
    rx_img = cv.imdecode(rx_img_np_arr, cv.IMREAD_COLOR)

    # cv.namedWindow('Python2 Server Realsene Img', cv.WINDOW_AUTOSIZE)
    # cv.imshow('Python2 Server Realsene Img', rx_img)
    # cv.waitKey(1)

    retval, buffer = cv.imencode('.jpg', rx_img)
    TX_data = base64.b64encode(buffer)

    deepVO_Node.response(TX_data.decode('utf-8'))

def realsenseROSNode_RX():

    global realsense_img_msgs
    global deepVO_Node

    global bridge

    bridge = CvBridge()

    deepVO_Node = Pyro4.Proxy('PYRONAME:deepVO_Node')

    rospy.init_node('realsenseROSNode_RX', anonymous=True)
    
    rospy.Subscriber('compressed_realsense_img', CompressedImage, callback, queue_size=1)

    rospy.loginfo('realsenseROSNode_RX Ready')

    #deepVO_Node = Pyro4.Proxy('PYRONAME:deepVO_Node')

    rospy.spin()

if __name__ == '__main__':

    realsenseROSNode_RX()
