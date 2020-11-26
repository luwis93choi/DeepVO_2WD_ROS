#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import Pyro4
import base64

import numpy as np
import cv2 as cv

import time

import threading

decoded_img = None
pyro4_status = 'OFF'

@Pyro4.expose
class Python2_Server(object):
    def response(self, data):

        global decoded_img

        decoded_string = np.fromstring(base64.b64decode(data), np.uint8)
        decoded_img = cv.imdecode(decoded_string, cv.IMREAD_COLOR)

        return '[Server] RX Time : {}'.format(time.time())

def create_pyro4Server():

    global pyro4_status

    pyro4_status = 'ON'
    pyroDaemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    uri = pyroDaemon.register(Python2_Server)
    ns.register('realsense_ROSNode', uri)

    pyroDaemon.requestLoop()

def talker():

    global decoded_img
    global pyro4_status

    bridge = CvBridge()

    pub = rospy.Publisher('realsense_img', Image, queue_size=10)
    rospy.init_node('realsenseROSNode', anonymous=True)
    rate = rospy.Rate(30)

    # Run Pyro4 server as an independent Daemon thread
    # Run it as Daemon thread in order to make it shut down when the main thread dies
    pyro4ServerThread = threading.Thread(target=create_pyro4Server, args=())
    pyro4ServerThread.daemon = True
    pyro4ServerThread.start()       # Detach and run the thread

    while not rospy.is_shutdown():

        print('pyro4Server status : {}'.format(pyro4_status))

        if decoded_img is not None:
            image_msgs = bridge.cv2_to_imgmsg(decoded_img, 'bgr8')
            pub.publish(image_msgs)

            rospy.loginfo('realsenseROSNode img publish : {}'.format(time.time()))


        rate.sleep()

if __name__ == '__main__':
    
    try:
        talker()

    except rospy.ROSInterruptException:
        pass