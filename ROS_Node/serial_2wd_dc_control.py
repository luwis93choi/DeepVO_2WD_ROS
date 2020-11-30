#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Int32

dc1_speed_control_msgs = Int32()
dc1_direction_control_msgs = Int32()

dc2_speed_control_msgs = Int32()
dc2_direction_control_msgs = Int32()

dc1_current_pwm = 0
dc1_current_direction = 0

dc2_current_pwm = 0
dc2_current_direction = 0

def dc1_speed_update(current_pwm):

    global dc1_current_pwm
    dc1_current_pwm = current_pwm.data

def dc1_direction_update(current_direction):

    global dc1_current_direction
    dc1_current_direction = current_direction.data

def dc2_speed_update(current_pwm):

    global dc2_current_pwm
    dc2_current_pwm = current_pwm.data

def dc2_direction_update(current_direction):

    global dc2_current_direction
    dc2_current_direction = current_direction.data

def dc_control_callback(control_msgs):

    global dc1_speed_control_msgs
    global dc1_direction_control_msgs

    global dc2_speed_control_msgs
    global dc2_direction_control_msgs

    global dc1_current_pwm
    global dc2_current_pwm

    if control_msgs.buttons[4] == 1:

        if dc1_current_direction == 1:

            rospy.loginfo('Stop Left Motor')
            dc1_speed_control_msgs.data = 0
            dc1_direction_control_msgs.data = 0

        else:

            rospy.loginfo('Activate Left Motor')
            dc1_speed_control_msgs.data = 0
            dc1_direction_control_msgs.data = 1
    else:

        if control_msgs.buttons[6] == 1:

            dc1_direction_control_msgs.data = dc1_current_direction * -1
            rospy.loginfo('Change Left Motor Direction : {}'.format(dc1_direction_control_msgs.data))

        if control_msgs.axes[1] >= 0.2:
            
            if dc1_current_pwm < 200:
                
                dc1_speed_control_msgs.data = dc1_current_pwm + 10
            
            else:
                
                dc1_speed_control_msgs.data = dc1_current_pwm

            rospy.loginfo('Increase Left Motor Speed : {}'.format(dc1_speed_control_msgs.data))

        elif control_msgs.axes[1] <= -0.2:

            if dc1_current_pwm > 0:
                
                dc1_speed_control_msgs.data = dc1_current_pwm - 10
            
            else:
                
                dc1_speed_control_msgs.data = dc1_current_pwm
                
            rospy.loginfo('Decrease Left Motor Speed : {}'.format(dc1_speed_control_msgs.data))

    if control_msgs.buttons[5] == 1:

        if dc2_current_direction == 1:

            rospy.loginfo('Stop Right Motor')
            dc2_speed_control_msgs.data = 0
            dc2_direction_control_msgs.data = 0

        else:

            rospy.loginfo('Activate Right Motor')
            dc2_speed_control_msgs.data = 0
            dc2_direction_control_msgs.data = 1

    else:

        if control_msgs.buttons[7] == 1:

            dc2_direction_control_msgs.data = dc2_current_direction * -1
            rospy.loginfo('Change Right Motor Direction : {}'.format(dc2_direction_control_msgs.data))

        if control_msgs.axes[3] >= 0.2:
            
            if dc2_current_pwm < 200:
                
                dc2_speed_control_msgs.data = dc2_current_pwm + 10
            
            else:
                
                dc2_speed_control_msgs.data = dc2_current_pwm

            rospy.loginfo('Increase Right Motor Speed : {}'.format(dc2_speed_control_msgs.data))

        elif control_msgs.axes[3] <= -0.2:

            if dc2_current_pwm > 0:
                
                dc2_speed_control_msgs.data = dc2_current_pwm - 10
            
            else:
                
                dc2_speed_control_msgs.data = dc2_current_pwm

            rospy.loginfo('Decrease Right Motor Speed : {}'.format(dc2_speed_control_msgs.data))

def serial_2wd_dc_control():

    global dc1_speed_control_msgs
    global dc1_direction_control_msgs

    global dc2_speed_control_msgs
    global dc2_direction_control_msgs

    rospy.init_node('serial_2wd_dc_control', anonymous=True)
    rate = rospy.Rate(15)

    rospy.Subscriber('joy', Joy, dc_control_callback, queue_size=1)

    rospy.Subscriber('dc1_speed', Int32, dc1_speed_update, queue_size=1)
    rospy.Subscriber('dc1_direction', Int32, dc1_direction_update, queue_size=1)

    rospy.Subscriber('dc2_speed', Int32, dc2_speed_update, queue_size=1)
    rospy.Subscriber('dc2_direction', Int32, dc2_direction_update, queue_size=1)

    dc1_speed_pub = rospy.Publisher('dc1_speed', Int32, queue_size=1)
    dc1_direction_pub = rospy.Publisher('dc1_direction', Int32, queue_size=1)
    
    dc2_speed_pub = rospy.Publisher('dc2_speed', Int32, queue_size=1)
    dc2_direction_pub = rospy.Publisher('dc2_direction', Int32, queue_size=1)

    rospy.loginfo('Serial 2WD DC Control Ready')

    while not rospy.is_shutdown():

        dc1_speed_pub.publish(dc1_speed_control_msgs)
        dc1_direction_pub.publish(dc1_direction_control_msgs)

        dc2_speed_pub.publish(dc2_speed_control_msgs)
        dc2_direction_pub.publish(dc2_direction_control_msgs)
        
        rate.sleep()

if __name__ == '__main__':

    serial_2wd_dc_control()