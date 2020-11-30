/*
 * Arduino IDE Linux Setup + Serial Port Permission Setting : https://www.arduino.cc/en/Guide/Linux
 * 
 * rosserial_arduino setup : http://wiki.ros.org/rosserial_arduino/Tutorials/Arduino%20IDE%20Setup
 * 
 * Arduino Due sync issue : https://github.com/ros-drivers/rosserial/issues/284#issuecomment-312502805
 *                        : http://wiki.ros.org/rosserial_arduino --> 1. Special defines
 *                        : https://answers.ros.org/question/264025/rosserial-examples-for-arduino/
 *                        : https://answers.ros.org/question/264764/rosserial-arduino-due-sync-issues/?answer=265298#post-id-265298
 * 
 * rosserial_arduino genpy issue : https://answers.ros.org/question/360537/unknown-error-handler-name-rosmsg/
 */

#define USE_USBCON  // Add this in order to support Arduino Due USB Communication
                    // This is for ARM-based Arduino boards with separate programming port and native USB port
#include <ros.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

int dc1_enable = 35;
int dc1_directionA = 32;
int dc1_directionB = 34;
int dc1_pwm = 2;

int dc2_enable = 25;
int dc2_directionA = 24;
int dc2_directionB = 22;
int dc2_pwm = 3;

//*** ROS Subscriber callback function ***//

// Right DC Motor Speed Control Callback Function
void dc1_speed_ctrl_callback(const std_msgs::Int32& dc1_speed_msgs){

  analogWrite(dc1_pwm, dc1_speed_msgs.data);    // Apply Int32 data as PWM signal directly into right motor PWM pin
}

// Right DC Motor Direction Control Callback Function
void dc1_direction_ctrl_callback(const std_msgs::Int32& dc1_direction_msgs){

  if(dc1_direction_msgs.data == 0){

    // Stop the motor
    digitalWrite(dc1_enable, LOW);        // Disable the motor
    digitalWrite(dc1_directionA, LOW);    // Neutralize H-bridge
    digitalWrite(dc1_directionB, LOW);
  }
  else if(dc1_direction_msgs.data == 1){

    // Forward the motor
    digitalWrite(dc1_enable, HIGH);       // Enable the motor
    digitalWrite(dc1_directionA, LOW);    // Activate H-brige in forward motion
    digitalWrite(dc1_directionB, HIGH); 
  }
  else if(dc1_direction_msgs.data == -1){

    // Reverse the the motor
    digitalWrite(dc1_enable, HIGH);       // Enable the motor
    digitalWrite(dc1_directionA, HIGH);   // Activate H-bridge in reverse motion
    digitalWrite(dc1_directionB, LOW); 
  }
}

// Left DC Motor Speed Control Callback Function
void dc2_speed_ctrl_callback(const std_msgs::Int32& dc2_speed_msgs){

  analogWrite(dc2_pwm, dc2_speed_msgs.data);  // Apply Int32 data as PWM signal directly into left motor PWM pin
}

// Left DC Motor Direction Control Callback Function
void dc2_direction_ctrl_callback(const std_msgs::Int32& dc2_direction_msgs){

  if(dc2_direction_msgs.data == 0){
    
    // Stop the motor
    digitalWrite(dc2_enable, LOW);          // Disable the motor
    digitalWrite(dc2_directionA, LOW);      // Neutralize H-bridge
    digitalWrite(dc2_directionB, LOW);
  }
  else if(dc2_direction_msgs.data == 1){
     
    // Forward the motor
    digitalWrite(dc2_enable, HIGH);         // Enable the motor
    digitalWrite(dc2_directionA, HIGH);     // Activate H-brige in forward motion
    digitalWrite(dc2_directionB, LOW); 
  }
  else if(dc2_direction_msgs.data == -1){
    
    // Reverse the the motor
    digitalWrite(dc2_enable, HIGH);         // Enable the motor
    digitalWrite(dc2_directionA, LOW);      // Activate H-bridge in reverse motion
    digitalWrite(dc2_directionB, HIGH);
  }
}

// ROS Node / Subscriber Init
ros::NodeHandle nodeHandler;

ros::Subscriber<std_msgs::Int32> dc1_speed_ctrl("dc1_speed", &dc1_speed_ctrl_callback);               // Right motor speed control subscriber
ros::Subscriber<std_msgs::Int32> dc1_direction_ctrl("dc1_direction", &dc1_direction_ctrl_callback);   // Right motor direction control subscriber

ros::Subscriber<std_msgs::Int32> dc2_speed_ctrl("dc2_speed", &dc2_speed_ctrl_callback);               // Left motor speed control subscriber
ros::Subscriber<std_msgs::Int32> dc2_direction_ctrl("dc2_direction", &dc2_direction_ctrl_callback);   // Left motor direction control subscriber

void setup() {

  // Motor 1 Init
  pinMode(dc1_enable, OUTPUT);
  pinMode(dc1_directionA, OUTPUT);
  pinMode(dc1_directionB, OUTPUT);
  pinMode(dc1_pwm, OUTPUT);

  digitalWrite(dc1_enable, LOW);
  digitalWrite(dc1_directionA, LOW);
  digitalWrite(dc1_directionB, LOW);
  analogWrite(dc1_pwm, 0);

  // Motor 2 Init
  pinMode(dc2_enable, OUTPUT);
  pinMode(dc2_directionA, OUTPUT);
  pinMode(dc2_directionB, OUTPUT);
  pinMode(dc2_pwm, OUTPUT);
  
  digitalWrite(dc2_enable, LOW);
  digitalWrite(dc2_directionA, LOW);
  digitalWrite(dc2_directionB, LOW);
  analogWrite(dc2_pwm, 0);

  // ROS Node / Serial Comm Setup
  nodeHandler.getHardware()->setBaud(57600);    // Set up baurdrate of serial communication before node initialization
  nodeHandler.initNode();

  nodeHandler.subscribe(dc1_speed_ctrl);
  nodeHandler.subscribe(dc1_direction_ctrl);
  
  nodeHandler.subscribe(dc2_speed_ctrl);
  nodeHandler.subscribe(dc2_direction_ctrl);
}

void loop() {

  nodeHandler.spinOnce();
  delay(1);
}
