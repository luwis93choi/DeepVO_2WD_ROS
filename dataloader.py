import os
import os.path
import numpy as np
import random
import math
import datetime

from PIL import Image   # Load images from the dataset

import torch.utils.data
import torchvision.transforms as tranforms

class voDataLoader(torch.utils.data.Dataset):

    def __init__(self, img_dataset_path, pose_dataset_path, 
                       transform=None, 
                       test=False, 
                       train_sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'], 
                       test_sequence=['01']):

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.train_sequence = train_sequence
        self.test_sequence = test_sequence

        self.test = test

        if self.test is False:
            self.sequence_idx = 0
            self.data_idx = 1        # Since DeepVO requires 2 consecutive images, Data Index starts from 1
            self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '/image_2')))
            self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '/image_2'))

            # Read 0th pose data and save it as current pose value
            self.pose_file = open(self.pose_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '.txt', 'r')
            line = self.pose_file.readline()
            pose = line.strip().split()
            self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
            self.current_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                               [float(pose[4]), float(pose[5]), float(pose[6])], 
                                               [float(pose[8]), float(pose[9]), float(pose[10])]])
            self.prev_pose_T = np.array([0.0, 0.0, 0.0])
            self.prev_pose_Rmat = np.array([0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0])

        else:
            self.sequence_idx = 0
            self.data_idx = 0
            self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '/image_2')))
            self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '/image_2'))

            # Read 0th pose data and save it as current pose value
            self.pose_file = open(self.pose_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '.txt', 'r')
            line = self.pose_file.readline()
            pose = line.strip().split()
            self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
            self.current_pose_Rmat = np.array([float(pose[0]), float(pose[1]), float(pose[2]), 
                                               float(pose[4]), float(pose[5]), float(pose[6]), 
                                               float(pose[8]), float(pose[9]), float(pose[10])])
            self.prev_pose_T = np.array([0.0, 0.0, 0.0])
            self.prev_pose_Rmat = np.array([0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0])

        self.len = 0    # The size of dataset in use

        if self.test is False:   # Count the number of data used in train dataset

            for i in range(len(self.train_sequence)):            
                
                f = open(self.pose_dataset_path + '/' + self.train_sequence[i] + '.txt', 'r')
                while True:
                    line = f.readline()
                    if not line: break

                    self.len += 1

            print('Train Sequence : {}'.format(self.train_sequence))
            print('Size of training dataset : {}'.format(self.len))

        else:   # Count the number of data used in test dataset

            for i in range(len(self.test_sequence)):            
                
                f = open(self.pose_dataset_path + '/' + self.test_sequence[i] + '.txt', 'r')
                while True:
                    line = f.readline()
                    if not line: break

                    self.len += 1

            print('Test Sequence : {}'.format(self.test_sequence))
            print('Size of test dataset : {}'.format(self.len))

    def __getitem__(self, index):
        
        # index is dummy value for pytorch
        
        # 'sequence_idx' and 'data_idx' are actual indices of dataset for training and testing.
        # Since the dataset is composed of multiple separate dataset sequences, sequence and data need their own indexing.
        
        # 'sequence_idx' is the index of KITTI dataset sequence in training or testing -> Dataset in Use
        # 'data_idx' is the index of data in the KITTI dataset sequence in training or testing -> Actual Data
        
        # In training mode
        if self.test is False:

            # Index reset if the index of data goes over the range of current dataset sequence
            if self.data_idx >= self.current_sequence_data_num:
                
                self.sequence_idx += 1
                
                self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '/image_2')))
                self.data_idx = 1

                self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '/image_2'))

                self.pose_file.close()
                self.pose_file = open(self.pose_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '.txt', 'r')

            ### Dataset Image Preparation ###
            base_path = self.img_dataset_path + '/' + self.train_sequence[self.sequence_idx] + '/image_2'

            prev_img = Image.open(base_path + '/' + self.img_path[self.data_idx-1]).convert('RGB')
            current_img = Image.open(base_path + '/' + self.img_path[self.data_idx]).convert('RGB')

            ### Pose Data (Pose difference/change between t-1 and t) Preparation ###

            # Save previous groundtruth as t-1 value
            self.prev_pose_T = self.current_pose_T
            self.prev_pose_Rmat = self.current_pose_Rmat

            # Load groundtruth at t
            line = self.pose_file.readline()
            pose = line.strip().split()
            self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
            self.current_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                               [float(pose[4]), float(pose[5]), float(pose[6])], 
                                               [float(pose[8]), float(pose[9]), float(pose[10])]])

            # Convert rotation matrix of groundtruth into euler angle
            prev_roll = math.atan2(self.prev_pose_Rmat[2][1], self.prev_pose_Rmat[2][2])
            prev_pitch = math.atan2(-1 * self.prev_pose_Rmat[2][0], math.sqrt(self.prev_pose_Rmat[2][1]**2 + self.prev_pose_Rmat[2][2]**2))
            prev_yaw = math.atan2(self.prev_pose_Rmat[1][0], self.prev_pose_Rmat[0][0])

            print(self.prev_pose_Rmat)
            print('Prev Roll {}, Prevl Pitch {}, Prev_Yaw {}'.format(prev_roll, prev_pitch, prev_yaw))
            print('-----------------------------------------')
            # Compute the difference between groundtruth at t-1 and t

            # Compute translation difference between groundtruth at t-1 and t
            
            # Prepare 6 DOF pose vector (X Y Z Roll Pitch Yaw)

            #########################################################################

        # In test mode
        else:
            
            # Index reset if the index of data goes over the range of current dataset sequence
            if self.data_idx >= self.current_sequence_data_num:
                
                self.sequence_idx += 1
                
                self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '/image_2')))
                self.data_idx = 1

                self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '/image_2'))

                self.pose_file.close()
                self.pose_file = open(self.pose_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '.txt', 'r')

            ### Dataset Image Preparation ###
            base_path = self.img_dataset_path + '/' + self.test_sequence[self.sequence_idx] + '/image_2'

            prev_img = Image.open(base_path + '/' + self.img_path[self.data_idx-1]).convert('RGB')
            current_img = Image.open(base_path + '/' + self.img_path[self.data_idx]).convert('RGB')

            ### Pose Data (Pose difference/change between t-1 and t) Preparation ###

            # Save previous groundtruth as t-1 value
            self.prev_pose_T = self.current_pose_T
            self.prev_pose_Rmat = self.current_pose_Rmat

            # Load groundtruth at t
            line = self.pose_file.readline()
            pose = line.strip().split()
            self.current_pose_T = np.array([float(pose[3]), float(pose[7]), float(pose[11])])
            self.current_pose_Rmat = np.array([[float(pose[0]), float(pose[1]), float(pose[2])], 
                                               [float(pose[4]), float(pose[5]), float(pose[6])], 
                                               [float(pose[8]), float(pose[9]), float(pose[10])]])

            # Convert rotation matrix of groundtruth into euler angle

            # Compute the euler angle difference between groundtruth at t-1 and t

            # Compute translation difference between groundtruth at t-1 and t

            # Prepare 6 DOF pose vector (X Y Z Roll Pitch Yaw)
            
            #########################################################################

        print(self.current_sequence_data_num)
        print(base_path + '/' + self.img_path[self.data_idx])

        self.data_idx += 1   # Increase data index everytime data is consumed by DeepVO network

        # Stack the image as indicated in DeepVO paper
        prev_current_stacked_img = np.asarray(np.concatenate([prev_img, current_img], axis=0))

        return prev_current_stacked_img

    def __len__(self):

        return self.len