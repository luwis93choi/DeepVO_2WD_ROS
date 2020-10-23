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
            self.sequence_in_use_idx = 0
            self.idx_in_sequence = 0
            self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_in_use_idx] + '/image_2')))
            self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_in_use_idx] + '/image_2'))

        else:
            self.sequence_in_use_idx = 0
            self.idx_in_sequence = 0
            self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_in_use_idx] + '/image_2')))
            self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_in_use_idx] + '/image_2'))

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

        if self.test is False:

            if self.idx_in_sequence >= self.current_sequence_data_num:
                
                self.sequence_in_use_idx += 1
                
                self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_in_use_idx] + '/image_2')))
                self.idx_in_sequence = 0

                self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.train_sequence[self.sequence_in_use_idx] + '/image_2'))

            base_path = self.img_dataset_path + '/' + self.train_sequence[self.sequence_in_use_idx] + '/image_2'

            Image.open(base_path + '/' + self.img_path[self.idx_in_sequence]).convert('RGB')

            print(self.current_sequence_data_num)
            print(base_path + '/' + self.img_path[self.idx_in_sequence])

            self.idx_in_sequence += 1

        else:

            if self.idx_in_sequence >= self.current_sequence_data_num:
                
                self.sequence_in_use_idx += 1
                
                self.current_sequence_data_num = len(sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_in_use_idx] + '/image_2')))
                self.idx_in_sequence = 0

                self.img_path = sorted(os.listdir(self.img_dataset_path + '/' + self.test_sequence[self.sequence_in_use_idx] + '/image_2'))

            base_path = self.img_dataset_path + '/' + self.test_sequence[self.sequence_in_use_idx] + '/image_2'

            Image.open(base_path + '/' + self.img_path[self.idx_in_sequence]).convert('RGB')

            self.idx_in_sequence += 1

        return 0

    def __len__(self):

        return self.len