from deepvoNet import DeepVONet
from dataloader import voDataLoader

from model_trainer import trainer

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

import datetime
import numpy as np
from matplotlib import pyplot as plt

img_dataset_path = '/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences'
pose_dataset_path = '/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses'

train_epoch = 2
train_sequence = ['01']
#train_sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
test_sequence = ['01']

normalize = transforms.Normalize(
    #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
    mean=[127. / 255., 127. / 255., 127. / 255.],
    std=[1 / 255., 1 / 255., 1 / 255.]
)

preprocess = transforms.Compose([
    transforms.Resize((384, 1280)),
    transforms.CenterCrop((384, 1280)),
    transforms.ToTensor(),
    normalize
])

deepvo_trainer = trainer(use_cuda=True,
                         loader_preprocess_param=preprocess,
                         img_dataset_path=img_dataset_path,
                         pose_dataset_path=pose_dataset_path,
                         train_epoch=2, train_sequence=train_sequence, train_batch=1,
                         plot_batch=False, plot_epoch=True)

deepvo_trainer.train()
