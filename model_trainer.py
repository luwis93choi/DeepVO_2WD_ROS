from deepvoNet import DeepVONet
from dataloader import voDataLoader

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

import datetime
import numpy as np
from matplotlib import pyplot as plt

class trainer():

    def __init__(self, use_cuda=True, 
                       loader_preprocess_param=transforms.Compose([]), 
                       train_epoch=1,
                       train_sequence=['01'],
                       plot_batch=False, plot_epoch=True):

        if use_cuda == True:        
            # Load main processing unit for neural network
            PROCESSOR = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        deepvo_model = DeepVONet()
        deepvo_model.to(PROCESSOR)


        if str(PROCESSOR) == 'cuda:0':
            deepvo_model.use_cuda = True
            deepvo_model.reset_hidden_states(size=1, zero=True)

        deepvo_model.train()
        deepvo_model.training = True

        train_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences',
                                                                pose_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses',
                                                                transform=loader_preprocess_param,
                                                                sequence=train_sequence),
                                                                batch_size=1, shuffle=True, drop_last=True)

        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(deepvo_model.parameters(), lr=0.0001)

        summary(deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(PROCESSOR)))

        train_plot_color = plt.cm.get_cmap('rainbow', len(train_sequence))
        train_plot_x = 0

        draw_broken_yaxis = False

        if plot_batch == True:

            ### Plotting graph setup with broken y-axis ######################################
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
            ax1.set_ylim(2, 30)
            ax2.set_ylim(0, 1.5)

            ax1.spines['bottom'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            ax1.xaxis.tick_top()
            ax1.tick_params(labeltop=False)
            ax2.xaxis.tick_bottom()

            d = .015    # how big to make the diagonal lines in axes coordinates
                        # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
            ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
            ################################################################################
