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
                       img_dataset_path='', pose_dataset_path='',
                       train_epoch=1, train_sequence=[], train_batch=1,
                       valid_sequence=[],
                       plot_batch=False, plot_epoch=True):

        self.use_cuda = use_cuda

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path

        self.train_epoch = train_epoch
        self.train_sequence = train_sequence
        self.train_batch = train_batch
        
        self.valid_epoch = train_epoch
        self.valid_sequence = valid_sequence
        self.valid_batch = train_batch

        self.plot_batch = plot_batch
        self.plot_epoch = plot_epoch

        if use_cuda == True:        
            # Load main processing unit for neural network
            self.PROCESSOR = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.deepvo_model = DeepVONet()
        self.deepvo_model.to(self.PROCESSOR)


        if str(self.PROCESSOR) == 'cuda:0':
            self.deepvo_model.use_cuda = True
            self.deepvo_model.reset_hidden_states(size=1, zero=True)

        self.deepvo_model.train()
        self.deepvo_model.training = True

        self.train_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                     pose_dataset_path=self.pose_dataset_path,
                                                                     transform=loader_preprocess_param,
                                                                     sequence=train_sequence),
                                                                     batch_size=self.train_batch, shuffle=True, drop_last=True)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.SGD(self.deepvo_model.parameters(), lr=0.0001)

        summary(self.deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(self.PROCESSOR)))

        # Prepare batch error graph
        if self.plot_batch == True:
            
            self.train_plot_color = plt.cm.get_cmap('rainbow', len(train_sequence))
            self.train_plot_x = 0

            ### Plotting graph setup with broken y-axis ######################################
            fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 8))
            self.ax1.set_ylim(2, 30)
            self.ax2.set_ylim(0, 1.5)

            self.ax1.spines['bottom'].set_visible(False)
            self.ax2.spines['top'].set_visible(False)

            self.ax1.xaxis.tick_top()
            self.ax1.tick_params(labeltop=False)
            self.ax2.xaxis.tick_bottom()

            d = .015    # how big to make the diagonal lines in axes coordinates
                        # arguments to pass to plot, just so we don't keep repeating them
            kwargs = dict(transform=self.ax1.transAxes, color='k', clip_on=False)
            self.ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
            self.ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

            kwargs.update(transform=self.ax2.transAxes)  # switch to the bottom axes
            self.ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
            self.ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
            ################################################################################

    def train(self):

        training_loss = []

        for epoch in range(self.train_epoch):

            print('[EPOCH] : {}'.format(epoch))

            loss_sum = 0.0

            for batch_idx, (prev_current_img, prev_current_odom) in enumerate(self.train_loader):

                if self.use_cuda == True:
                    prev_current_img = Variable(prev_current_img.to(self.PROCESSOR))
                    prev_current_odom = Variable(prev_current_odom.to(self.PROCESSOR))

                    estimated_odom = Variable(torch.zeros(prev_current_odom.shape))

                if self.train_loader.dataset.sequence_change == True:

                    # Sequence has changed LSTM reset
                    print('[Sequence Change] LSTM Reset')

                    self.deepvo_model.reset_hidden_states(size=1, zero=True)

                self.optimizer.zero_grad()
                
                estimated_odom = self.deepvo_model(prev_current_img)
                self.deepvo_model.reset_hidden_states(size=1, zero=False)

                loss = self.criterion(estimated_odom, prev_current_odom.float())

                loss.backward()
                self.optimizer.step()

                print('[EPOCH {}] Batch : {} / Loss : {}'.format(epoch, batch_idx, loss))
                    
                # Plotting batch error graph
                if self.plot_batch == True:
                    self.ax1.plot(self.train_plot_x, loss.item(), c=self.train_plot_color(self.train_loader.dataset.sequence_idx), marker='o')
                    self.ax2.plot(self.train_plot_x, loss.item(), c=self.train_plot_color(self.train_loader.dataset.sequence_idx), marker='o')

                    self.ax1.set_title('DeepVO Training with KITTI [MSE Loss at each batch]\nTraining Sequence ' + str(train_sequence))
                    self.ax2.set_xlabel('Training Length')
                    self.ax2.set_ylabel('MSELoss')
                    
                    self.train_plot_x += 1

                loss_sum += loss.item()

            training_loss.append(loss_sum / len(self.train_loader))

            # Save batch error graph
            if self.plot_batch == True:
                plt.savefig('./Training Results ' + str(datetime.datetime.now()) + '.png')

            print('[Epoch {} Complete] Loader Reset'.format(epoch))
            self.train_loader.dataset.reset_loader()

            print('[Epoch {} Complete] LSTM Reset'.format(epoch))
            self.deepvo_model.reset_hidden_states(size=1, zero=True)

        # Plotting average loss on each epoch
        if self.plot_epoch == True:
            plt.clf()
            plt.figure(figsize=(20, 8))
            plt.plot(range(self.train_epoch), training_loss, 'bo-')
            plt.title('DeepVO Training with KITTI [Average MSE Loss]\nTraining Sequence ' + str(self.train_sequence))
            plt.xlabel('Training Length')
            plt.ylabel('MSELoss')
            plt.savefig('./Training Results ' + str(datetime.datetime.now()) + '.png')

        torch.save(self.deepvo_model, './DeepVO_' + str(datetime.datetime.now()) + '.pth')

        return self.deepvo_model, training_loss

