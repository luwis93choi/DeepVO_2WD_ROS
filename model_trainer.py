from deepvoNet import DeepVONet
from dataloader import voDataLoader

from notifier import notifier_Outlook

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

import datetime
import time
import numpy as np
from matplotlib import pyplot as plt

class trainer():

    def __init__(self, NN_model=None,
                       use_cuda=True, cuda_num='',
                       loader_preprocess_param=transforms.Compose([]), 
                       model_path='./',
                       img_dataset_path='', pose_dataset_path='',
                       learning_rate=0.001,
                       train_epoch=1, train_sequence=[], train_batch=1,
                       valid_sequence=[],
                       plot_batch=False, plot_epoch=True,
                       sender_email='', sender_email_pw='', receiver_email=''):

        self.use_cuda = use_cuda
        self.cuda_num = cuda_num

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path
        self.model_path = model_path

        self.learning_rate = learning_rate

        self.train_epoch = train_epoch
        self.train_sequence = train_sequence
        self.train_batch = train_batch
        
        self.valid_epoch = train_epoch
        self.valid_sequence = valid_sequence
        self.valid_batch = train_batch

        self.plot_batch = plot_batch
        self.plot_epoch = plot_epoch

        self.sender_email = sender_email
        self.sender_pw = sender_email_pw
        self.receiver_email = receiver_email

        if (use_cuda == True) and (cuda_num != ''):        
            # Load main processing unit for neural network
            self.PROCESSOR = torch.device('cuda:'+self.cuda_num if torch.cuda.is_available() else 'cpu')

        else:
            self.PROCESSOR = torch.device('cpu')

        print(str(self.PROCESSOR))

        if NN_model == None:

            self.deepvo_model = DeepVONet()
            self.deepvo_model.to(self.PROCESSOR)

        else:

            self.deepvo_model = NN_model
            self.deepvo_model.to(self.PROCESSOR)
            self.model_path = './'

        if 'cuda' in str(self.PROCESSOR):
            self.deepvo_model.use_cuda = True
            self.deepvo_model.reset_hidden_states(size=1, zero=True, cuda_num=self.cuda_num)

        self.deepvo_model.train()
        self.deepvo_model.training = True

        self.train_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                     pose_dataset_path=self.pose_dataset_path,
                                                                     transform=loader_preprocess_param,
                                                                     sequence=train_sequence),
                                                                     batch_size=self.train_batch, shuffle=True, drop_last=True)

        self.criterion = torch.nn.MSELoss()
        #self.optimizer = optim.SGD(self.deepvo_model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adagrad(self.deepvo_model.parameters(), lr=self.learning_rate)

        summary(self.deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(self.PROCESSOR)))

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

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

        start_time = str(datetime.datetime.now())

        training_loss = []

        for epoch in range(self.train_epoch):

            print('[EPOCH] : {}'.format(epoch))

            loss_sum = 0.0

            before_epoch = time.time()

            for batch_idx, (prev_current_img, prev_current_odom) in enumerate(self.train_loader):

                if self.use_cuda == True:
                    prev_current_img = Variable(prev_current_img.to(self.PROCESSOR))
                    prev_current_odom = Variable(prev_current_odom.to(self.PROCESSOR))

                    estimated_odom = Variable(torch.zeros(prev_current_odom.shape).to(self.PROCESSOR))

                if self.train_loader.dataset.sequence_change == True:

                    # Sequence has changed LSTM reset
                    print('[Sequence Change] LSTM Reset')

                    if 'cuda' in str(self.PROCESSOR):
                        self.deepvo_model.reset_hidden_states(size=1, zero=True, cuda_num=self.cuda_num)
                    else:
                        self.deepvo_model.reset_hidden_states(size=1, zero=True)

                self.optimizer.zero_grad()
                
                estimated_odom = self.deepvo_model(prev_current_img)

                if 'cuda' in str(self.PROCESSOR):
                    self.deepvo_model.reset_hidden_states(size=1, zero=False)
                else:
                    self.deepvo_model.reset_hidden_states(size=1, zero=True)

                #loss = self.criterion(estimated_odom, prev_current_odom.float())
                loss = self.deepvo_model.get_pose_loss(estimated_odom, prev_current_odom)

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

            after_epoch = time.time()

            training_loss.append(loss_sum / len(self.train_loader))

            # Send the result of each epoch
            self.notifier.send(receiver_email=self.receiver_email, 
                               title='[Epoch {} / {} Complete]'.format(epoch+1, self.train_epoch),
                               contents=str(loss_sum / len(self.train_loader)) + '\n' + 'Time taken : {} sec'.format(after_epoch-before_epoch))

            # Save batch error graph
            if self.plot_batch == True:
                plt.savefig('./Training Results ' + str(datetime.datetime.now()) + '.png')

            print('[Epoch {} Complete] Loader Reset'.format(epoch))
            self.train_loader.dataset.reset_loader()

            print('[Epoch {} Complete] LSTM Reset'.format(epoch))
            if 'cuda' in str(self.PROCESSOR):
                self.deepvo_model.reset_hidden_states(size=1, zero=True, cuda_num=self.cuda_num)
            else:
                self.deepvo_model.reset_hidden_states(size=1, zero=True)

            torch.save(self.deepvo_model, './DeepVO_' + start_time + '.pth')

        # Plotting average loss on each epoch
        if self.plot_epoch == True:
            plt.clf()
            plt.figure(figsize=(20, 8))
            plt.plot(range(self.train_epoch), training_loss, 'bo-')
            plt.title('DeepVO Training with KITTI [Average MSE Loss]\nTraining Sequence ' + str(self.train_sequence))
            plt.xlabel('Training Length')
            plt.ylabel('MSELoss')
            plt.savefig('./Training Results ' + str(datetime.datetime.now()) + '.png')


        return self.deepvo_model, training_loss

