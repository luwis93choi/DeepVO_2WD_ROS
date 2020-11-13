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
import math
from matplotlib import pyplot as plt

class tester():

    def __init__(self, NN_model=None,
                       model_path='./',
                       use_cuda=True, cuda_num='',
                       loader_preprocess_param=transforms.Compose([]), 
                       img_dataset_path='', pose_dataset_path='',
                       test_epoch=1, test_sequence=[], test_batch=1,
                       plot_batch=False, plot_epoch=True,
                       sender_email='', sender_email_pw='', receiver_email=''):

        self.NN_model = NN_model
        self.use_cuda = use_cuda
        self.cuda_num = cuda_num

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path
        self.model_path = model_path

        self.test_epoch = test_epoch
        self.test_sequence = test_sequence
        self.test_batch = test_batch

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
        
        self.NN_model.to(self.PROCESSOR)

        if 'cuda' in str(self.PROCESSOR):
            self.NN_model.use_cuda = True
            self.NN_model.reset_hidden_states(size=1, zero=True, cuda_num=self.cuda_num)

        self.NN_model.eval()
        self.NN_model.evaluation = True

        self.test_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                    pose_dataset_path=self.pose_dataset_path,
                                                                    transform=loader_preprocess_param,
                                                                    sequence=test_sequence),
                                                                    batch_size=self.test_batch, shuffle=True, drop_last=True)

        self.criterion = torch.nn.MSELoss()
        
        summary(self.NN_model, Variable(torch.zeros((1, 6, 384, 1280)).to(self.PROCESSOR)))

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def run_test(self):

        estimated_x = 0.0
        estimated_y = 0.0
        estimated_z = 0.0

        current_pose_T = np.array([[0], 
                                   [0], 
                                   [0]])

        current_pose_R = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        test_loss = []

        fig = plt.figure(figsize=(20, 10))
        plt.grid(True)

        for epoch in range(self.test_epoch):

            print('[EPOCH] : {}'.format(epoch))

            loss_sum = 0.0

            before_epoch = time.time()

            for batch_idx, (prev_current_img, prev_current_odom) in enumerate(self.test_loader):

                if self.use_cuda == True:
                    prev_current_img = Variable(prev_current_img.to(self.PROCESSOR))
                    prev_current_odom = Variable(prev_current_odom.to(self.PROCESSOR))

                    estimated_odom = Variable(torch.zeros(prev_current_odom.shape).to(self.PROCESSOR))

                if self.test_loader.dataset.sequence_change == True:

                    # Sequence has changed LSTM reset
                    print('[Sequence Change] LSTM Reset')

                    if 'cuda' in str(self.PROCESSOR):
                        self.NN_model.reset_hidden_states(size=1, zero=False)
                    else:
                        self.NN_model.reset_hidden_states(size=1, zero=True)

                estimated_odom = self.NN_model(prev_current_img)
                
                if 'cuda' in str(self.PROCESSOR):
                    self.NN_model.reset_hidden_states(size=1, zero=False)
                else:
                    self.NN_model.reset_hidden_states(size=1, zero=True)

                #loss = self.criterion(estimated_odom, prev_current_odom.float())
                loss = self.NN_model.get_pose_loss(estimated_odom, prev_current_odom)

                print('[EPOCH {}] Batch : {} / Loss : {}'.format(epoch, batch_idx, loss))

                predicted_odom = estimated_odom.data.cpu().numpy()

                predicted_dx = predicted_odom[0][0]
                predicted_dy = predicted_odom[0][1]
                predicted_dz = predicted_odom[0][2]

                predicted_roll = predicted_odom[0][3]
                predicted_pitch = predicted_odom[0][4]
                predicted_yaw = predicted_odom[0][5]

                rotation_Mat = np.array([[np.cos(predicted_pitch)*np.cos(predicted_yaw), np.sin(predicted_roll)*np.sin(predicted_pitch)*np.cos(predicted_yaw) - np.cos(predicted_roll)*np.sin(predicted_yaw), np.cos(predicted_roll)*np.sin(predicted_pitch)*np.cos(predicted_yaw) + np.sin(predicted_roll)*np.sin(predicted_pitch)], 
                                         [np.cos(predicted_pitch)*np.sin(predicted_yaw), np.sin(predicted_roll)*np.sin(predicted_pitch)*np.sin(predicted_yaw) + np.cos(predicted_roll)*np.cos(predicted_yaw), np.cos(predicted_roll)*np.sin(predicted_pitch)*np.sin(predicted_yaw) - np.sin(predicted_roll)*np.cos(predicted_pitch)], 
                                         [-np.sin(predicted_pitch),                      np.sin(predicted_roll)*np.cos(predicted_pitch),                                                                      np.cos(predicted_roll)*np.cos(predicted_pitch)]])

                translation_Mat = np.array([[predicted_dx], 
                                            [predicted_dy], 
                                            [predicted_dz]])

                #current_pose_T = current_pose_T + current_pose_R.dot(translation_Mat)
                #current_pose_R = rotation_Mat.dot(current_pose_R)

                current_pose_T = rotation_Mat.dot(current_pose_T) + translation_Mat

                estimated_x = estimated_x + predicted_dx
                estimated_z = estimated_z + predicted_dz

                print(rotation_Mat)
                print(translation_Mat)
                print(current_pose_T)

                #plt.plot(current_pose_T[0][0], current_pose_T[2][0], 'bo')
                plt.plot(estimated_x, estimated_z, 'bo')
                plt.pause(0.001)
                plt.show(block=False)

                loss_sum += loss.item()

            after_epoch = time.time()

            test_loss.append(loss_sum / len(self.test_loader))

            # Send the result of each epoch
            self.notifier.send(receiver_email=self.receiver_email, 
                               title='[Epoch {} / {} Complete]'.format(epoch+1, self.test_epoch),
                               contents=str(loss_sum / len(self.test_loader)) + '\n' + 'Time taken : {} sec'.format(after_epoch-before_epoch))

            print('[Epoch {} Complete] Loader Reset'.format(epoch))
            self.test_loader.dataset.reset_loader()

            print('[Epoch {} Complete] LSTM Reset'.format(epoch))
            if 'cuda' in str(self.PROCESSOR):
                self.NN_model.reset_hidden_states(size=1, zero=True, cuda_num=self.cuda_num)
            else:
                self.NN_model.reset_hidden_states(size=1, zero=True)

        # Plotting average loss on each epoch
        if self.plot_epoch == True:
            plt.clf()
            plt.figure(figsize=(20, 8))
            plt.plot(range(self.test_epoch), test_loss, 'bo-')
            plt.title('DeepVO Test with KITTI [Average MSE Loss]\nTest Sequence ' + str(self.test_sequence))
            plt.xlabel('Test Length')
            plt.ylabel('MSELoss')
            plt.savefig(self.model_path + 'Test Results ' + str(datetime.datetime.now()) + '.png')

        return test_loss