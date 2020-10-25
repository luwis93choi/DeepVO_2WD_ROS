import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

class DeepVONet(nn.Module):

    # DeepVO NN Initialization
    # Overriding base class of neural network (nn.Module)
    def __init__(self):
        super(DeepVONet, self).__init__()

        # CNN Layer 1
        self.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.relu1 = nn.ReLU(inplace=True)  # inplace = True : ReLU will modify the input directly without allocating any additional memory for ouput
                                            #                  This option is used to save memory usage, but it has to be used with care
                                            #                  https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
                                            #                  https://github.com/pytorch/vision/issues/807

        # CNN Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu2 = nn.ReLU(inplace=True)

        # CNN Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu3 = nn.ReLU(inplace=True)

        # CNN Layer 3_1
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)

        # CNN Layer 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)

        # CNN Layer 4_1
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)

        # CNN Layer 5
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU(inplace=True)

        # CNN Layer 5_1
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)

        # CNN Layer 6
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        # LSTM 1
        self.lstm1 = nn.LSTMCell(20*6*1024, 1000)

        # LSTM 2
        self.lstm2 = nn.LSTMCell(1000, 1000)

        # Linear Regression between RNN output features (1x1000) and Pose vector between t-1 and t (1x6) (dX dY dZ dRoll dPitch dYaw)
        self.fc = nn.Linear(in_features=1000, out_features=6)

        # Initialize hidden states of RNN
        self.reset_hidden_states()

    def reset_hidden_states(self, size=1, zero=True):

        if zero == True:
            
            # Hidden State 1 for LSTM 1 (1x1000)
            self.hx1 = Variable(torch.zeros(size, 1000)) 
            # Cell State for LSTM 1 (1x1000)
            self.cx1 = Variable(torch.zeros(size, 1000))

            # Hidden State 2 for LSTM 2 (1x1000)
            self.hx2 = Variable(torch.zeros(size, 1000))
            # Cell State for LSTM 2 (1x1000)
            self.cx2 = Variable(torch.zeros(size, 1000))

        else:

            # Hidden State 1 for LSTM 1 (1x1000)
            self.hx1 = Variable(self.hx1) 
            # Cell State for LSTM 1 (1x1000)
            self.cx1 = Variable(self.cx1)

            # Hidden State 2 for LSTM 2 (1x1000)
            self.hx2 = Variable(self.hx2)
            # Cell State for LSTM 2 (1x1000)
            self.cx2 = Variable(self.cx2)

        # If CUDA is available, prepare and copy hidden states and cell states in CUDA memory
        if next(self.parameters()).is_cuda == True:

            # Hidden State 1 for LSTM 1 in CUDA memory (1x1000)
            self.hx1 = self.hx1.cuda()
            # Cell State 1 for LSTM 1 in CUDA memory (1x1000)
            self.cx1 = self.cx1.cuda()

            # Hidden State 2 for LSTM 2 in CUDA memory (1x1000)
            self.hx2 = self.hx2.cuda()
            # Cell State 2 for LSTM 2 in CUDA memory (1x1000)
            self.cx2 = self.cx2.cuda()

    # Foward pass of DeepVO NN
    def forward(self, x):

        # Forward pass through CNN Layer 1
        x = self.conv1(x)
        x = self.relu1(x)

        # Forward pass through CNN Layer 2
        x = self.conv2(x)
        x = self.relu2(x)

        # Forward pass through CNN Layer 3
        x = self.conv3(x)
        x = self.relu3(x)

        # Forward pass through CNN Layer 3_1
        x = self.conv3_1(x)
        x = self.relu3_1(x)

        # Forward pass through CNN Layer 4
        x = self.conv4(x)
        x = self.relu4(x)

        # Forward pass through CNN Layer 4_1
        x = self.conv4_1(x)
        x = self.relu4_1(x)

        # Forward pass through CNN Layer 5
        x = self.conv5(x)
        x = self.relu5(x)

        # Foward pass through CNN Layer 6
        x = self.conv6(x)

        # Reshpae/Flatten the output of CNN in order to use it as the input of RNN
        x = x.view(x.size(0), 20 * 6 * 1024)

        # Forward pass into LSTM 1
        # hx1, cx1 are the values from previous sequence of CNN prediction and LSTM 1
        # hx1, cx1 are fed into current sequence of RNN in order to compute the geometric changes of CNN features
        self.hx1, self.cx1 = self.lstm1(x, (self.hx1, self.cx1))

        x = self.hx1    # Forwarded values are now updated with Hidden State 1

        # Forward pass into LSTM 2
        # hx2, cx2 are the values from previous sequence of CNN prediction and LSTM 2
        # hx2, cx2 are fed into current sequence of RNN in order to compute the geometric changes of CNN features
        self.hx2, self.cx2 = self.lstm2(x, (self.hx2, self.cx2))

        x = self.hx2    # Forwarded values are now updated with Hidden State 2

        # Forward pass into Linear Regression in order to change output vectors into Pose vector
        x = self.fc(x)

        return x