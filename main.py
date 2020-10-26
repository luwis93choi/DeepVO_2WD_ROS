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

# Load main processing unit for neural network
PROCESSOR = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

deepvo_model = DeepVONet()
deepvo_model.to(PROCESSOR)

if str(PROCESSOR) == 'cuda:0':
    deepvo_model.use_cuda = True
    deepvo_model.reset_hidden_states(size=1, zero=True)

deepvo_model.train()
deepvo_model.training = True

train_epoch = 3
train_sequence = ['01']
#train_sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
test_sequence = ['01']

train_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences',
                                                        pose_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses',
                                                        transform=preprocess,
                                                        test=False,
                                                        train_sequence=train_sequence,
                                                        test_sequence=test_sequence),
                                                        batch_size=1, shuffle=True, drop_last=True)

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(deepvo_model.parameters(), lr=0.0001)

summary(deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(PROCESSOR)))

train_plot_color = plt.cm.get_cmap('rainbow', len(train_sequence))
train_plot_x = 0

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

for epoch in range(train_epoch):

    print('[EPOCH] : {}'.format(epoch))

    for batch_idx, (prev_current_img, prev_current_odom) in enumerate(train_loader):

        prev_current_img = Variable(prev_current_img.to(PROCESSOR))
        prev_current_odom = Variable(prev_current_odom.to(PROCESSOR))

        estimated_odom = Variable(torch.zeros(prev_current_odom.shape))

        if train_loader.dataset.sequence_change == True:

            # Sequence has changed LSTM reset
            print('[Sequence Change] LSTM Reset')

            deepvo_model.reset_hidden_states(size=1, zero=True)

        optimizer.zero_grad()
        
        estimated_odom = deepvo_model(prev_current_img)
        deepvo_model.reset_hidden_states(size=1, zero=False)

        loss = criterion(estimated_odom, prev_current_odom.float())

        loss.backward()
        optimizer.step()

        print('[EPOCH {}] Batch : {} / Loss : {}'.format(epoch, batch_idx, loss))
        
        ax1.plot(train_plot_x, loss.item(), c=train_plot_color(train_loader.dataset.sequence_idx), marker='o')
        ax2.plot(train_plot_x, loss.item(), c=train_plot_color(train_loader.dataset.sequence_idx), marker='o')

        ax1.set_title('DeepVO Training with KITTI\nTraining Sequence ' + str(train_sequence))
        ax2.set_xlabel('Training Length')
        ax2.set_ylabel('MSELoss')

        #plt.pause(0.0001)
        #plt.show(block=False)
        train_plot_x += 1

    plt.savefig('./Training Results ' + str(datetime.datetime.now()) + '.png')

    train_loader.dataset.reset_loader()

torch.save(deepvo_model, './DeepVO_' + str(datetime.datetime.now()) + '.pth')