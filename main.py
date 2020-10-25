from deepvoNet import DeepVONet
from dataloader import voDataLoader

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

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

train_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences',
                                                        pose_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses',
                                                        transform=preprocess,
                                                        test=False,
                                                        train_sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                                                        test_sequence=['01']),
                                                        batch_size=1, shuffle=True, drop_last=True)

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(deepvo_model.parameters(), lr=0.0001)

summary(deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(PROCESSOR)))

for batch_idx, (prev_current_img, prev_current_odom) in enumerate(train_loader):

    prev_current_img = Variable(prev_current_img.to(PROCESSOR))
    prev_current_odom = Variable(prev_current_odom.to(PROCESSOR))

    estimated_odom = Variable(torch.zeros(prev_current_odom.shape))

    deepvo_model.reset_hidden_states(size=1, zero=True)

    estimated_odom = deepvo_model(prev_current_img)

    loss = criterion(estimated_odom, prev_current_odom.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Batch : {} / Loss : {}'.format(batch_idx, loss))

