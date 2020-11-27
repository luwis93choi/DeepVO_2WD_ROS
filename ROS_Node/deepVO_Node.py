import Pyro4
import base64
import cv2 as cv
import threading
import time

from deepvoNet import DeepVONet
from dataloader import voDataLoader

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary
from PIL import Image

import numpy as np

from matplotlib import pyplot as plt

import argparse
import os

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

disp_preprocess = transforms.Compose([
    transforms.Resize((384, 1280)),
    transforms.CenterCrop((384, 1280))
])

init_required = True        # Initialization flag for camera buffer
img_ready = False           # Check whether realsense node is sending image to DeepVO node

decoded_img = None
pyro4_status = 'OFF'

@Pyro4.expose
class Python3_Server(object):
    def response(self, data):

        global decoded_img

        global init_required
        global img_ready

        decoded_string = np.fromstring(base64.b64decode(data), np.uint8)
        decoded_img = cv.imdecode(decoded_string, cv.IMREAD_COLOR)

        img_ready = True

        return '[Server] RX Time : {}'.format(time.time())

def create_pyro4Server():

    global pyro4_status

    pyro4_status = 'ON'
    pyroDaemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    uri = pyroDaemon.register(Python3_Server)
    ns.register('deepVO_Node', uri)

    pyroDaemon.requestLoop()

def deepVO_Node():

    global decoded_img
    global pyro4_status
    global init_required

    # Input data into DeepVO
    input_img_array = [None, None]      # [prev_img, current_img]
    input_stacked_img = None

    # Output odometry results from DeepVO
    estimated_x = 0.0
    estimated_y = 0.0
    estimated_z = 0.0

    # Prepare CUDA if available
    PROCESSOR = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare dataloader
    test_loader = torch.utils.data.DataLoader(voDataLoader(transform=preprocess), batch_size=1)

    # Load pre-trained DeepVO Model on CUDA if available
    if torch.cuda.is_available():
        deepvo_model = torch.load('/home/luwis/ICSL_Project/DeepVO_TrainedModel/DeepVO_2020-11-18 03_54_13.559726.pth', map_location='cuda:0')
        deepvo_model.to(PROCESSOR)
        deepvo_model.use_cuda = True
        deepvo_model.reset_hidden_states(size=1, zero=True, cuda_num='0')
        deepvo_model.eval()
        deepvo_model.evaluation = True

        summary(deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(PROCESSOR)))

    else:
        deepvo_model = torch.load('/home/luwis/ICSL_Project/DeepVO_TrainedModel/DeepVO_2020-11-18 03_54_13.559726.pth', map_location='cpu')
        deepvo_model.to(PROCESSOR)
        deepvo_model.use_cuda = False
        deepvo_model.reset_hidden_states(size=1, zero=True)
        deepvo_model.eval()
        deepvo_model.evaluation = True

        summary(deepvo_model, Variable(torch.zeros((1, 6, 384, 1280)).to(PROCESSOR)))

    # Prepare Pyro4 server daemon sub thread
    pyro4ServerThread = threading.Thread(target=create_pyro4Server, args=())
    pyro4ServerThread.daemon = True
    pyro4ServerThread.start()

    print('pyro4Server status : {}'.format(pyro4_status))

    while True:

        if decoded_img is not None:

            # OpenCV image to PIL Image conversion for Neural Network
            # (Reference : https://stackoverflow.com/questions/43232813/convert-opencv-image-format-to-pil-image-format)
            PIL_img = Image.fromarray(decoded_img)

            transformed_img = preprocess(PIL_img)   # Normalize input image
            
            # At the intial boot, load current image into input image array
            if (init_required is True) and (img_ready is True):

                input_img_array[1] = transformed_img
                
                init_required = False

            # After initialization, push previous image to lower index and load current image on higher index
            # Stack previous imagen and current image in order to feed DeepVO network
            elif init_required is False:

                input_img_array[0] = input_img_array[1]
                input_img_array[1] = transformed_img

                # Pass stacked image to dataloader by accessing dataset attribute
                test_loader.dataset.load_image_data(input_img_array[0], input_img_array[1])

                for batch_idx, prev_current_img in enumerate(test_loader):

                    input_stacked_img = Variable(prev_current_img.to(PROCESSOR))
                    estimated_odom = (torch.from_numpy(np.asarray([0, 0, 0, 0, 0, 0])).unsqueeze(0)).to(PROCESSOR)  # dx dy dz droll dpitch dyaw
                    
                    estimated_odom = deepvo_model(input_stacked_img)    # Visual Odometry Prediction
                
                    # Prepare prediction results
                    predicted_odom = estimated_odom.data.cpu().numpy()

                    predicted_dx = predicted_odom[0][0]
                    predicted_dy = predicted_odom[0][1]
                    predicted_dz = predicted_odom[0][2]

                    predicted_roll = predicted_odom[0][3]
                    predicted_pitch = predicted_odom[0][4]
                    predicted_yaw = predicted_odom[0][5]

                    # Accumulate odometry prediction results
                    estimated_x = estimated_x + predicted_dx
                    estimated_z = estimated_z + predicted_dz

                    deepvo_model.reset_hidden_states(size=1, zero=False)

                    print('[Prediction] X : {} | Z : {}'.format(estimated_x, estimated_z))

                # Plot odometry trajectory
                plt.plot(estimated_x, estimated_z, 'bo')
                plt.pause(0.001)
                plt.show(block=False)

            # Display the preprocessed input image before it is converted to tensor and normalized
            reverse_img = np.array(disp_preprocess(PIL_img))                     # Coversion back to numpy array from PIL Image
            cv_reversed_img=cv.cvtColor(reverse_img, cv.COLOR_RGB2BGR)      # Conversion back to OpenCV Image matrix

            cv.namedWindow('Python3 Server Realsene Img', cv.WINDOW_AUTOSIZE)
            cv.imshow('Python3 Server Realsene Img', cv_reversed_img)
            cv.waitKey(1)

if __name__ == '__main__':

    deepVO_Node()