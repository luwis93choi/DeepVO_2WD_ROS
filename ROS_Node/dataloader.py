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

    def __init__(self, transform=None):

        self.len = 1    # The size of dataset in use

        self.transform = transform      # Image transformation conditions (Resolution Change)

        self.prev_image = None
        self.current_image = None

    def load_image_data(self, prev_image, current_image):

        self.prev_image = prev_image
        self.current_image = current_image

    def __getitem__(self, index):

        print('get item')

        prev_current_stacked_img = np.asarray(np.concatenate([self.prev_image, self.current_image], axis=0))

        return prev_current_stacked_img

    def __len__(self):

        return self.len
