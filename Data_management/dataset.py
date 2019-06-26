import torch
from torch.utils import data
import numpy as np
import time
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def pure_pil_alpha_to_color_v1(image, color=(255, 255, 255)):
    im = image.copy()
    #im = im.convert('RGBA').split()[-1]
    background = Image.new('RGBA', image.size, (255,255,255))
    alpha_composite = Image.alpha_composite(background, im)
    return alpha_composite.convert('RGB')

    """Alpha composite an RGBA Image with a specified color.

    Source: https://stackoverflow.com/a/9168169/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    
    def blend_value(back, front, a):
        return (front * a + back * (255 - a)) / 255

    def blend_rgba(back, front):
        result = [blend_value(back[i], front[i], front[3]) for i in (0, 1, 2)]
        return tuple(result + [255])

    im = image.copy()  # don't edit the reference directly
    print(im.size)
    p = im  # load pixel array
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            im[x, y] = blend_rgba(color + (255,), im[x, y])
    print(im.mode)

    return p
    """



class Dataset(data.Dataset):
    """
    Class Dataset:
    - Parameters:
        list_IDs: Vector containing the image paths
        labels: Dictionary containing as keys the image paths and the application as value
        is_train: If true perform data augmentation, else nothing


    Dataset statistical parameters:
    Means: [0.7729613  0.81375206 0.8217029 ]
    STDs: [0.2150494  0.17785758 0.16651867]

    """
    def __init__(self, list_IDs, labels, is_train = True):
        self.labels = labels
        self.list_IDs = list_IDs
        self.is_train = is_train
        self.RGB_transforms_val = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)


    def __getitem__(self, index):
        '''Generates one sample of data'''
        start_time = time.time()
        # Select sample

        img_name = self.list_IDs[index]
        rgb = self.read_jpg_train(img_name[3:])
        label = self.labels[img_name]
        return rgb, label


    def read_jpg_train(self,file):
        '''Read and preprocess rgb frame'''
        
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(file, 'rb') as f:
            img = Image.open(f).convert('RGB')
            # Take care transparencies
            type_ = img.mode
            img = self.RGB_transforms_val(img)
        return img


if __name__ == '__main__':
    # Testing:
    # Sample data
    labels = np.load('labels_img.npy').item()
    img_part = np.load('partition_img.npy').item()
    dataset = Dataset(img_part['train'],labels)
    rgb, label = dataset.__getitem__(1)
    print(rgb.size(), label)



