import numpy as np
from PIL import Image
from dataset import Dataset
from torch.utils import data

def read_jpg_train(file):
    '''Read and returns PIL image and type'''
    
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(file, 'rb') as f:
        img = Image.open(f).convert('RGB')

        type_ = img.mode

    return img, type_



# Llegim fitxer partition
labels = np.load('labels_img.npy').item()
imgs = np.load('partition_img.npy').item()['train']
# Create dataset instance
dataset = Dataset(imgs,labels)


# Parameters of the generator
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True}


# Create data loaders
training_generator = data.DataLoader(dataset,**params)



means = []
stds = []

for rgb, label in training_generator:
	label = label.numpy()
	rgb = rgb.numpy()
	b_size, n_channels, width_ ,height = rgb.shape
	flatten_img = rgb.reshape(n_channels,b_size*width_*height)
	print(rgb.shape)
	try:
		means.append(np.mean(flatten_img,axis=1))
		stds.append(np.std(flatten_img,axis=1))
	except Exception as e:
		print(e)





means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

print("Means: {}".format(global_mean))
print("STDs: {}".format(global_std))

