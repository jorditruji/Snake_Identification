import numpy as np
from PIL import Image


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

means = []
stds = []
widths = []
heights = []
for file in imgs:

	img, type_ = read_jpg_train(file)
	width, height = img.size
	img = np.array(img, dtype = float)/255.0

	n_channels, width_ ,height = img.shape
	flatten_img = img.reshape(n_channels,width_*height)
	print(img.shape, flatten_img.shape )

	try:
		means.append(np.mean(flatten_img,axis=1))
		stds.append(np.std(flatten_img,axis=1))
	except Exception as e:
		print(e)
	# Let's get means and stds: (rank [0-1])


means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

print("Means: {}".format(global_mean))
print("STDs: {}".format(global_std))
