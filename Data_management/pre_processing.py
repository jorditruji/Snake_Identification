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
for _i, file in enumerate(imgs):

	img, type_ = read_jpg_train(file)
	width, height = img.size
	print(img.size)
	widths.append(width)
	heights.append(height)
	img = np.array(img, dtype = float)/255.0
	width_ ,height, n_channels = img.shape
	flatten_img = img.reshape(n_channels,width_*height)

	try:
		means.append(np.mean(flatten_img,axis=1))
		stds.append(np.std(flatten_img,axis=1))
	except Exception as e:
		print(e)


	print("Means",means)
	print("width: {}, height: {}".format(width, height))
	if _i == 1:
		break


means = np.array(means)
stds = np.array(stds)

global_mean = np.mean(means,axis = 0)
global_std = np.mean(stds,axis = 0)

print("Means: {}".format(global_mean))
print("STDs: {}".format(global_std))

print("Saving")
print(heights, widths)
np.save('heights.npy', heights)
np.save('widths.npy', widths)