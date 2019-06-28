import numpy as np


# Load already done partitions:
labels = np.load('Data_management/labels_img.npy').item()
img_part_train = np.load('Data_management/partition_img.npy').item()['train']

classes_dict = {}
for i in range(45):
	classes_dict[i] = [k for k,v in labels.items() if v == i]



