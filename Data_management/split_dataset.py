import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob


def read_from_folder(path):
	return glob.glob(path+"*/*")


def read_img_from_folder(path):
	return glob.glob(path+"*.jpg")


def split_dataset(data,labels, train_per=0.7, val_per=0.3):
	#Splits dataset in 2 or 3 partitions (if train%+val%<1 the rest goes to test)
	if train_per+val_per<1:
		x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1-train_per)
		x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=(val_per/(1-train_per)))
		return x_train, y_train, x_val, y_val, x_test, y_test
	else:
		x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=1-train_per)
		return x_train, y_train,x_test, y_test

def save_partition(data,labels,name):
	'''Saves matrixes intro numpy file'''
	data=[[x_sample, x_label] for x_sample, x_label in zip(data, name)]
	np.save(name, data, allow_pickle=True, fix_imports=True)



path = '/home/jordimoreratrujillo/train/'
files = read_from_folder(path)

labels = []
samples = []
for img_path in files:
	samples.append(img_path)
	labels.append(img_path.split('/')[-2])

label_encoder = preprocessing.LabelEncoder()
labels=label_encoder.fit_transform(labels)

dict_labels={}
for label,path in zip(labels,samples):
	dict_labels[path]=label

print("total_images: {}".format(len(samples)))

# Split into train/val/test
x_train, y_train,x_test, y_test = split_dataset(samples,labels, train_per=0.7, val_per=0.3)

print("train_images: {}".format(len(x_train)))
print("test_images: {}".format(len(x_test)))



#Save into dict:
partition={}
partition['train']=x_train
partition['validation']=x_test


np.save('partition_img',partition,allow_pickle=True, fix_imports=True)
np.save('labels_img',dict_labels,allow_pickle=True, fix_imports=True)

