import numpy as np


# Load already done partitions:
labels = np.load('labels_img.npy').item()
img_part_train = np.load('partition_img.npy').item()['train']

classes_dict = {} # Mapeja idx de classe a totes les imatges de train de la classe
class_tracker = {} # Mapeja idx de classe a nom de la classe
for i in range(45):
	classes_dict[str(i)] = [k for k,v in labels.items() if v == i]
	class_tracker[str(i)] = [k for k,v in labels.items() if v == i][0].split('/')[-1]

np.save('class2samples.npy', classes_dict)
np.save('class2folder.npy', class_tracker)



