import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils import data
from Data_management.dataset import Dataset
from torchvision import datasets, models, transforms
import time
import os
import copy
import datetime
from torch.autograd import Variable
from Models.residual_attention_network import ResidualAttentionModel_92



def train(model_ft, criterion, optimizer_ft, train_generator, val_generator, regularize = False, n_epochs= 20 , lr_scheduler = None ):
	start_time = time.time()
	# Current time 
	data_actual = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	# Interesting metrics to keep
	loss_train = []
	acc_train = []
	loss_val = []
	acc_val = []

	best_val_acc = 0.

	# Main loop
	for epoch in range(n_epochs):
		# Train
		model_ft.train()
		cont = 0
		running_loss = 0.0
		running_corrects = 0

		# Learning rate decay
		if lr_scheduler:
			lr_scheduler.step()

		for rgbs, labels in training_generator:
			cont+=1
			# Get items from generator
			if torch.cuda.is_available():
				inputs = rgbs.cuda()
				labels = labels.cuda()
			
			else:
				inputs = rgbs
			
			# Clean grads
			optimizer_ft.zero_grad()

			#Forward
			outs = model_ft(inputs)
			_, preds = torch.max(outs, 1)
			loss = criterion(outs, labels)



			# Track losses + correct predictions
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
			loss.backward()
			optimizer_ft.step()


			

		# Get avg loss + accuracies in %
		epoch_loss = running_loss / dataset.__len__()
		epoch_acc = running_corrects.double().detach() / dataset.__len__()
		print('{} Loss: {:.4f} Acc: {:.4f}'.format('Train epoch '+str(epoch), epoch_loss, epoch_acc))
		loss_train.append(epoch_loss)#.data.cpu().numpy()[0])
		acc_train.append(epoch_acc.data.cpu().numpy())

		# Val
		model_ft.eval()
		cont = 0
		running_loss = 0.0
		running_corrects = 0
		predicts = []
		val_labels = []

		for rgbs, labels in val_generator:
			cont+=1
			val_labels+=list(labels.numpy())
			# Get items from generator
			if torch.cuda.is_available():
				inputs = rgbs.cuda()
				labels = labels.cuda()
			else:
				inputs = rgbs
			# Clean grads
			optimizer_ft.zero_grad()

			#Forward
			outs = model_ft(inputs)
			_, preds = torch.max(outs, 1)
			predicts+=list(preds.cpu().numpy())
			loss = criterion(outs, labels)
			loss.backward()
			optimizer_ft.step()

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
			


		epoch_loss = running_loss / dataset_val.__len__()
		epoch_acc = running_corrects.double().detach() / dataset_val.__len__()
		epoch_acc = epoch_acc.data.cpu().numpy()
		print('{} Loss: {:.4f} Acc: {:.4f}'.format('Val epoch '+str(epoch), epoch_loss, epoch_acc))
		loss_val.append(epoch_loss)#.data.cpu().numpy())
		acc_val.append(epoch_acc)

		# Save model and early stop?
		if epoch_acc > best_val_acc:
			best_model_wts = copy.deepcopy(model_ft.state_dict())
			best_predicts = predicts
			best_labels = val_labels
			torch.save(best_model_wts, 'resnet_'+data_actual)			

	results = { }
	loss = {}
	acc = {}
	# losses
	loss['train'] = np.array(loss_train)
	loss['val'] = np.array(loss_val)


	# accuracies
	acc['train'] = np.array(acc_train)
	acc['val'] = np.array(acc_val)

	results['losses']=loss
	results['acc'] = acc
	print("--- %s seconds ---" % (time.time() - start_time))
	return results, best_labels, best_predicts, data_actual


# Use gpu if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(1993)
	torch.manual_seed(1993)
else:
	torch.manual_seed(1993)
# Create dataset
labels = np.load('Data_management/labels_img.npy').item()
img_part_train = np.load('Data_management/partition_img.npy').item()['train']
img_part_train_net = [img for img in img_part_train if img.split('.')[-1]  in ('jpg','png')]

dataset = Dataset(img_part_train_net,labels)

img_part_val = np.load('Data_management/partition_img.npy').item()['validation']
img_part_val_net =  [img for img in img_part_val if img.split('.')[-1]  in ('jpg','png')]

dataset_val = Dataset(img_part_val_net,labels, is_train = False)

# Parameters
params = {'batch_size': 64 ,
          'shuffle': True,
          'num_workers': 12,
          'pin_memory': True}


# Create data loaders
training_generator = data.DataLoader(dataset,**params)
val_generator = data.DataLoader(dataset_val,**params)


# Create resnet model
#model_ft, input_size = initialize_model(model_name)
model_ft = resnet18()
print("Amount of parameters:")
print(sum(p.numel() for p in model_ft.parameters()))


#Regularization:
lambda1, lambda2 = 0.5, 0.01

# Decay LR by a factor of 0.1 every  epochs

print(model_ft)
ct = 0

''' FREEZE LAYERS
for child in model_ft.children():
	ct += 1
	if ct < 5:
   		for param in child.parameters():
        		param.requires_grad = False
'''

if torch.cuda.is_available():
	model_ft.cuda()
	criterion = nn.CrossEntropyLoss().cuda()
else:
	criterion = nn.CrossEntropyLoss()

#optimizer_ft = optim.Adam(model_ft.parameters(), lr=2e-4)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.5)

res,val_labels, predicts, data_actual = train(model_ft, criterion, optimizer_ft, 
	training_generator, val_generator, regularize = False, n_epochs= 13, lr_scheduler=None) 

'''exp_lr_scheduler)'''

# Save the results:

np.save('results_'+str(data_actual), res)

# --- 37.59446835517883 seconds --- AWS
# --- 141.05659770965576 seconds ---
