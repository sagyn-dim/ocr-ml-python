import numpy as np
import torch
from data_load import get_train_data,get_val_data, CharRecog
from train import train_model, CnnModel, ResNet9, LogRegModel
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import albumentations as A

def mainf():
	#Set number of training and validation samples per character, total character number 35
	spc_train = 4000 
	spc_val = 1000

	#Pre load the data from the directory
	train_data = get_train_data(spc_train)
	val_data = get_val_data(spc_val)

	#Add data transform 
	train_transform = A.Compose(
	    [
	        A.Rotate(limit = 30, p=0.3),
	        A.RandomCrop(height=26, width=26, p = 0.3),
	        A.PadIfNeeded(min_height = 32, min_width = 32)
	    ]
	)

	val_transform = A.Compose(
	    [
	        A.CenterCrop(height=26, width=26, p = 0.1),
	        A.PadIfNeeded(min_height = 32, min_width = 32)
	    ]
	)

	#Create train and val datasets using custom Dataset class
	train_ds = CharRecog(data_preload = train_data, transform = None)
	val_ds = CharRecog(data_preload = val_data, transform = None)

	#Create dataloaders
	batch_size = 128

	train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
	val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)

	#Tranfering to GPU if available
	def get_default_device():
	    """Pick GPU if available, else CPU"""
	    if torch.cuda.is_available():
	        return torch.device('cuda')
	    else:
	        return torch.device('cpu')
	    
	def to_device(data, device):
	    """Move tensor(s) to chosen device"""
	    if isinstance(data, (list,tuple)):
	        return [to_device(x, device) for x in data]
	    return data.to(device, non_blocking=True)

	class DeviceDataLoader():

	    """Wrap a dataloader to move data to a device"""
	    def __init__(self, dl, device):
	        self.dl = dl
	        self.device = device
	        self.dataset = len(dl.dataset)
	        
	    def __iter__(self):
	        """Yield a batch of data after moving it to device"""
	        for b in self.dl: 
	            yield to_device(b, self.device)

	    def __len__(self):
	        """Number of batches"""
	        return len(self.dl)


	#See if GPU is available
	device = get_default_device()

	#Wrap the dataloaders to be used in GPU
	train_dl = DeviceDataLoader(train_dl, device)
	val_dl = DeviceDataLoader(val_dl, device)

	#Store in dict
	dataloaders_dict = {'train': train_dl, 'val': val_dl}

	#Define training parameters
	num_epochs = 40
	optimizer = torch.optim.Adam
	lr = 0.0001


	for m in ['resnet9', 'cnn', 'logreg']: # 'resnet9':

	    val_acc_history = []
	    train_acc_history = []
	            
	    if m == 'resnet9':
	        #Instance of the model
	        model = to_device(ResNet9(in_channels = 1, num_classes = 35), device)

	    elif m == 'cnn':
	        model = to_device(CnnModel(), device)

	    elif m == 'logreg':
	    	model = to_device(LogRegModel(), device)
	                
	    best_model, val_acc_history, train_acc_history = train_model(model, lr, dataloaders_dict, num_epochs)

	    acc_backup = [val_acc_history, train_acc_history].copy()

	    with open(m + str(lr) + '.pickle', 'wb') as handle:
	        pickle.dump(best_model, handle)
	        handle.close()

	            # for d in range(len(val_acc_history)):
	            #     val_acc_history[d] = val_acc_history[d].to(torch.device('cpu'))
	            #     train_acc_history[d] = train_acc_history[d].to(torch.device('cpu'))

	    with open(m + str(lr) +  '.pickle', 'wb') as handle:
	        pickle.dump([val_acc_history, train_acc_history], handle)
	        handle.close()

if __name__ == '__main__':
	mainf()