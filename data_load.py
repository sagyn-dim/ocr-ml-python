import os
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer

#Import training data
def get_train_data(spc_tr, dir = "Train/"):
    print('In get data')
    train_data = []
    img_size = 32
    non_chars = ["#","$","&","@"]
    for i in os.listdir(dir):
        if i in non_chars:
    	    continue
        count = 0
        sub_directory = os.path.join(dir,i)
        for j in os.listdir(sub_directory):
            count+=1
            if count > spc_tr:
                break
            img = cv2.imread(os.path.join(sub_directory,j),0)
            img = cv2.resize(img,(img_size,img_size))
            train_data.append([img,i])
    return train_data

def get_val_data(spc_val, dir = "Validation/"):
    print('In get data')
    val_data = []
    img_size = 32
    non_chars = ["#","$","&","@"]
    for i in os.listdir(dir):
        if i in non_chars:
            continue
        count = 0
        sub_directory = os.path.join(dir,i)
        for j in os.listdir(sub_directory):
            count+=1
            if count > spc_val:
                break
            img = cv2.imread(os.path.join(sub_directory,j),0)
            img = cv2.resize(img,(img_size,img_size))
            val_data.append([img, i])
    return val_data


#Import validation data
def get_test_data(spc_test, dir = "Train/"):
    test_data = []
    img_size = 32
    non_chars = ["#","$","&","@"]
    for i in os.listdir(val_dir):
        if i in non_chars:
            continue
        count = 0
        sub_directory = os.path.join(dir,i)
        for j in os.listdir(sub_directory):
            count+=1
            if count > 9000 + spc_test:
                break
            elif count > 9000:
                img = cv2.imread(os.path.join(sub_directory,j),0)
                img = cv2.resize(img,(img_size,img_size))
                test_data.append([img,i])
    return test_data



#Create custom dataset class
#Create custom dataset class
class CharRecog(Dataset):

    def __init__(self, data_preload, transform=None):
        self.transform = transform
        self.data_X = []
        self.data_Y = []
        for features,label in data_preload:
            self.data_X.append(features)
            self.data_Y.append(label)
        LB = LabelBinarizer()
        self.data_Y = LB.fit_transform(self.data_Y)
        self.data_X = np.array(self.data_X)/255.0
        # self.data_Y = np.array(self.data_Y)
        
    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        #Data is expected to return one sample at a time
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Extract the image
        img = self.data_X[idx]
        
        #Transform
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = np.resize(img, (1,32,32))
        img = torch.from_numpy(img)
        #Extract the label
        label = self.data_Y[idx]
        label = torch.from_numpy(label)
        sample = (img, label)
        
        return sample