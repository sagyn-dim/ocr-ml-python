import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Base for the CNN model
class ImageClassificationBase(nn.Module):
    #Find loss and correct preds for each batch
    def training_step(self, batch):

        images, labels = batch
        labels = labels.long()
        images = images.float() 
        out = self(images)
        _, preds = torch.max(out, dim=1)
        _, label_pred = torch.max(labels, dim = 1)
        corrects = torch.tensor(torch.sum(preds == label_pred)).item() #Number of Corrects

        loss = F.cross_entropy(out, label_pred) # Calculate loss
        
        return loss, corrects

#Define the cnn model
class CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1), # output 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 32 x 16 x 16
            # nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding =0), #64 x 14 x 14
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 7 x 7
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), #128 x 5 x 5
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 2 x 2
            # nn.BatchNorm2d(128),

            nn.Dropout2d(0.25),
            nn.Flatten(), # output: 128*2*2 = 512
            
            nn.Linear(512, 128),
            nn.ReLU(),
            
            nn.Dropout(0.25),
            
            nn.Linear(128, 35))
            # nn.Softmax())
            
    def forward(self, xb):
        return self.network(xb)

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

class LogRegModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size = 1024, num_classes = 35)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 1024)
        out = self.linear(xb)
        return out

def train_model(model, lr,  dataloaders, num_epochs, optimizer = torch.optim.SGD):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    optimizer = optimizer(model.parameters(), lr)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    loss, corrects = model.training_step(batch)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch[0].size(0)
                running_corrects += corrects

            epoch_loss = running_loss / dataloaders[phase].dataset
            epoch_acc = float(running_corrects) / dataloaders[phase].dataset

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history