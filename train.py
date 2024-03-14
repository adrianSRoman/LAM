import torch
import torchvision
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from torchsummary import summary

from resnet import *
from dataloder import *

import h5py
matplotlib.style.use('ggplot')

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

device = get_device()

model = ResNet18Latent(num_classes=370).to(device).to(torch.float64)
learning_rate = 0.001
###### loss function #####
criterion = nn.MSELoss()
######## optimizer #######
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

########## Network Training Hyper-parameters ############

epochs = 100 
batch_size = 8

################## Load Datasets ########################

trainset = DatasetLoader("/home/asroman/repos/DBPN-Pytorch/data/metu_train9ch.hdf")
testset = DatasetLoader("/home/asroman/repos/DBPN-Pytorch/data/metu_test9ch.hdf")

# trainloader
trainloader = DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True
)
# testloader
testloader = DataLoader(
    testset, 
    batch_size=batch_size, 
    shuffle=False
)

################# Sparse Loss Applied to Latent ################

def sparse_loss(autoencoder, images):
    loss = 0
    values = images
    upsamp_cnt = 0
    for i in range(len(model_children)):
#         print("layers children", model_children[i])
#         print("shape of values", values.shape)
        values_re = F.relu((model_children[i](values.real)))
        values_im = F.relu((model_children[i+len(model_children)//2](values.imag)))
        values = torch.complex(values_re, values_im)
#         print("outputs shape", values.shape)
        loss += torch.mean(torch.abs(values))
    return loss

################# Training & Evaluation Loop ###################

def fit(model, dataloader, epoch):
    print('Training')
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(trainset)/dataloader.batch_size)):
        counter += 1
        _, img = data
        img = img#[:, 4, :, :]
        img = img.to(device)
        #img = img.unsqueeze(1)
        #img = torch.cat((torch.real(img), torch.imag(img)), dim=1)
        imag_label = img.squeeze((0)).detach().cpu().numpy()
        optimizer.zero_grad()
        outputs = model(img)
        outputs = outputs.unsqueeze(1)
        #print("output shape", outputs.shape)
        img = img[:, 4, :, :]
        img = img.unsqueeze(1)
        #print(img.shape)
        img_re_im = torch.cat((torch.real(img), torch.imag(img)), dim=1)
        outputs_re_im = torch.cat((torch.real(outputs), torch.imag(outputs)), dim=1)
        mse_loss = criterion(outputs_re_im, img_re_im)
        if counter % 10 == 0:
            plt.figure(figsize=(12, 6))
            #print(img)
            imag_label = img.squeeze((0)).detach().cpu().numpy()
            output_pred = outputs.squeeze((0)).detach().cpu().numpy()
            #print("output pred", output_pred.shape)
            #print("imag label", imag_label.shape)
            plt.subplot(1, 2, 1)
            plt.imshow(np.real(imag_label[0, 0]), cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Intensity')
            plt.title('Label Visibility matrix - real')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')

            # Plotting the second matrix
            plt.subplot(1, 2, 2)
            plt.imshow(np.real(output_pred[0, 0]), cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Intensity')
            plt.title('Predicted Visibility matrix - real')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            plt.savefig('overfit.png')
        
        loss = mse_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / counter
    print(f"Train Loss: {loss:.12f}")
    return epoch_loss

def validate(model, dataloader, epoch):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(testset)/dataloader.batch_size)):
            counter += 1
            _, img = data
            img = img#[:, 4, :, :]
            img = img.to(device)
            #img = img.unsqueeze(1)
            # img = torch.cat((torch.real(img), torch.imag(img)), dim=1)
            outputs = model(img)
            outputs = outputs.unsqueeze(1)
            img_re_im = torch.cat((torch.real(img), torch.imag(img)), dim=1)           
            outputs_re_im = torch.cat((torch.real(outputs), torch.imag(outputs)), dim=1)
            loss = criterion(outputs_re_im, img_re_im)
            running_loss += loss.item()
    epoch_loss = running_loss / counter
    print(f"Val Loss: {loss:.8f}")  
    # save the reconstructed images every 5 epochs
#     if epoch % 5 == 0:
#         outputs = outputs.view(outputs.size(0), 1, 32, 32).cpu().data
#         save_image(outputs, f"./outputs/images/reconstruction{epoch}.png")
    return epoch_loss


best_val_loss = float('inf')
best_epoch = 0
train_loss = []
val_loss = []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, trainloader, epoch)
    #val_epoch_loss = validate(model, testloader, epoch)
    train_loss.append(train_epoch_loss)
    #val_loss.append(val_epoch_loss)
    #if val_epoch_loss < best_val_loss:
    #    best_val_loss = val_epoch_loss
    #    best_epoch = epoch
    #    print(f"Best eval loss seen at epoch {epoch}", best_val_loss)
    #    #torch.save(model.state_dict(), 'best_model_weights.pth')
end = time.time()
 
print(f"{(end-start)/60:.3} minutes")
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./loss.png')
