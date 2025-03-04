import matplotlib.pylab as pylab
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
import h5py
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import datetime

import torchvision
from torchvision import models, transforms, datasets

import matplotlib.pylab as pylab

params = {'axes.titlesize':20,
          'xtick.direction': 'in' ,
          'ytick.direction' : 'in',
          'xtick.top' : True,
          'ytick.right' : True,
          'ytick.labelsize':16,
          'xtick.labelsize':16
         }

pylab.rcParams.update(params)

print(f"torch version: {torch.__version__}")

print(f"torchvision version: {torchvision.__version__}")

#import atomai as aoi
#import kornia as K
import cv2
import scipy
import argparse
import skimage
from skimage.util import random_noise
from skimage import feature
import glob
from scipy import ndimage
import scipy as sp
import random

import warnings
warnings.filterwarnings('ignore') 

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

from auto4dstem.nn.Train_Function import TrainClass
from auto4dstem.Viz.util import mask_class
from auto4dstem.Viz.viz import set_format_Auto4D, visualize_simulate_result, visual_performance_plot,normalized_strain_matrices
# #from auto4dstem.Data.DataProcess import *
# from auto4dstem.nn.CC_ST_AE import *
# from auto4dstem.nn.Loss_Function import *
# from auto4dstem.nn.Train_Function import *
# from auto4dstem.Viz.util import *
# from auto4dstem.Viz.viz import *
from m3util.util.IO import make_folder
import warnings
warnings.filterwarnings('ignore') 

class conv_block(nn.Module):
    def __init__(self, t_size, n_step):
        super(conv_block, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_2 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov1d_3 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_3 = nn.LayerNorm(n_step)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        x_input = x
        out = self.cov1d_1(x)
        out = self.relu_1(out)
        out = self.cov1d_2(out)
        out = self.relu_2(out)
        out = self.cov1d_3(out)
        out = self.norm_3(out)
        out = self.relu_3(out)
        out = out.add(x_input)

        return out

class identity_block(nn.Module):
    def __init__(self, t_size, n_step):
        super(identity_block, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_input = x
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out

class Joint(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Joint, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.mask_size = encoder.find_mask()

        self.mask = encoder.rotate_mask()

        self.interpolate = encoder.check_inp()
        self.up_size = encoder.check_upsize()

    #        print(self.mask)

    def rotate_mask(self):

        return self.mask

    def forward(self, x, rotate_value=None):
        if self.interpolate:
            predicted_revise, k_out, scaler_shear, rotation, adj_mask, x_inp = self.encoder(x, rotate_value)
        else:
            predicted_revise, k_out, scaler_shear, rotation, adj_mask = self.encoder(x, rotate_value)

        identity = torch.tensor([0, 0, 1], dtype=torch.float).reshape(1, 1, 3).repeat(x.shape[0], 1, 1).to(self.device)
        

        new_theta_1 = torch.cat((scaler_shear, identity), axis=1).to(self.device)
        new_theta_2 = torch.cat((rotation, identity), axis=1).to(self.device)

        inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
        inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)

        predicted_base = self.decoder(k_out)

        if self.interpolate:

            predicted_base_inp = F.interpolate(
                predicted_base, size=(self.up_size, self.up_size), mode="bicubic"
            )

            grid_1 = F.affine_grid(
                inver_theta_1.to(self.device), predicted_base_inp.size()
            ).to(self.device)
            grid_2 = F.affine_grid(
                inver_theta_2.to(self.device), predicted_base_inp.size()
            ).to(self.device)

            predicted_rotate = F.grid_sample(predicted_base_inp, grid_2, mode="bicubic")
            predicted_input = F.grid_sample(predicted_rotate, grid_1, mode="bicubic")

        else:

            grid_1 = F.affine_grid(inver_theta_1.to(self.device), x.size()).to(
                self.device
            )
            grid_2 = F.affine_grid(inver_theta_2.to(self.device), x.size()).to(
                self.device
            )

            predicted_rotate = F.grid_sample(predicted_base, grid_2)

            predicted_input = F.grid_sample(predicted_rotate, grid_1)

        new_list = []
        #        interpolate_list = []

        for mask_ in self.mask:

            #                print(x.shape)
            #                print(mask_.shape)
            batch_mask = mask_.reshape(1, 1, mask_.shape[-2], mask_.shape[-1]).repeat(x.shape[0], 1, 1, 1).to(self.device)

            batch_mask = torch.tensor(batch_mask, dtype=torch.float).to(self.device)

            rotated_mask = F.grid_sample(batch_mask, grid_2)

            if self.interpolate:
                #                Add reverse affine transform of scale and shear to make all spots in the mask region
                rotated_mask = F.grid_sample(rotated_mask, grid_1)

            rotated_mask[rotated_mask < 0.5] = 0
            rotated_mask[rotated_mask >= 0.5] = 1

            rotated_mask = torch.tensor(rotated_mask, dtype=torch.bool).squeeze().to(self.device)
            

            new_list.append(rotated_mask)

        if self.interpolate:
            ## 1.5 is totally fine for 5% bkg
            predicted_input_revise = revise_size_on_affine_gpu(
                predicted_input,
                new_list,
                x.shape[0],
                inver_theta_1,
                self.device,
                adj_para=adj_mask,
                radius=60,
                coef=1.5,
                pare_reverse=True,
            )
            return (
                predicted_revise,
                predicted_base_inp,
                predicted_input_revise,
                k_out,
                scaler_shear,
                rotation,
                adj_mask,
                new_list,
                x_inp
            )

        else:
            return (
                predicted_revise,
                predicted_base,
                predicted_input,
                k_out,
                scaler_shear,
                rotation,
                adj_mask,
                new_list,
            )

class Encoder(nn.Module):
    def __init__(
        self,
        original_step_size,
        pool_list,
        embedding_size,
        conv_size,
        device,
        num_basis=2,
        fixed_mask=None,
        num_mask=1,
        interpolate=False,
        up_size=800,
    ):
        super(Encoder, self).__init__()

        self.device = device
        blocks = []
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        number_of_blocks = len(pool_list)
        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(pool_list[0], stride=pool_list[0]))
        for i in range(1, number_of_blocks):
            original_step_size = [
                original_step_size[0] // pool_list[i - 1],
                original_step_size[1] // pool_list[i - 1],
            ]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(nn.MaxPool2d(pool_list[i], stride=pool_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)
        original_step_size = [
            original_step_size[0] // pool_list[-1],
            original_step_size[1] // pool_list[-1],
        ]

        input_size = original_step_size[0] * original_step_size[1]
        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.tanh = nn.Tanh()
        self.before = nn.Linear(input_size, 20)
        self.embedding_size = embedding_size
        self.mask_size = num_mask

        self.interpolate = interpolate
        self.up_size = up_size

        if fixed_mask != None:
            # Set the mask_ to upscale mask if the interpolate set True
            if self.interpolate:
                mask_with_inp = []
                for mask_ in fixed_mask:
                    temp_mask = torch.tensor(
                        mask_.reshape(1, 1, self.input_size_0, self.input_size_1),
                        dtype=torch.float,
                    )
                    temp_mask = F.interpolate(
                        temp_mask, size=(self.up_size, self.up_size), mode="bicubic"
                    )
                    temp_mask[temp_mask < 0.5] = 0
                    temp_mask[temp_mask >= 0.5] = 1
                    temp_mask = torch.tensor(temp_mask.squeeze(), dtype=torch.bool)
                    mask_with_inp.append(temp_mask)

                self.mask = mask_with_inp

            else:

                self.mask = fixed_mask
        else:
            self.mask = None

        if num_mask == None:
            self.dense = nn.Linear(20 + num_basis, self.embedding_size)
        else:
            # Set the all the adj parameter to be the same
            self.dense = nn.Linear(20 + num_basis, self.embedding_size + 1)

        self.for_k = nn.Linear(20, num_basis)
        self.norm = nn.LayerNorm(num_basis)
        self.softmax = nn.Softmax()
        self.num_k_sparse = 1

    def ktop(self, x):
        kout = self.for_k(x)
        kout = self.norm(kout)
        kout = self.softmax(kout)
        k_no = kout.clone()

        k = self.num_k_sparse
        with torch.no_grad():
            if k <= kout.shape[1]:
                for raw in k_no:
                    indices = torch.topk(raw, k)[1].to(self.device)
                    mask = torch.ones(raw.shape, dtype=bool).to(self.device)
                    mask[indices] = False
                    raw[mask] = 0
                    raw[~mask] = 1
        return k_no

    def find_type(self):

        return self.emoji

    def find_mask(self):

        return self.mask_size

    def rotate_mask(self):

        return self.mask

    def check_inp(self):

        return self.interpolate

    def check_upsize(self):

        return self.up_size

    def forward(self, x, rotate_value=None):

        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        kout = self.before(out)

        k_out = self.ktop(kout)
        out = torch.cat((kout, k_out), dim=1).to(self.device)
        out = self.dense(out)
        scale_1 = 0.05 * nn.Tanh()(out[:, 0]) + 1
        scale_2 = 0.05 * nn.Tanh()(out[:, 1]) + 1

        if rotate_value != None:

            # use large mask no need to limit to too small range

            rotate = rotate_value.reshape(out[:, 2].shape) + 0.1 * nn.Tanh()(out[:, 2])

        else:

            rotate = nn.ReLU()(out[:, 2])

        shear_1 = 0.1 * nn.Tanh()(out[:, 3])
        #        shear_2 = 0.1*nn.Tanh()(out[:,4])
        #        print(rotate)
        a_1 = torch.cos(rotate)
        #        a_2 = -torch.sin(selection)
        a_2 = torch.sin(rotate)
        a_4 = torch.ones(rotate.shape).to(self.device)
        a_5 = rotate * 0

        # combine shear and strain together
        c1 = torch.stack((scale_1, shear_1), dim=1).squeeze()
        c2 = torch.stack((shear_1, scale_2), dim=1).squeeze()
        c3 = torch.stack((a_5, a_5), dim=1).squeeze()
        scaler_shear = torch.stack((c1, c2, c3), dim=2)

        # Add the rotation after the shear and strain
        b1 = torch.stack((a_1, a_2), dim=1).squeeze()
        b2 = torch.stack((-a_2, a_1), dim=1).squeeze()
        b3 = torch.stack((a_5, a_5), dim=1).squeeze()
        rotation = torch.stack((b1, b2, b3), dim=2)

        if self.interpolate == False:

            grid_1 = F.affine_grid(scaler_shear.to(self.device), x.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x, grid_1)

            grid_2 = F.affine_grid(rotation.to(self.device), x.size()).to(self.device)
            output = F.grid_sample(out_sc_sh, grid_2)

        else:

            x_inp = x.view(-1, 1, self.input_size_0, self.input_size_1)

            x_inp = F.interpolate(
                x_inp, size=(self.up_size, self.up_size), mode="bicubic"
            )

            grid_1 = F.affine_grid(scaler_shear.to(self.device), x_inp.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x_inp, grid_1, mode="bicubic")

            grid_2 = F.affine_grid(rotation.to(self.device), x_inp.size()).to(
                self.device
            )
            output = F.grid_sample(out_sc_sh, grid_2, mode="bicubic")
        #        print(output.shape)

        #        print(out_revise)

        # remove adjust parameter from each mask Region, if multiplied by 0
        mask_parameter = (
            0 * nn.Tanh()(out[:, self.embedding_size : self.embedding_size + 1]) + 1
        )

        if self.interpolate:
            ## Test 1.5 is good for 5% BKG
            out_revise = revise_size_on_affine_gpu(
                output,
                self.mask,
                x.shape[0],
                scaler_shear,
                self.device,
                adj_para=mask_parameter,
                radius=60,
                coef=1.5,
            )

            #            out_revise = F.interpolate(out_revise,size=(self.input_size_0,self.input_size_1),mode = 'bicubic')

            return out_revise, k_out, scaler_shear, rotation, mask_parameter, x_inp

        else:

            #                 out_revise = revise_size_on_affine_gpu(output, self.mask, x.shape[0], scaler_shear,\
            #                                                    self.device,adj_para=mask_parameter,radius=15)

            return output, k_out, scaler_shear, rotation, mask_parameter

class Decoder(nn.Module):
    def __init__(
        self,
        original_step_size,
        up_list,
        embedding_size,
        conv_size,
        device,
        num_basis=2,
    ):
        super(Decoder, self).__init__()

        self.device = device

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(num_basis, original_step_size[0] * original_step_size[1])
        self.cov2d = nn.Conv2d(
            1, conv_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.cov2d_1 = nn.Conv2d(
            conv_size, 1, 3, stride=1, padding=1, padding_mode="zeros"
        )

        blocks = []
        number_of_blocks = len(up_list)
        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        for i in range(number_of_blocks):
            blocks.append(
                nn.Upsample(
                    scale_factor=up_list[i], mode="bilinear", align_corners=True
                )
            )
            original_step_size = [
                original_step_size[0] * up_list[i],
                original_step_size[1] * up_list[i],
            ]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        self.output_size_0 = original_step_size[0]
        self.output_size_1 = original_step_size[1]

        #        input_size = original_step_size[0]*original_step_size[1]
        self.relu_1 = nn.LeakyReLU(0.001)

    def forward(self, x):
        #       print(x.shape)

        out = self.dense(x)
        out = out.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = self.relu_1(out)

        #        out = out.view()
        #        out = self.softmax(out)

        return out

print("Build two types of mask for two training process")

print(f"Set a 200x200 image")

mean_ = np.zeros([200, 200])

print(f"Build the mask 2")

def load_data(data_dir, w_bg=0.60):
    
    '''
    
        data_dir: path of the dataset
        label_index: path of the pretrained rotation 
    
    '''
    
    f = h5py.File(data_dir,'r')
    op4d = f['output4D']
    op4d = op4d[:,:,28:228,28:228]
    op4d = np.transpose(op4d, (1, 0, 3, 2))
    op4d = op4d.reshape(-1,200,200)
    f.close()
    
    if w_bg == 0:
        
        noisy_data = op4d*1e5/4
    
    else:
    
        noisy_data = np.zeros([65536,200,200])
        im=np.zeros([200,200])
        counts_per_probe = 1e5
        for i in tqdm(range(65536),leave=True,total=65536):
            test_img = np.copy(op4d[i])
            qx = np.fft.fftfreq( im.shape[0], d = 1)
            qy = np.fft.fftfreq( im.shape[1], d = 1)
            qya, qxa = np.meshgrid(qy, qx)
            qxa = np.fft.fftshift(qxa)
            qya = np.fft.fftshift(qya) 
            qra2 = qxa**2 + qya**2
            im_bg = 1./( 1 + qra2 / 1e-2**2 )
            im_bg = im_bg / np.sum(im_bg) 
            int_comb = test_img * (1 - w_bg) + im_bg * w_bg 
            int_noisy = np.random.poisson(int_comb * counts_per_probe) / counts_per_probe
            int_noisy = int_noisy*1e5/4
            noisy_data[i] = int_noisy
        
    del op4d
    
    noisy_data = noisy_data.reshape(-1,1,200,200)
#     angle = np.mod(np.arctan2(
#         pre_rot[:,1].reshape(256,256),
#         pre_rot[:,0].reshape(256,256)),np.pi/3).reshape(-1)
    


#     # combine the data and label for test
#     whole_data_with_rotation = []
#     for i in tqdm(range(noisy_data.shape[0]),leave=True, total=noisy_data.shape[0]):
#         whole_data_with_rotation.append([noisy_data[i], angle[i]])
        
    return noisy_data


print("Load the data")

# Set data direction
data_dir = os.path.abspath("Simulated_4dstem/Extremely_Noisy_4DSTEM_Strain_Mapping_Using_CC_ST_AE_Simulated/polycrystal_output4D.mat")


folder_name = ''

print(f"Integrate Label")

folder_name = 'Simulated_4dstem/Extremely_Noisy_4DSTEM_Strain_Mapping_Using_CC_ST_AE_Simulated'
label_rotation_path = folder_name +'/Label_rotation.npy'
label_xx_path = folder_name +'/Label_strain_xx.npy'
label_yy_path = folder_name +'/Label_strain_yy.npy'
label_xy_path = folder_name +'/Label_shear_xy.npy'

label_xx = np.load(label_xx_path).reshape(-1)
label_yy = np.load(label_yy_path).reshape(-1)
label_xy = np.load(label_xy_path).reshape(-1)
label_rot = np.load(label_rotation_path).reshape(-1)

label_cat = np.stack([label_xx,label_yy,label_xy,label_rot])

label_cat = np.transpose(label_cat,(1,0))

print(f"Label shape: {label_cat.shape}")

print(f"Label cat: {label_cat[0]}")

x_train = load_data(data_dir,w_bg=0.25)

print(f"X train shape: {x_train.shape}")

def train_with_label(x,y):
    x_with_y = []
    for i in tqdm(range(x.shape[0]),leave=True, total=x.shape[0]):
        x_with_y.append([x[i], y[i]])
    return x_with_y

train_set = train_with_label(x_train,label_cat)

print(f"Train set shape: {train_set[0][1].shape}")

del(x_train)

device = torch.device('cuda')


print(f"ResNet 50")
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features

print(f"Number of features: {num_ftrs}")

resnet50.fc = nn.Linear(num_ftrs, 4)

resnet50.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False
        )

print(f"Resnet 50")
print(resnet50)

print("Loss Function")


def loss_function_strain(model,
                      train_iterator,
                      optimizer,
                      device,
                     ):

    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0
    
    for x,y in tqdm(train_iterator, leave=True, total=len(train_iterator)):
     

        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)
        optimizer.zero_grad()
        

        y_pred = model(x)
        
        loss = F.mse_loss(y,y_pred)
    
        train_loss += loss.item()
        loss.backward()
        # update the weights
        optimizer.step()

    return train_loss


print("Train Function")

folder_path = 'Resnet50_25Per_4dstem' 

def train(data_,
         model,
         optimizer,
         epochs=5000,   
         learning_rate = 1e-4,
         max_rate = 1e-3,
         batch_size = 64,
         epoch_ = None,
         file_path = None,
         folder_path=folder_path,
         step_size_up=50,
         best_train_loss= None,
         set_scheduler = True,
        ):
        
            make_folder(folder_path)            
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            # put model on device
            model.to(device)
            print('.........b step.........')
            patience = 0


            if set_scheduler:

                lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=max_rate,
                                                      step_size_up=step_size_up,cycle_momentum=False)
            else: 

                lr_scheduler = None


            print('..........successfully generate model')

            train_iterator = DataLoader(data_, batch_size=batch_size, shuffle=True, num_workers=0)


            N_EPOCHS = epochs

            if best_train_loss == None:
                best_train_loss = float('inf')


            if epoch_==None:
                start_epoch = 0
            else:
                start_epoch = epoch_+1
            print('...........successfully generate train interator')

            for epoch in range(start_epoch,epochs):

                optimizer.param_groups[0]['lr'] = learning_rate   

                train = loss_function_strain(model,train_iterator,
                                      optimizer,device)

                input_length = len(train_iterator)


                train_loss = train

                train_loss /= input_length

        #        VAE_L /= len(train_iterator)
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
        #        print(f'......... VAE Loss: {VAE_L:.4f}')
                print('.............................')

                checkpoint = {
                    "net":model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'trainloss': train_loss,
                }
                if epoch >=0:
                    lr_ = optimizer.param_groups[0]['lr']
    #                    l1_form = format(coef_1,'.4f')
                    file_path = folder_path+f'/0215_25Per_epoch:{epoch:04d}_lr:{lr_:.6f}_trainloss:{train_loss:.6f}_.pkl'

                    if best_train_loss >= train_loss:
                        best_train_loss = train_loss
                        torch.save(checkpoint, file_path)
                    else:
                        patience+=1
                        
                        if patience >=100:
                            break


                                

                if lr_scheduler!= None:
                    lr_scheduler.step()
                    
                if epoch==epochs-1:
                    del model

optimizer = optim.Adam(resnet50.parameters(),lr=1e-4)

# train(train_set,resnet50,optimizer)

def Generate_scale_shear_loss(exx_Shuyu,eyy_Shuyu,exy_Shuyu,label_xx,label_yy,label_xy):
    dif_shuyu_xx = exx_Shuyu - label_xx
    dif_shuyu_yy = eyy_Shuyu - label_yy
    dif_shuyu_xy = exy_Shuyu - label_xy
    mae_shuyu_xx = np.mean(abs(dif_shuyu_xx)) 
    mae_shuyu_yy = np.mean(abs(dif_shuyu_yy)) 
    mae_shuyu_xy = np.mean(abs(dif_shuyu_xy)) 
    
    combine_loss = mae_shuyu_xx + mae_shuyu_yy + mae_shuyu_xy
    
    return combine_loss

def basis2probe(rotation_,
                scale_shear_):

    M = []
    
    for i in tqdm(range(rotation_.shape[0]),leave=True,total=rotation_.shape[0]):
        
        # switch cosine and sine into radius
        theta = np.arctan2(rotation_[i][1], rotation_[i][0]) 
        
        # generate scale transformation matrix and rotation transformation matrix with parameters from corresponding input index
        xx = scale_shear_[i][0]
        yy = scale_shear_[i][3]
        xy = scale_shear_[i][1]
        yx = scale_shear_[i][2]

        # generate rotation matrix with cosine and sine value
        r = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
            ])
        
        # generate scale and shear with input value 
        t = np.array([
            [xx,xy],
            [yx,yy]
            ])
        
        # matrix multiplication 
        m = np.linalg.inv(t) @ np.linalg.inv(r)

        M.append(m)

    M  = np.array(M)
    return M

def inverse_base(name_of_file, input_mask_list, coef=2, radius=7):
    """generate updated mask list that center the spots

    Args:
        name_of_file (string): file directory
        input_mask_list (list of tensor): mask list used for updating
        coef (float): threshold for center the spots. Defaults to 2.
        radius (int): radius of updated mask. Defaults to 7.

    Returns:
        list of tensor, tensor: mask list and mask
    """
    # Load h5 file with correct path structure
    load_file = h5py.File("py4DSTEM_strain.h5", "r")
    # Load strain map data from correct dataset path
    load_base = load_file["4DSTEM_experiment"]["data"]["realslices"]["strain_map"]["data"][0]
    
    # Resize load_base to match mask size (200x200)
    target_size = input_mask_list[0].shape
    load_base_resized = cv2.resize(load_base, (target_size[-1], target_size[-2]))
    
    # Reshape base into tensor with correct dimensions
    base_ = torch.tensor(load_base_resized, dtype=torch.float).reshape(
        1, 1, target_size[-2], target_size[-1]
    )
    
    # Update mask list region using center_mask_list_function
    center_mask_list, rotate_center = center_mask_list_function(
        base_, input_mask_list, coef, radius=radius
    )

    return center_mask_list, rotate_center

def center_mask_list_function(image, mask_list, coef, radius=7):
    # create mask list
    center_mask_list = []
    # create image with zero value
    mean_ = np.zeros([image.shape[-2], image.shape[-1]])

    input_size = mask_list[0].shape[-1]
    up_size = image.shape[-1]
    # upgrid image if necessary
    if input_size != up_size:
        mask_list = upsample_mask(mask_list, input_size, up_size)

    for j, mask in enumerate(mask_list):
        mask_ = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])

        new_image = image * mask_
    # compute coordinate with center of mass 
        center_x, center_y = center_of_mass(new_image.squeeze(), mask_.squeeze(), coef)

        center_x = int(np.round(np.array(center_x)))
        center_y = int(np.round(np.array(center_y)))
        print(center_x, center_y)
    # create small mask region using center coordinate
        small_mask = mask_function(
            mean_, radius=radius, center_coordinates=(center_y, center_x)
        )
    # switch type into tensor
        small_mask = torch.tensor(small_mask, dtype=torch.bool)

        center_mask_list.append(small_mask)
    # change mask size if necessary
    if input_size != up_size:
        center_mask_list = upsample_mask(center_mask_list, up_size, input_size)
    # create whole mask region in one image
    rotate_mask_up = torch.clone(center_mask_list[0])

    for i in range(1, len(center_mask_list)):
        rotate_mask_up += center_mask_list[i]

    return center_mask_list, rotate_mask_up

def mask_function(img, radius=7, center_coordinates=(100, 100)):
    image = np.copy(img.squeeze())
    thickness = -1
    color = 100
    image_2 = cv2.circle(image, center_coordinates, radius, color, thickness)
    image_2 = np.array(image_2)
    mask = image_2 == 100
    mask = np.array(mask)
    return mask

def center_of_mass(img, mask, coef=1.5):
    """Compute center of mass for a masked image.
    
    Args:
        img: Input image tensor
        mask: Mask tensor
        coef: Threshold coefficient (default: 1.5)
        
    Returns:
        tuple: (x, y) coordinates of center of mass
    """
    # Convert inputs to tensors if they're not already
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float)
    
    # Get image shape
    h, w = img.shape[-2:]
    
    # Create coordinate grids as tensors
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float), 
                         torch.arange(w, dtype=torch.float),
                         indexing='ij')
    
    # Apply mask and get threshold
    masked_img = img * mask
    threshold = coef * masked_img.mean()
    binary = (masked_img > threshold)
    
    # Calculate center of mass
    total_mass = binary.sum()
    if total_mass == 0:
        return h//2, w//2  # Return center if no mass found
        
    center_x = (x * binary).sum() / total_mass
    center_y = (y * binary).sum() / total_mass
    
    return center_x.item(), center_y.item()

def upsample_mask(mask_list, input_size, up_size):
    """Upsample a list of masks to a new size.
    
    Args:
        mask_list: List of mask tensors
        input_size: Original size
        up_size: Target size
        
    Returns:
        list: List of upsampled mask tensors
    """
    up_mask_list = []
    for mask in mask_list:
        mask = mask.reshape(1, 1, input_size, input_size)
        mask = torch.tensor(mask, dtype=torch.float)
        up_mask = F.interpolate(mask, size=(up_size, up_size), mode='nearest')
        up_mask = torch.tensor(up_mask.squeeze(), dtype=torch.bool)
        up_mask_list.append(up_mask)
    return up_mask_list

mask_0 = mask_function(mean_, radius=11, center_coordinates=(99, 162))
mask_1 = mask_function(mean_, radius=11, center_coordinates=(154, 130))
mask_2 = mask_function(mean_, radius=11, center_coordinates=(154, 68))
mask_3 = mask_function(mean_, radius=11, center_coordinates=(99, 36))
mask_4 = mask_function(mean_, radius=11, center_coordinates=(45, 68))
mask_5 = mask_function(mean_, radius=11, center_coordinates=(45, 130))
# Combine all components together

mask_up_2 = mask_0 + mask_1 + mask_2 + mask_3 + mask_4 + mask_5

# save the maskup2

plt.imsave("mask_up_2.png", mask_up_2)

mask_0 = torch.tensor(mask_0)
mask_1 = torch.tensor(mask_1)
mask_2 = torch.tensor(mask_2)
mask_3 = torch.tensor(mask_3)
mask_4 = torch.tensor(mask_4)
mask_5 = torch.tensor(mask_5)

mask_list_2 = [mask_0, mask_1, mask_2, mask_3, mask_4, mask_5]

def rotate_mask_list(mask_list, theta_):
    modified_mask_list_2 = []
    a_1 = torch.cos(theta_).reshape(1, 1)
    a_2 = torch.sin(theta_).reshape(1, 1)
    a_5 = torch.zeros([1, 1])
    b1 = torch.stack((a_1, a_2), dim=1)
    b2 = torch.stack((-a_2, a_1), dim=1)
    b3 = torch.stack((a_5, a_5), dim=1)
    rotation = torch.stack((b1, b2, b3), dim=2)
    rotation = rotation.reshape(1, 2, 3)
    zero_tensor = torch.zeros(mask_list[0].shape)
    print(zero_tensor.shape)
    zero_tensor = zero_tensor.reshape(
        1, 1, zero_tensor.shape[-2], zero_tensor.shape[-1]
    )
    grid_2 = F.affine_grid(rotation, zero_tensor.size())

    for mask_ in mask_list:
        tmp = torch.clone(mask_).reshape(1, 1, mask_.shape[-2], mask_.shape[-1])
        tmp = torch.tensor(tmp, dtype=torch.float)
        rotate_tmp = F.grid_sample(tmp, grid_2)
        rotate_tmp = torch.tensor(rotate_tmp, dtype=torch.bool).squeeze()
        modified_mask_list_2.append(rotate_tmp)

    rotate_mask_up = torch.clone(modified_mask_list_2[0])

    for i in range(1, len(mask_list)):
        rotate_mask_up += modified_mask_list_2[i]

    return modified_mask_list_2, rotate_mask_up

# Initialize theta_ (rotation angle in radians)
theta_ = torch.zeros(1, dtype=torch.float32)

modified_mask_list_2, rotate_mask_up = rotate_mask_list(mask_list_2, theta_)

# Load the base image
load_file = h5py.File("py4DSTEM_strain.h5", "r")
load_base = load_file["4DSTEM_experiment"]["data"]["realslices"]["strain_map"]["data"][0]

# Resize and convert to tensor
target_size = mask_list_2[0].shape
load_base_resized = cv2.resize(load_base, (target_size[-1], target_size[-2]))
base_ = torch.tensor(load_base_resized, dtype=torch.float).reshape(1, 1, target_size[-2], target_size[-1])

# Generate centered masks using center_mask_list_function
center_mask_list, rotate_center = center_mask_list_function(base_, mask_list_2, coef=2, radius=7)

# Model parameters
en_original_step_size = [200, 200]
pool_list = [5, 4, 2]
de_original_step_size = [5, 5]
up_list = [2, 4, 5]
embedding_size = 4
conv_size = 128
num_basis = 1
up_size = 800

num_mask_2 = 6
fixed_mask_2 = modified_mask_list_2
interpolate_2 = True

def make_model_2(
    device,
    en_original_step_size=en_original_step_size,
    pool_list=pool_list,
    de_original_step_size=de_original_step_size,
    up_list=up_list,
    embedding_size=embedding_size,
    conv_size=conv_size,
    num_basis=num_basis,
    num_mask=num_mask_2,
    fixed_mask=fixed_mask_2,
    learning_rate=3e-5,
    interpolate=interpolate_2,
    up_size=up_size,
):

    encoder = Encoder(
        original_step_size=en_original_step_size,
        pool_list=pool_list,
        embedding_size=embedding_size,
        conv_size=conv_size,
        device=device,
        num_basis=num_basis,
        fixed_mask=fixed_mask,
        num_mask=num_mask,
        interpolate=interpolate,
        up_size=up_size,
    ).to(device)
    decoder = Decoder(
        original_step_size=de_original_step_size,
        up_list=up_list,
        embedding_size=embedding_size,
        conv_size=conv_size,
        device=device,
        num_basis=num_basis,
    ).to(device)
    join = Joint(encoder, decoder, device).to(device)

    optimizer = optim.Adam(join.parameters(), lr=learning_rate)

    checkpoint = "2nd_train_weight_25Per.pkl"
    pre_weight = torch.load(checkpoint)
    
    # Remove 'module.' prefix from state dict keys if present
    new_state_dict = {}
    for k, v in pre_weight["net"].items():
        name = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[name] = v
    
    join.load_state_dict(new_state_dict)

    return encoder, decoder, join, optimizer


encoder,decoder, join, optimizer = make_model_2(device,learning_rate = 3e-5,fixed_mask = modified_mask_list_2,interpolate = True)

check_path = '0215_25Per_epoch-0129_lr-0.000100_trainloss-0.000007_.pkl'

check_point = torch.load(check_path)
# Remove 'module.' prefix from state dict keys if present
new_state_dict = {}
for k, v in check_point['net'].items():
    name = k.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[name] = v

# Load the cleaned state dict into resnet50 model
resnet50.load_state_dict(new_state_dict)
resnet50 = resnet50.to(device)  # Move model to the same device as input data

learned_rotation_1 = np.zeros([65536,2])
learned_rotation_2 = np.zeros([65536,2])
learned_scale_shear_ = np.zeros([65536,4])
encoder_out = np.zeros([65536,4])
decoder_out = np.zeros([65536,4])

train_iterator = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)


# shift_angle = -23
# gouba = np.mod(shift_angle + 1*np.rad2deg(np.arctan2(
#                         learned_rotation_1[:,1].reshape(-1),
#                         learned_rotation_1[:,0].reshape(-1))
#                                          ),60)

# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].imshow(gouba.reshape(256,256),cmap = 'RdBu_r',clim=[0,60])
# ax[1].hist(gouba.reshape(-1),200,range=[0,60]);

def is_bn_layer(name, param):
    """Check if the parameter belongs to a batch normalization layer"""
    return 'bn' in name.lower() or isinstance(param, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))

def normalize_direction(direction, parameters, normalization='filter'):
    """
    Normalize the direction vector with special handling for different layer types.
    
    Args:
        direction: list of tensors to normalize
        parameters: list of model parameters
        normalization: type of normalization ('filter', 'layer', or 'model')
    """
    if normalization == 'filter':
        if isinstance(parameters[0], tuple):  # Named parameters
            for (name, p), (_, d) in zip(parameters, direction):
                if is_bn_layer(name, p):  # BatchNorm layers
                    if hasattr(p, 'running_var'):
                        d.mul_(torch.sqrt(p.running_var + p.eps))
                elif len(d.size()) >= 2:  # Conv/Linear layers
                    for c in range(d.size(0)):
                        filter_norm = p[c].norm()
                        if filter_norm > 0:  # Avoid division by zero
                            d[c].mul_(filter_norm / (d[c].norm() + 1e-10))
                else:  # Other 1D tensors (biases, etc.)
                    p_norm = p.norm()
                    if p_norm > 0:  # Avoid division by zero
                        d.mul_(p_norm / (d.norm() + 1e-10))
        else:  # Unnamed parameters
            for d, p in zip(direction, parameters):
                if len(d.size()) >= 2:  # Conv/Linear layers
                    for c in range(d.size(0)):
                        filter_norm = p[c].norm()
                        if filter_norm > 0:  # Avoid division by zero
                            d[c].mul_(filter_norm / (d[c].norm() + 1e-10))
                else:  # Other 1D tensors (biases, etc.)
                    p_norm = p.norm()
                    if p_norm > 0:  # Avoid division by zero
                        d.mul_(p_norm / (d.norm() + 1e-10))
    
    elif normalization == 'layer':
        if isinstance(parameters[0], tuple):  # Named parameters
            for (name, p), (_, d) in zip(parameters, direction):
                if is_bn_layer(name, p):  # BatchNorm layers
                    if hasattr(p, 'running_var'):
                        d.mul_(torch.sqrt(p.running_var + p.eps))
                else:
                    d_norm = d.norm()
                    p_norm = p.norm()
                    if d_norm > 0 and p_norm > 0:  # Avoid division by zero
                        d.mul_(p_norm / d_norm)
        else:  # Unnamed parameters
            for d, p in zip(direction, parameters):
                d_norm = d.norm()
                p_norm = p.norm()
                if d_norm > 0 and p_norm > 0:  # Avoid division by zero
                    d.mul_(p_norm / d_norm)
    
    elif normalization == 'model':
        d_norm = get_model_norm(direction)
        if isinstance(parameters[0], tuple):  # Named parameters
            p_norm = get_model_norm([p for _, p in parameters])
        else:  # Unnamed parameters
            p_norm = get_model_norm(parameters)
        if d_norm > 0 and p_norm > 0:  # Avoid division by zero
            for d in direction:
                if isinstance(d, tuple):  # Named parameters
                    d[1].mul_(p_norm / d_norm)
                else:  # Unnamed parameters
                    d.mul_(p_norm / d_norm)
    
    return direction

def get_model_parameters(model):
    """Get model parameters with their names"""
    return [(name, p.data) for name, p in model.named_parameters()]

def rand_uniform_like(parameters):
    """Generate random direction with the same shape as parameters"""
    direction = []
    if isinstance(parameters[0], tuple):  # Named parameters
        for name, p in parameters:
            if is_bn_layer(name, p):  # BatchNorm layers
                flat_d = torch.randn_like(p)
                if hasattr(p, 'running_var'):
                    flat_d.mul_(torch.sqrt(p.running_var + p.eps))
            elif p.dim() >= 2:  # Conv/Linear layers
                # Initialize with random values
                flat_d = torch.randn_like(p)
                # Normalize each output channel
                for c in range(p.size(0)):
                    filter_norm = p[c].norm()
                    if filter_norm > 0:  # Avoid division by zero
                        flat_d[c].mul_(filter_norm / (flat_d[c].norm() + 1e-10))
            else:  # Other 1D tensors (biases, etc.)
                flat_d = torch.randn_like(p)
                p_norm = p.norm()
                if p_norm > 0:  # Avoid division by zero
                    flat_d.mul_(p_norm / (flat_d.norm() + 1e-10))
            direction.append((name, flat_d))
    else:  # Unnamed parameters (just tensors)
        for p in parameters:
            if p.dim() >= 2:  # Conv/Linear layers
                # Initialize with random values
                flat_d = torch.randn_like(p)
                # Normalize each output channel
                for c in range(p.size(0)):
                    filter_norm = p[c].norm()
                    if filter_norm > 0:  # Avoid division by zero
                        flat_d[c].mul_(filter_norm / (flat_d[c].norm() + 1e-10))
            else:  # Other 1D tensors (biases, etc.)
                flat_d = torch.randn_like(p)
                p_norm = p.norm()
                if p_norm > 0:  # Avoid division by zero
                    flat_d.mul_(p_norm / (flat_d.norm() + 1e-10))
            direction.append(flat_d)
    return direction

def clone_parameters(parameters):
    """Deep copy of model parameters"""
    return [(name, p.clone()) for name, p in parameters]

def set_parameters(model, parameters):
    """Set model parameters"""
    for (name, p), (_, new_p) in zip(model.named_parameters(), parameters):
        p.data = new_p

def get_model_norm(parameters, order=2):
    """Compute the norm of model parameters"""
    if isinstance(parameters[0], tuple):  # If parameters contain names
        return torch.sqrt(sum(p.norm(order).pow(2) for _, p in parameters))
    else:  # If parameters are just tensors
        return torch.sqrt(sum(p.norm(order).pow(2) for p in parameters))

def scale_direction(direction, scale):
    """Scale a direction vector by some value"""
    if isinstance(direction[0], tuple):  # If direction contains names
        for _, d in direction:
            d.mul_(scale)
    else:  # If direction contains just tensors
        for d in direction:
            d.mul_(scale)
    return direction

def add_direction(parameters, direction):
    """Add a direction vector to parameters"""
    if isinstance(parameters[0], tuple):  # If parameters contain names
        for (_, p), (_, d) in zip(parameters, direction):
            p.data.add_(d)
    else:  # If parameters are just tensors
        for p, d in zip(parameters, direction):
            p.data.add_(d)

def sub_direction(parameters, direction):
    """Subtract a direction vector from parameters"""
    if isinstance(parameters[0], tuple):  # If parameters contain names
        for (_, p), (_, d) in zip(parameters, direction):
            p.data.sub_(d)
    else:  # If parameters are just tensors
        for p, d in zip(parameters, direction):
            p.data.sub_(d)

def make_orthogonal(direction_one):
    """Generate a random direction orthogonal to direction_one"""
    direction_two = rand_uniform_like(direction_one)
    
    if isinstance(direction_one[0], tuple):  # Named parameters
        for (_, d2), (_, d1) in zip(direction_two, direction_one):
            d2.sub_((d2 * d1).sum() / (d1 * d1).sum() * d1)
    else:  # Unnamed parameters
        for d2, d1 in zip(direction_two, direction_one):
            d2.sub_((d2 * d1).sum() / (d1 * d1).sum() * d1)
    return direction_two

def compute_loss_landscape(model, train_iterator, optimizer, device, steps=41, distance=1):
    """
    Compute the loss landscape along a planar subspace of the parameter space.
    
    Args:
        model: The model to analyze
        train_iterator: DataLoader for training data
        optimizer: The optimizer
        device: Device to run computations on
        steps: Number of steps in each direction (default: 41)
        distance: Maximum distance from start point (default: 0.2)
        
    Returns:
        loss_surface: 2D array of loss values
    """
    try:
        # Get starting parameters and save original weights
        with torch.no_grad():
            start_point = get_model_parameters(model)
            original_weights = clone_parameters(start_point)
        
        # Generate random orthogonal directions
        dir_one = rand_uniform_like(start_point)
        dir_two = make_orthogonal(dir_one)
        
        # Normalize directions
        dir_one = normalize_direction(dir_one, start_point, normalization='filter')
        dir_two = normalize_direction(dir_two, start_point, normalization='filter')
        
        # Scale directions to match steps and total distance
        model_norm = get_model_norm(start_point)
        
        # Scale to match steps and distance
        dir_one_norm = get_model_norm(dir_one)
        dir_two_norm = get_model_norm(dir_two)
        scale_direction(dir_one, ((model_norm * distance) / steps) / dir_one_norm)
        scale_direction(dir_two, ((model_norm * distance) / steps) / dir_two_norm)
        
        # Move start point to corner
        scale_direction(dir_one, steps / 2)
        scale_direction(dir_two, steps / 2)
        current_point = clone_parameters(original_weights)
        sub_direction(current_point, dir_one)
        sub_direction(current_point, dir_two)
        scale_direction(dir_one, 2.0 / steps)
        scale_direction(dir_two, 2.0 / steps)
        
        # Initialize loss surface
        loss_surface = np.zeros((steps, steps))
        
        # Compute loss landscape
        with torch.no_grad():
            for i in tqdm(range(steps), desc="Computing loss landscape"):
                for j in range(steps):
                    # Set model weights to current grid point
                    set_parameters(model, current_point)
                    
                    # Get batch of data
                    x, y = next(iter(train_iterator))
                    x = x.to(device, dtype=torch.float)
                    y = y.to(device, dtype=torch.float)
                    
                    # Compute loss at current point
                    y_pred = model(x)
                    loss = F.mse_loss(y_pred, y)
                    
                    loss_surface[i, j] = loss.item()
                    
                    # Move in direction two
                    add_direction(current_point, dir_two)
                
                # Move in direction one and reset direction two
                add_direction(current_point, dir_one)
                sub_direction(current_point, scale_direction(clone_parameters(dir_two), steps))
                
                # Clear GPU memory periodically
                if i % 2 == 0:
                    torch.cuda.empty_cache()
        
        # Create directory if it doesn't exist
        save_dir = 'loss_landscapes'
        os.makedirs(save_dir, exist_ok=True)
        
        # Save results
        filename = f"resnet_steps{steps}_dist{distance:.1f}_loss_landscape.npz"
        save_path = os.path.join(save_dir, filename)
        np.savez(save_path, 
                 loss_surface=loss_surface,
                 x_coordinates=np.linspace(-distance, distance, steps),
                 y_coordinates=np.linspace(-distance, distance, steps))
        
        print(f"Successfully saved loss landscape data to {save_path}")
        
    except Exception as e:
        print(f"Error during loss landscape computation: {e}")
        raise
        
    finally:
        # Restore original weights
        set_parameters(model, original_weights)
        
    return loss_surface



loss_surface = compute_loss_landscape(resnet50, train_iterator, optimizer, device, steps=101, distance=0.5)
print(loss_surface)