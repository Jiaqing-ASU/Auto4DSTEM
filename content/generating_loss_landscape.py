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

warnings.filterwarnings("ignore")

params = {
    "axes.titlesize": 20,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "ytick.labelsize": 16,
    "xtick.labelsize": 16,
}

pylab.rcParams.update(params)

# Set up logging
log_filename = f'training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print(f"torch version: {torch.__version__}")

print(f"cuda available: {torch.cuda.is_available()}")
print(f"Set the mask function")

def mask_function(img, radius=7, center_coordinates=(100, 100)):
    image = np.copy(img.squeeze())
    thickness = -1
    color = 100
    image_2 = cv2.circle(image, center_coordinates, radius, color, thickness)
    image_2 = np.array(image_2)
    mask = image_2 == 100
    mask = np.array(mask)

    return mask

print(f"Set a 200x200 image")

mean_ = np.zeros([200, 200])

print(f"Build the mask 1")

mask_0 = mask_function(mean_, radius=20, center_coordinates=(100, 100))
mask_1 = mask_function(mean_, radius=85, center_coordinates=(100, 100))
# mask_2 = mask_function(mean_,radius=12,center_coordinates=(163,113))
# mask_3 = mask_function(mean_,radius=12,center_coordinates=(144,51))
# mask_4 = mask_function(mean_,radius=12,center_coordinates=(80,38))
# mask_5 = mask_function(mean_,radius=12,center_coordinates=(37,86))

# Combine all components together
mask_up_1 = ~mask_0 * mask_1
new_mask_1 = torch.tensor(mask_up_1)
mask_list_1 = [new_mask_1]

print(f"Build the mask 2")

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
new_mask_2 = torch.tensor(mask_up_2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(mask_up_1)
ax[1].imshow(mask_up_2)
plt.savefig("mask_up_1_2.png")

print(f"load the data")
data_dir = os.path.abspath(
    "./Simulated_4dstemExtremely_Noisy_4DSTEM_Strain_Mapping_Using_CC_ST_AE_Simulated/polycrystal_output4D.mat"
)

print(f" Load data function 2 for second training process")

def load_data_4_process2(data_dir, pre_rot, w_bg=0.60):
    """

    data_dir: path of the dataset
    label_index: path of the pretrained rotation

    """

    f = h5py.File(data_dir, "r")
    op4d = f["output4D"]
    op4d = op4d[:, :, 28:228, 28:228]
    op4d = np.transpose(op4d, (1, 0, 3, 2))
    op4d = op4d.reshape(-1, 200, 200)
    f.close()

    if w_bg == 0:
        noisy_data = op4d * 1e5 / 4

    else:
        noisy_data = np.zeros([65536, 200, 200])
        im = np.zeros([200, 200])
        counts_per_probe = 1e5
        for i in tqdm(range(65536), leave=True, total=65536):
            test_img = np.copy(op4d[i])
            qx = np.fft.fftfreq(im.shape[0], d=1)
            qy = np.fft.fftfreq(im.shape[1], d=1)
            qya, qxa = np.meshgrid(qy, qx)
            qxa = np.fft.fftshift(qxa)
            qya = np.fft.fftshift(qya)
            qra2 = qxa**2 + qya**2
            im_bg = 1.0 / (1 + qra2 / 1e-2**2)
            im_bg = im_bg / np.sum(im_bg)
            int_comb = test_img * (1 - w_bg) + im_bg * w_bg
            int_noisy = (
                np.random.poisson(int_comb * counts_per_probe) / counts_per_probe
            )
            int_noisy = int_noisy * 1e5 / 4
            noisy_data[i] = int_noisy

    del op4d

    noisy_data = noisy_data.reshape(-1, 1, 200, 200)
    angle = np.mod(
        np.arctan2(pre_rot[:, 1].reshape(256, 256), pre_rot[:, 0].reshape(256, 256)),
        np.pi / 3,
    ).reshape(-1)

    # combine the data and label for test
    whole_data_with_rotation = []
    for i in tqdm(range(noisy_data.shape[0]), leave=True, total=noisy_data.shape[0]):
        whole_data_with_rotation.append([noisy_data[i], angle[i]])

    return whole_data_with_rotation

print(f"autoencoder")

def crop_small_square(center_coordinates, radius=50):

    center_coordinates = torch.round(center_coordinates)

    x_coor = (int(center_coordinates[0] - radius), int(center_coordinates[0] + radius))

    y_coor = (int(center_coordinates[1] - radius), int(center_coordinates[1] + radius))

    return x_coor, y_coor

def center_of_mass(img, mask, coef=1.5):

    cor_x, cor_y = torch.where(mask != 0)
    mean_mass = torch.mean(img[mask])
    mass = F.relu(img[mask] - coef * mean_mass)
    img_after = torch.clone(img)
    img_after[mask] = mass

    sum_mass = torch.sum(mass)

    if sum_mass == 0:
        weighted_x = torch.sum(cor_x) / len(cor_x)
        weighted_y = torch.sum(cor_y) / len(cor_y)
    else:
        weighted_x = torch.sum(cor_x * mass) / sum_mass

        weighted_y = torch.sum(cor_y * mass) / sum_mass
    return weighted_x, weighted_y

def revise_size_on_affine_gpu(
    image,
    mask_list,
    batch_size,
    theta,
    device,
    adj_para=None,
    radius=12,
    coef=2,
    pare_reverse=False,
):

    #    img0 = np.zeros([image.shape[-1],image.shape[-1]])
    # Add another mask dealing with the diffraction pattern only
    np_img = np.zeros([radius * 2, radius * 2])
    dot_size = int(4 * image.shape[-1] / 200)
    small_square_mask = mask_function(
        np_img, radius=dot_size, center_coordinates=(radius, radius)
    )
    small_square_mask = torch.tensor(small_square_mask, dtype=torch.bool).to(device)

    img = torch.clone(image).to(device)
    #    print(img.shape)
    identity = (
        torch.tensor([0, 0, 1], dtype=torch.float)
        .reshape(1, 1, 3)
        .repeat(batch_size, 1, 1)
        .to(device)
    )
    new_theta = torch.cat((theta, identity), axis=1).to(device)
    # Clone the tensor before inverse operation
    new_theta_clone = new_theta.clone()
    inver_theta = torch.linalg.inv(new_theta_clone)[:, 0:2].to(device)
    #    print(theta.shape)
    ##    print(inver_theta.shape)
    #    print('....')
    for j, mask in enumerate(mask_list):
        if mask.shape[0] != batch_size:
            mask_ = (
                mask.squeeze()
                .unsqueeze(0)
                .unsqueeze(1)
                .repeat(batch_size, 1, 1, 1)
                .to(device)
            )
        else:
            mask_ = mask.reshape(batch_size, 1, mask.shape[-2], mask.shape[-1]).to(
                device
            )

        new_image = image * mask_.to(device)

        for i in range(batch_size):
            center_x, center_y = center_of_mass(
                new_image[i].squeeze(), mask_[i].squeeze(), coef
            )

            center = torch.tensor([center_x, center_y]).to(device)
            x_coor, y_coor = crop_small_square(
                center_coordinates=center.clone(), radius=radius
            )

            # crop small square on image after affine transformation

            small_image = (
                img[i]
                .squeeze()[x_coor[0] : x_coor[1], y_coor[0] : y_coor[1]]
                .unsqueeze(0)
                .unsqueeze(1)
                .clone()
                .to(device)
            )
            re_grid = F.affine_grid(
                inver_theta[i].unsqueeze(0).to(device), small_image.size()
            ).to(device)

            if adj_para == None:

                re_aff_small_image = F.grid_sample(small_image, re_grid, mode="bicubic")
                img[i, :, x_coor[0] : x_coor[1], y_coor[0] : y_coor[1]] = (
                    re_aff_small_image.squeeze()
                )

            else:

                small_image_copy = torch.clone(small_image.squeeze()).to(device)
                # Use the same parameter to fit all the diffraction patterns in mask reigon
                if pare_reverse:
                    small_image_copy[small_square_mask] /= adj_para[i]
                else:
                    small_image_copy[small_square_mask] *= adj_para[i]

                small_image_copy = small_image_copy.unsqueeze(0).unsqueeze(1)

                re_aff_small_image = F.grid_sample(
                    small_image_copy, re_grid, mode="bicubic"
                )
                img[i, :, x_coor[0] : x_coor[1], y_coor[0] : y_coor[1]] = (
                    re_aff_small_image.squeeze()
                )

    return img

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

# narrow the range of the adjust parameter for the mask region, since it is not the noise free dataset,
# this will increase the background noise's influence to the MSE loss
#
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {device}")

print(f"check mask_list_2 is list: {type(mask_list_2) == list}")

print("setting parameters shared with both model architectures")

en_original_step_size = [200, 200]
pool_list = [5, 4, 2]
de_original_step_size = [5, 5]
up_list = [2, 4, 5]
embedding_size = 4
conv_size = 128
num_basis = 1
up_size = 800

print(
    " Use the generated rotation and scale shear to check on the base position and create new mask region"
)

def inverse_base(name_of_file, input_mask_list, coef=2, radius=7):

    load_file = h5py.File(name_of_file + ".h5", "r")
    load_base = load_file["base"][0].squeeze()

    base_ = torch.tensor(load_base, dtype=torch.float).reshape(
        1, 1, load_base.shape[-1], load_base.shape[-2]
    )

    center_mask_list, rotate_center = center_mask_list_function(
        base_, input_mask_list, coef, radius=radius
    )

    return center_mask_list, rotate_center

file_py4DSTEM = "py4DSTEM_strain.h5"
f = h5py.File(file_py4DSTEM, "r")
strain_map = f["4DSTEM_experiment"]["data"]["realslices"]["strain_map"]["data"][:]

rotation_ = np.load("25Percent_rotation_071323.npy")

print(f"rotation_.shape: {rotation_.shape}")

print("Add random rotation to the rotation from first training process ")

def add_disturb(rotation, dist=-15):
    angles = np.rad2deg(
        np.arctan2(rotation[:, 1].reshape(256, 256), rotation[:, 0].reshape(256, 256))
    )
    angles = angles.reshape(-1)
    angles = angles + dist

    angles = np.deg2rad(angles)

    new_rotation = np.zeros([angles.shape[0], 2])

    cos_ = np.cos(angles)
    sin_ = np.sin(angles)

    new_rotation[:, 0] = cos_
    new_rotation[:, 1] = sin_

    return new_rotation

new_rotation = add_disturb(rotation_)

theta_Colin = np.mod(np.rad2deg(strain_map[:, :, 3]), 60)
theta_Shuyu = np.mod(
    np.rad2deg(
        np.arctan2(
            new_rotation[:, 1].reshape(256, 256), new_rotation[:, 0].reshape(256, 256)
        )
    ),
    60.0,
)
angle_diff = (theta_Shuyu - theta_Colin).reshape(-1)
index_ = np.where(angle_diff < 0)
angle_diff[index_] += 60

theta_ = np.pi * np.mean(angle_diff) / 180
theta_ = torch.tensor(theta_, dtype=torch.float)

print(f"theta_: {theta_}")

subtract = torch.tensor(angle_diff, dtype=torch.float)

print(f"subtract: {subtract}")

print(f"torch.var(subtract, unbiased=False): {torch.var(subtract, unbiased=False)}")

print("Recorrect the Function for rotate mask list")

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

modified_mask_list_2, rotate_mask_up = rotate_mask_list(mask_list_2, theta_)

print("Generate base and recorrect the mask region and decrease the mask region")

def upsample_mask(mask_list, input_size, up_size):

    if mask_list[0].shape[-1] == up_size:
        return mask_list

    mask_with_inp = []
    for mask_ in mask_list:
        temp_mask = torch.tensor(
            mask_.reshape(1, 1, input_size, input_size), dtype=torch.float
        )
        temp_mask = F.interpolate(temp_mask, size=(up_size, up_size), mode="bicubic")
        temp_mask[temp_mask < 0.5] = 0
        temp_mask[temp_mask >= 0.5] = 1
        temp_mask = torch.tensor(temp_mask.squeeze(), dtype=torch.bool)
        mask_with_inp.append(temp_mask)

    return mask_with_inp

def center_mask_list_function(image, mask_list, coef, radius=7):

    center_mask_list = []
    mean_ = np.zeros([image.shape[-2], image.shape[-1]])

    input_size = mask_list[0].shape[-1]
    up_size = image.shape[-1]

    if input_size != up_size:

        mask_list = upsample_mask(mask_list, input_size, up_size)

    for j, mask in enumerate(mask_list):

        mask_ = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])

        new_image = image * mask_

        center_x, center_y = center_of_mass(new_image.squeeze(), mask_.squeeze(), coef)

        center_x = int(np.round(np.array(center_x)))
        center_y = int(np.round(np.array(center_y)))
        print(center_x, center_y)

        small_mask = mask_function(
            mean_, radius=radius, center_coordinates=(center_y, center_x)
        )

        small_mask = torch.tensor(small_mask, dtype=torch.bool)

        center_mask_list.append(small_mask)

    if input_size != up_size:

        center_mask_list = upsample_mask(center_mask_list, up_size, input_size)

    rotate_mask_up = torch.clone(center_mask_list[0])

    for i in range(1, len(center_mask_list)):
        rotate_mask_up += center_mask_list[i]

    return center_mask_list, rotate_mask_up

print("Parameters for second training process")

num_mask_2 = 6
fixed_mask_2 = modified_mask_list_2
interpolate_2 = True

print("Model for second process training")

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
    
    # Remove 'module.' prefix from state dict keys
    new_state_dict = {}
    for k, v in pre_weight["net"].items():
        name = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[name] = v
    
    join.load_state_dict(new_state_dict)

    return encoder, decoder, join, optimizer

print("Loss Function used for small batch size")

def loss_function_2nd(
    join,
    train_iterator,
    optimizer,
    device,
    coef1=0,
    coef2=0,
    coef3=0,
    ln_parm=1,
    mask_=None,
    up_inp=False,
):
    weight_decay = coef1
    scale_coef = coef2
    shear_coef = coef3

    # set the train mode
    join.eval()

    # loss of the epoch
    train_loss = 0
    L2_loss = 0
    Scale_Loss = 0
    Shear_Loss = 0
    
    with torch.no_grad():
        # Take only one batch from train_iterator
        x, y = next(iter(train_iterator))

        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)

        if up_inp:
            predicted_x, predicted_base, predicted_input, kout, theta_1, theta_2, adj_mask, new_list, x_inp = join(x, y)
            mask_ = upsample_mask(mask_, x.shape[-1], x_inp.shape[-1])
            logging.info(f"Upsampled mask shape: {x_inp.shape[-1]}")
        else:
            predicted_x, predicted_base, predicted_input, kout, theta_1, theta_2, adj_mask, new_list = join(x, y)

        # Log model outputs
        logging.info(f"Predicted base min/max: {predicted_base.min():.6f}/{predicted_base.max():.6f}")
        logging.info(f"Predicted x min/max: {predicted_x.min():.6f}/{predicted_x.max():.6f}")
        
        # L2 regularization loss
        l2_loss = weight_decay * torch.norm(predicted_base.squeeze(), p=ln_parm) / x.shape[0]
        logging.info(f"L2 loss (before scaling): {l2_loss:.6f}")
        
        # Scale loss computation
        scale_diff_1 = abs(theta_1[:, 0, 0] - 1)
        scale_diff_2 = abs(theta_1[:, 1, 1] - 1)
        scale_loss = scale_coef * (
            torch.mean(F.relu(scale_diff_1 - 0.04)) +
            torch.mean(F.relu(scale_diff_2 - 0.04))
        )
        logging.info(f"Scale differences: {scale_diff_1.mean():.6f}, {scale_diff_2.mean():.6f}")
        logging.info(f"Scale loss (before scaling): {scale_loss:.6f}")
        
        # Shear loss computation
        shear_diff = abs(theta_1[:, 0, 1])
        shear_loss = shear_coef * torch.mean(F.relu(shear_diff - 0.04))
        logging.info(f"Shear difference: {shear_diff.mean():.6f}")
        logging.info(f"Shear loss (before scaling): {shear_loss:.6f}")

        # Initial loss computation
        loss = (l2_loss + scale_loss + shear_loss) * len(mask_)

        # Compute losses for each mask
        for i, mask in enumerate(mask_):
            # MSE loss between predicted base and predicted x in mask region
            base_x_loss = F.mse_loss(
                predicted_base.squeeze()[:, mask],
                predicted_x.squeeze()[:, mask],
                reduction="mean",
            )
            logging.info(f"Mask {i} base-x MSE loss: {base_x_loss:.6f}")
            loss += base_x_loss

            # MSE loss between predicted input and original input in mask region
            sub_loss = 0
            for k in range(x.shape[0]):
                if up_inp:
                    batch_loss = F.mse_loss(
                        predicted_input[k].squeeze()[new_list[i][k]],
                        x_inp[k].squeeze()[new_list[i][k]],
                        reduction="mean",
                    )
                else:
                    batch_loss = F.mse_loss(
                        predicted_input[k].squeeze()[new_list[i][k]],
                        x[k].squeeze()[new_list[i][k]],
                        reduction="mean",
                    )
                sub_loss += batch_loss
                logging.info(f"Mask {i} Batch {k} input MSE loss: {batch_loss:.6f}")

            sub_loss = sub_loss / x.shape[0]
            logging.info(f"Mask {i} average input MSE loss: {sub_loss:.6f}")
            loss += sub_loss

        # Scale down the total loss
        loss = loss / (len(mask_) * 15)
        logging.info(f"Loss after first scaling: {loss:.6f}")

        # Handle large losses
        # if loss > 1.5:

        #     logging.info(f"Loss > 1.5: {loss:.6f} switching to L1 loss...")
        #     loss = (l2_loss + scale_loss + shear_loss) * len(mask_)
        #     logging.info(f"Loss after adding L2, Scale, and Shear loss and len(mask_) * 15: {loss:.6f}")
            
        #     for i, mask in enumerate(mask_):
        #         # L1 loss between predicted base and predicted x
        #         base_x_loss = F.l1_loss(
        #             predicted_base.squeeze()[:, mask],
        #             predicted_x.squeeze()[:, mask],
        #             reduction="mean",
        #         )
        #         logging.info(f"Mask {i} base-x L1 loss: {base_x_loss:.6f}")
        #         loss += base_x_loss

        #         sub_loss = 0
        #         for k in range(x.shape[0]):
        #             if up_inp:
        #                 batch_loss = F.l1_loss(
        #                     predicted_input[k].squeeze()[new_list[i][k]],
        #                     x_inp[k].squeeze()[new_list[i][k]],
        #                     reduction="mean",
        #                 )
        #             else:
        #                 batch_loss = F.l1_loss(
        #                     predicted_input[k].squeeze()[new_list[i][k]],
        #                     x[k].squeeze()[new_list[i][k]],
        #                     reduction="mean",
        #                 )
        #             sub_loss += batch_loss
        #             logging.info(f"Mask {i} Batch {k} input L1 loss: {batch_loss:.6f}")

        #         sub_loss = sub_loss / x.shape[0]
        #         logging.info(f"Mask {i} average input L1 loss: {sub_loss:.6f}")
        #         loss += sub_loss

        #     logging.info(f"Loss before L1 divided by len(mask_) * 15: {loss:.6f}")
        #     loss = loss / (len(mask_) * 15)
        #     logging.info(f"Loss before L1 adjustment: {loss:.6f}")
        #     loss = loss - 1
        #     logging.info(f"Loss after L1 adjustment: {loss:.6f}")

        # if loss > (2 * len(mask_)):
        #     logging.info(f"Loss > {2 * len(mask_)}, capping loss...")
        #     loss = 2 * len(mask_) + l2_loss + scale_loss + shear_loss
        #     logging.info(f"Final capped loss: {loss:.6f}")

        # Update component losses
        train_loss += loss.item()
        L2_loss += l2_loss
        Scale_Loss += scale_loss
        Shear_Loss += shear_loss

        logging.info(f"Final losses - Train: {train_loss:.6f}, L2: {L2_loss:.6f}, Scale: {Scale_Loss:.6f}, Shear: {Shear_Loss:.6f}")

    return train_loss, L2_loss, Scale_Loss, Shear_Loss

print("The function for the scale and shear evaluation")

def Show_Process(model, test_iterator, mask_list, name_of_file, device, up_inp):

    model.eval()

    rotation_ = np.zeros([65536, 2])
    scale_shear_ = np.zeros([65536, 4])
    number_loss = len(mask_list)
    loss_map = np.zeros([65536, number_loss])

    for i, x in enumerate(tqdm(test_iterator, leave=True, total=len(test_iterator))):
        with torch.no_grad():
            value, rot = x
            rot = Variable(rot.cuda()).float()
            test_value = Variable(value.cuda())
            test_value = test_value.float()

            if up_inp:
                (
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    kout,
                    theta_1,
                    theta_2,
                    adj_mask,
                    new_list,
                    x_inp,
                ) = model(
                    test_value.to(device, dtype=torch.float),
                    rot.to(device, dtype=torch.float),
                )

                mask_list = upsample_mask(
                    mask_list, test_value.shape[-1], x_inp.shape[-1]
                )

            else:
                (
                    predicted_x,
                    predicted_base,
                    predicted_input,
                    kout,
                    theta_1,
                    theta_2,
                    adj_mask,
                    new_list,
                ) = model(
                    test_value.to(device, dtype=torch.float),
                    rot.to(device, dtype=torch.float),
                )

            batch_size = test_value.shape[0]

            rotation_[i * batch_size : (i + 1) * batch_size] = (
                theta_2[:, :, 0].cpu().detach().numpy()
            )

            scale_shear_[i * batch_size : (i + 1) * batch_size] = (
                theta_1[:, :, 0:2].cpu().detach().numpy().reshape(-1, 4)
            )

            for j, mask in enumerate(mask_list):

                temp_loss = 0

                temp_loss += torch.mean(
                    (predicted_base.squeeze()[:, mask] - predicted_x.squeeze()[:, mask])
                    ** 2,
                    1,
                )

                for k in range(test_value.shape[0]):
                    if up_inp:
                        single_loss = F.mse_loss(
                            predicted_input[k].squeeze()[new_list[j][k]],
                            x_inp[k].squeeze()[new_list[j][k]],
                            reduction="mean",
                        )
                    else:
                        single_loss = F.mse_loss(
                            predicted_input[k].squeeze()[new_list[j][k]],
                            test_value[k].squeeze()[new_list[j][k]],
                            reduction="mean",
                        )
                    temp_loss[k] += single_loss

                loss_map[i * batch_size : (i + 1) * batch_size, j] = (
                    temp_loss.cpu().detach().numpy().reshape(batch_size)
                )

    loss_map = np.mean(loss_map, axis=1)

    predicted_base = predicted_base[0].cpu().detach().numpy()

    h5f = h5py.File(name_of_file + ".h5", "w")
    h5f.create_dataset("rotation", data=rotation_)
    h5f.create_dataset("scale_shear", data=scale_shear_)
    h5f.create_dataset("loss", data=loss_map)
    h5f.create_dataset("base", data=predicted_base)

    # stack the list to torch tensor for saving in the h5 format
    gou_list = torch.cat(mask_list)
    gou_list = gou_list.reshape(
        len(mask_list), mask_list[0].shape[-2], mask_list[0].shape[-1]
    )

    h5f.create_dataset("mask_list", data=gou_list)
    h5f.close()

    im_size = (256, 256)
    M_init = basis2probe(rotation_, scale_shear_).reshape(im_size[0], im_size[1], 2, 2)

    M_ref = np.median(M_init[30:60, 10:40], axis=(0, 1))

    # u_ref, p_ref = sp.linalg.polar(M_ref, side='right')

    exx_Shuyu = np.zeros((im_size[0], im_size[1]))
    eyy_Shuyu = np.zeros((im_size[0], im_size[1]))
    exy_Shuyu = np.zeros((im_size[0], im_size[1]))

    for rx in range(im_size[0]):
        for ry in range(im_size[1]):

            T = M_init[rx, ry] @ np.linalg.inv(M_ref)
            u, p = sp.linalg.polar(T, side="left")

            transformation = np.array(
                [
                    [p[0, 0] - 1, p[0, 1]],
                    [p[0, 1], p[1, 1] - 1],
                ]
            )

            #       transformation = u @ transformation @ u.T

            exx_Shuyu[rx, ry] = transformation[1, 1]
            eyy_Shuyu[rx, ry] = transformation[0, 0]
            exy_Shuyu[rx, ry] = transformation[0, 1]

    return exx_Shuyu, eyy_Shuyu, exy_Shuyu

def Generate_scale_shear_loss(
    exx_Shuyu, eyy_Shuyu, exy_Shuyu, label_xx, label_yy, label_xy
):
    dif_shuyu_xx = exx_Shuyu - label_xx
    dif_shuyu_yy = eyy_Shuyu - label_yy
    dif_shuyu_xy = exy_Shuyu - label_xy
    mae_shuyu_xx = np.mean(abs(dif_shuyu_xx))
    mae_shuyu_yy = np.mean(abs(dif_shuyu_yy))
    mae_shuyu_xy = np.mean(abs(dif_shuyu_xy))

    combine_loss = mae_shuyu_xx + mae_shuyu_yy + mae_shuyu_xy

    return combine_loss

print("Second training process: set training parameters")

data_dir = os.path.abspath(
    "Simulated_4dstem/Extremely_Noisy_4DSTEM_Strain_Mapping_Using_CC_ST_AE_Simulated/polycrystal_output4D.mat"
)
# folder_path = os.path.abspath("./07_13_25Percent_Upsample_SCALE_SHEAR_On_MASK_Test_large_MASK_update_MASK_once_15DEGREE")
# pretrain_weight = os.path.abspath("./04_20_RAYTUNE_lr:0.000065_scale_cof:80.350_shear_cof:16.100_MAE:0.0063_seed:42_epoch:0004_trainloss:0.002384_l1:0.00014_scal:0.00000_shr:0.00000.pkl")

whole_data_with_rotation = load_data_4_process2(data_dir, new_rotation, w_bg=0.25)

def clone_parameters(parameters):
    """Deep copy a list of parameters, ensuring tensors are properly cloned to new memory"""
    return [p.clone().detach() for p in parameters]

def get_model_parameters(model):
    """Get all parameters of the model as a list of tensors"""
    return [p.data.clone().detach() for p in model.parameters()]

def set_parameters(model, parameters):
    """Set model parameters from a list of tensors"""
    with torch.no_grad():  # Ensure we don't track operations
        for model_param, param in zip(model.parameters(), parameters):
            model_param.data.copy_(param)

def get_model_norm(parameters, order=2):
    """
    Compute the norm of the entire model parameters.
    Args:
        parameters: list of parameter tensors
        order: order of the norm (default: 2 for L2 norm)
    Returns:
        torch.Tensor: the model-wise norm
    """
    return torch.sqrt(sum(p.norm(order).pow(2) for p in parameters))

def normalize_layer_wise(direction, parameters):
    """Normalize direction layer by layer"""
    normalized = []
    for d, p in zip(direction, parameters):
        layer_norm = d.norm()
        normalized.append(d / (layer_norm + 1e-10))
    return normalized

def normalize_filter_wise(direction, parameters):
    """Normalize direction filter by filter"""
    normalized = []
    for d, p in zip(direction, parameters):
        if len(d.size()) <= 1:  # For bias vectors
            d_norm = d.norm()
            normalized.append(d / (d_norm + 1e-10))
        else:  # For weight tensors
            normalized_layer = []
            for f in range(d.size(0)):  # Iterate over filters
                f_norm = d[f].norm()
                d[f] = d[f] / (f_norm + 1e-10)
            normalized.append(d)
    return normalized

def normalize_model_wise(direction, parameters):
    """Normalize direction using the entire model norm"""
    model_norm = torch.sqrt(sum((d * d).sum() for d in direction))
    return [d / (model_norm + 1e-10) for d in direction]

def normalize_direction(direction, parameters, normalization='filter'):
    """Normalize direction according to specified method"""
    if normalization == 'model':
        return normalize_model_wise(direction, parameters)
    elif normalization == 'layer':
        return normalize_layer_wise(direction, parameters)
    elif normalization == 'filter':
        return normalize_filter_wise(direction, parameters)
    else:
        raise ValueError("normalization must be 'model', 'layer', or 'filter'")

def rand_uniform_like(parameters):
    """Generate random uniform direction with same shape as parameters"""
    return [torch.rand_like(p) for p in parameters]

def make_orthogonal(direction_one):
    """Create a new direction that is orthogonal to direction_one"""
    # Generate random direction first
    direction_two = rand_uniform_like(direction_one)
    
    # Compute dot product and norm
    dot_product = sum((d1 * d2).sum() for d1, d2 in zip(direction_one, direction_two))
    model_norm = get_model_norm(direction_one)
    
    # Make orthogonal using projection
    orthogonal = []
    for d1, d2 in zip(direction_one, direction_two):
        orthogonal.append(d2 - (dot_product/(model_norm**2)) * d1)
    return orthogonal

def scale_direction(direction, scale):
    """Scale a direction by a scalar"""
    # Convert scale to tensor if it's not already
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=direction[0].dtype, device=direction[0].device)
    
    # Instead of creating new list, modify in-place
    for d in direction:
        d.mul_(scale)
    return direction

def mul_(parameters, scale):
    """In-place multiplication by scalar"""
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=parameters[0].dtype, device=parameters[0].device)
    for p in parameters:
        p.mul_(scale)
    return parameters

def truediv_(parameters, scale):
    """In-place division by scalar"""
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=parameters[0].dtype, device=parameters[0].device)
    for p in parameters:
        p.div_(scale)
    return parameters

def add_direction(parameters, direction):
    """Add direction to parameters in-place"""
    for p, d in zip(parameters, direction):
        p.add_(d)

def sub_direction(parameters, direction):
    """Subtract direction from parameters in-place"""
    for p, d in zip(parameters, direction):
        p.sub_(d)

# Global configuration
STEPS = 41  # number of steps in each direction
DISTANCE = 0.2  # maximum distance from start point

def compute_loss_landscape(join, train_iterator, optimizer, device, coef_1, coef_2, coef_3, regul_type, mask_, interpolate_2, rotation_file="25Percent_rotation_071323"):
    """
    Compute the loss landscape along a planar subspace of the parameter space.
    Implementation matches random_plane from model_parameters.py
    """
    try:
        # Get starting parameters and save original weights with proper deep copy
        with torch.no_grad():
            start_point = get_model_parameters(join)
            original_weights = clone_parameters(start_point)
        
        # Generate random orthogonal directions
        dir_one = rand_uniform_like(start_point)
        dir_two = make_orthogonal(dir_one)
        
        # Normalize directions using filter normalization
        dir_one = normalize_direction(dir_one, start_point, normalization='filter')
        dir_two = normalize_direction(dir_two, start_point, normalization='filter')
        
        # Grid setup - use global variables
        steps = STEPS
        distance = DISTANCE
        
        # Scale directions to match steps and total distance
        model_norm = get_model_norm(start_point)
        
        # Scale to match steps and total distance (exactly as in random_plane)
        dir_one_norm = get_model_norm(dir_one)
        dir_two_norm = get_model_norm(dir_two)
        mul_(dir_one, ((model_norm * distance) / steps) / dir_one_norm)
        mul_(dir_two, ((model_norm * distance) / steps) / dir_two_norm)
        
        # Move start point to corner and adjust step size (exactly as in random_plane)
        mul_(dir_one, steps / 2)
        mul_(dir_two, steps / 2)
        current_point = clone_parameters(original_weights)
        sub_direction(current_point, dir_one)
        sub_direction(current_point, dir_two)
        truediv_(dir_one, steps / 2)
        truediv_(dir_two, steps / 2)
        
        # Initialize loss surface
        loss_surface = torch.zeros((steps, steps), device=device)
        
        # Verify original_weights and current_point are different
        weight_diff = sum(torch.sum((c - o).abs()) for c, o in zip(current_point, original_weights))
        logging.info(f"Initial weight difference from original: {weight_diff.item()}")
        print(f"Initial weight difference from original: {weight_diff.item()}")
        
        if weight_diff.item() < 1e-6:
            raise ValueError("Starting point is too close to original weights - directions may not be properly scaled")
        # # Verify center point will be original weights
        # center_point = clone_parameters(current_point)
        # mul_(dir_one, steps / 2)
        # add_direction(center_point, dir_one)
        # mul_(dir_two, steps / 2) 
        # add_direction(center_point, dir_two)

        # truediv_(dir_one, steps / 2)
        # truediv_(dir_two, steps / 2)
            
        # # Check if center point matches original weights
        # center_diff = sum(torch.sum((c - o).abs()) for c, o in zip(center_point, original_weights))
        # logging.info(f"Center point difference from original: {center_diff.item()}")
        # print(f"!!!!!!Center point difference from original: {center_diff.item()}")
        
        # Compute loss landscape
        data_matrix = []  # Store data in a list first, like in random_plane
        with torch.no_grad():
            for i in tqdm(range(steps), desc="Computing loss landscape"):
                data_column = []  # Store each column of data
                
                for j in range(steps):
                    # Set model weights to current grid point
                    set_parameters(join, current_point)
                    
                    # Compute loss at current point
                    train_metrics = loss_function_2nd(
                        join, train_iterator, optimizer, device,
                        coef_1, coef_2, coef_3, regul_type,
                        mask_, interpolate_2
                    )
                    
                    # For every other column, reverse the order in which the column is generated
                    if i % 2 == 0:
                        add_direction(current_point, dir_two)
                        data_column.append(train_metrics[0])
                    else:
                        sub_direction(current_point, dir_two)
                        data_column.insert(0, train_metrics[0])
                
                data_matrix.append(data_column)
                add_direction(current_point, dir_one)
                
                # Clear GPU memory periodically
                if i % 2 == 0:
                    torch.cuda.empty_cache()
                
                logging.info(f"Completed row {i+1}/{steps}")
        
        # Convert to numpy array at the end
        loss_surface = np.array(data_matrix)
        
        # Create directory if it doesn't exist
        save_dir = 'loss_landscapes'
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract rotation percentage from filename
        percentage = rotation_file.split('_')[0]
        
        # Create filename with parameters
        filename = f"{percentage}_steps{steps}_dist{distance:.1f}_loss_landscape.npz"
        save_path = os.path.join(save_dir, filename)
        
        # Save results
        np.savez(save_path, 
                 loss_surface=loss_surface,
                 x_coordinates=np.linspace(-distance, distance, steps),
                 y_coordinates=np.linspace(-distance, distance, steps))
        
        logging.info(f"Successfully saved loss landscape data to {save_path}")
        
    except Exception as e:
        logging.error(f"Error during loss landscape computation: {e}")
        raise
        
    finally:
        # Restore original weights
        set_parameters(join, original_weights)
        
    return loss_surface

def Test_Process(
    data_set,
    epochs=1,
    activity_regular="l1",
    mask_=modified_mask_list_2,
    check_mask=modified_mask_list_2,
    Up_inp=interpolate_2,
    epoch_=None,
    file_path=None,
    folder_path="",
    best_train_loss=None,
):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Set random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logging.info("Set random seed to 42")

    # Training parameters
    learning_rate = 3e-4
    coef_1 = 5e-7
    coef_2 = 70
    coef_3 = 10
    batch_size = 64  # Added explicit batch_size

    # Round parameters for consistency
    learning_rate = round(learning_rate * 1e6) / 1e6
    coef_1 = round(coef_1 * 1e9) / 1e9
    coef_2 = round(coef_2 * 1e2) / 1e2
    coef_3 = round(coef_3 * 1e2) / 1e2

    logging.info(f"Learning parameters - LR: {learning_rate}, Coef1: {coef_1}, Coef2: {coef_2}, Coef3: {coef_3}")

    # Model setup
    encoder, decoder, join, optimizer = make_model_2(
        device, learning_rate=learning_rate, fixed_mask=mask_
    )

    # Load checkpoint
    try:
        checkpoint = "2nd_train_weight_25Per.pkl"
        pre_weight = torch.load(checkpoint, map_location=device)
        
        # Remove 'module.' prefix from state dict keys
        new_state_dict = {}
        for k, v in pre_weight["net"].items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        join.load_state_dict(new_state_dict)
        logging.info(f"Successfully loaded checkpoint: {checkpoint}")
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        raise

    # Set regularization type
    regul_type = 1 if activity_regular == "l1" else (2 if activity_regular == "l2" else 0)

    # DataLoader setup
    train_iterator = DataLoader(
        data_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initial evaluation
    # train_metrics = loss_function_2nd(
    #     join,
    #     train_iterator,
    #     optimizer,
    #     device,
    #     coef_1,
    #     coef_2,
    #     coef_3,
    #     regul_type,
    #     mask_,
    #     interpolate_2,
    # )

    # train_loss, L2_loss, Scale_Loss, Shear_Loss = train_metrics
    # input_length = len(train_iterator)

    # # Normalize losses
    # train_loss /= input_length
    # L2_loss /= input_length
    # Scale_Loss /= input_length
    # Shear_Loss /= input_length

    logging.info("Starting loss landscape computation...")
    
    try:
        # Compute loss landscape
        loss_surface = compute_loss_landscape(
            join, train_iterator, optimizer, device,
            coef_1, coef_2, coef_3, regul_type, mask_, interpolate_2
        )

    except Exception as e:
        logging.error(f"Error during loss landscape computation: {e}")
        raise

    # return train_loss, L2_loss, Scale_Loss, Shear_Loss

if __name__ == "__main__":
    logging.info("Starting test_ll.py")
    Test_Process(whole_data_with_rotation)
    logging.info("Completed test_ll.py")
