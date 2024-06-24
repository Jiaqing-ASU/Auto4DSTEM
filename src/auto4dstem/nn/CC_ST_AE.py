import numpy as np
from ..Viz.util import mask_function, center_of_mass
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import m3_learning.nn.STEM_AE_DCT.spot_fitting as spot
import warnings
warnings.filterwarnings("ignore")


def crop_small_square(center_coordinates, radius=50):
    """function to crop small square image for revise operation

    Args:
        center_coordinates (torch.tensor): coordinates of diffraction spots after COM
        radius (int, optional): the radius of small square for revise operation . Defaults to 50.

    Returns:
        tuple: the coordinates of corners of  small square image
    """

    center_coordinates = torch.round(center_coordinates)
    # calculate the x axis and y axis coordinate of diffraction spots (format integer)
    x_coordinate = (
        int(center_coordinates[0] - radius),
        int(center_coordinates[0] + radius),
    )

    y_coordinate = (
        int(center_coordinates[1] - radius),
        int(center_coordinates[1] + radius),
    )

    return x_coordinate, y_coordinate


def revise_size_on_affine_gpu(
    image,
    mask_list,
    batch_size,
    theta,
    device,
    adj_para=None,
    radius=12,
    coef=1.5,
    pare_reverse=False,
    affine_mode="bicubic",
):
    """function for revise size of diffraction spots

    Args:
        image (torch.tensor): image with diffraction spots
        mask_list (list): list of binary mask images
        batch_size (int): number of images in each minibatch
        theta (torch.tensor): affine transformation matrix (scale and shear)
        device (torch.device): set the device to run the model
        adj_para (float, optional): Parameter to change the intensity of each diffraction spot, Defaults to None.
        radius (int, optional): to determine the size of square image for revise operation
        coef (int, optional): the parameter to control the value of threshold for COM operation. Defaults to 1.5.
        pare_reverse (bool, optional): switch multiplying or dividing adj_para . Defaults to False.
        affine_mode(string, optional): set the affine mode to function F.affine_grid(). Defaults 'bicubic'.

    Returns:
        torch.tenosr: image after revise operation
    """
    
    # set size of small square image for revise affine 
    np_img = np.zeros([radius * 2, radius * 2])
    
    # crop small circle only include diffraction spots
    dot_size = int(4 * image.shape[-1] / 200)
    small_square_mask = mask_function(
        np_img, radius=dot_size, center_coordinates=(radius, radius)
    )
    small_square_mask = torch.tensor(small_square_mask, dtype=torch.bool).to(device)

    img = torch.clone(image).to(device)

    # create identity matrix to make affine matrix into size [batch,3,3] for computing inverse matrix
    identity = (
        torch.tensor([0, 0, 1], dtype=torch.float)
        .reshape(1, 1, 3)
        .repeat(batch_size, 1, 1)
        .to(device)
    )
    # create 3x3 affine matrix 
    new_theta = torch.cat((theta, identity), axis=1).to(device)
    # computing inverse matrix
    inver_theta = torch.linalg.inv(new_theta)[:, 0:2].to(device)

    # replicate each mask into the same size of input
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
        # only keep values inside mask region
        new_image = image * mask_.to(device)

        for i in range(batch_size):
            
        # extract center coordinates of each diffraction spots
            center_x, center_y = center_of_mass(
                new_image[i].squeeze(), mask_[i].squeeze(), coef
            )
            center = torch.tensor([center_x, center_y]).to(device)
            
        # extract coordinates of corners of  small square image which has diffraction spots
            x_coordinate, y_coordinate = crop_small_square(
                center_coordinates=center.clone(), radius=radius
            )

        # crop the small image according to coordinates
            small_image = (
                img[i]
                .squeeze()[
                    x_coordinate[0] : x_coordinate[1], y_coordinate[0] : y_coordinate[1]
                ]
                .unsqueeze(0)
                .unsqueeze(1)
                .clone()
                .to(device)
            )
        # apply inverse affine transform on small images
            re_grid = F.affine_grid(
                inver_theta[i].unsqueeze(0).to(device), small_image.size()
            ).to(device)
       
            if adj_para == None:
                re_aff_small_image = F.grid_sample(
                    small_image, re_grid, mode=affine_mode
                )
                img[
                    i,
                    :,
                    x_coordinate[0] : x_coordinate[1],
                    y_coordinate[0] : y_coordinate[1],
                ] = re_aff_small_image.squeeze()

            else:
                small_image_copy = torch.clone(small_image.squeeze()).to(device)

                if pare_reverse:
                    small_image_copy[small_square_mask] /= adj_para[i]
                else:
                    small_image_copy[small_square_mask] *= adj_para[i]

                small_image_copy = small_image_copy.unsqueeze(0).unsqueeze(1)

                re_aff_small_image = F.grid_sample(
                    small_image_copy, re_grid, mode=affine_mode
                )
                img[
                    i,
                    :,
                    x_coordinate[0] : x_coordinate[1],
                    y_coordinate[0] : y_coordinate[1],
                ] = re_aff_small_image.squeeze()

    return img


class conv_block(nn.Module):
    def __init__(self, t_size, n_step):
        """Initializes the convolutional block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """
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
        """Forward pass of the convolutional block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
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
        """Initializes the identity block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """
        super(identity_block, self).__init__()
        self.cov1d_1 = nn.Conv2d(
            t_size, t_size, 3, stride=1, padding=1, padding_mode="zeros"
        )
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


class Affine_Transform(nn.Module):
    def __init__(
        self,
        device,
        scale=True,
        shear=True,
        rotation=True,
        rotate_clockwise=True,
        translation=False,
        Symmetric=True,
        mask_intensity=True,
        scale_limit=0.05,
        shear_limit=0.1,
        rotation_limit=0.1,
        trans_limit=0.15,
        adj_mask_para=0,
    ):
        """_summary_

        Args:
            device (torch.device): set the device to run the model
            scale (bool): set to True if the model include scale affine transform
            shear (bool): set to True if the model include shear affine transform
            rotation (bool): set to True if the model include rotation affine transform
            rotate_clockwise (bool): set to True if the image should be rotated along one direction
            translation (bool): set to True if the model include translation affine transform
            Symmetric (bool): set to True if the shear affine transform is symmetric
            mask_intensity (bool):set to True if the intensity of the mask region is learnable
            scale_limit (float, optional): limit the range of scale parameter. Defaults to 0.05.
            shear_limit (float, optional): limit the range of shear parameter. Defaults to 0.1.
            rotation_limit (float, optional): limit the range of rotation parameter. Defaults to 0.1.
            trans_limit (float, optional): limit the range of translation parameter. Defaults to 0.15.
            adj_mask_para (int, optional): limit the range of learnable intensity in mask region. Defaults to 0.

        """

        super(Affine_Transform, self).__init__()
        self.scale = scale
        self.shear = shear
        self.rotation = rotation
        self.rotate_clockwise = rotate_clockwise
        self.translation = translation
        self.Symmetric = Symmetric
        self.scale_limit = scale_limit
        self.shear_limit = shear_limit
        self.rotation_limit = rotation_limit
        self.trans_limit = trans_limit
        self.adj_mask_para = adj_mask_para
        self.mask_intensity = mask_intensity
        self.device = device
        self.count = 0

    def forward(self, out, rotate_value=None):
        """Forward pass of the affine transform

        Args:
            out (Tensor): Input tensor
            rotate_value (tensor, optional): pretrained rotation if have. Defaults to None.

        Returns:
            Tensor: affine matrix and adjust parameters
        """
        # determine the type of affine transform and create matrix according to it
        if self.scale:
            scale_1 = self.scale_limit * nn.Tanh()(out[:, self.count]) + 1
            scale_2 = self.scale_limit * nn.Tanh()(out[:, self.count + 1]) + 1
            self.count += 2
        else:
            scale_1 = torch.ones([out.shape[0]]).to(self.device)
            scale_2 = torch.ones([out.shape[0]]).to(self.device)


        if self.rotation:
            if rotate_value != None:
                # use large mask no need to limit to too small range

                rotate = rotate_value.reshape(
                    out[:, self.count].shape
                ) + self.rotation_limit * nn.Tanh()(out[:, self.count])

            else:
                if self.rotate_clockwise:
                    rotate = nn.ReLU()(out[:, self.count])
                else:
                    rotate = self.rotation_limit * nn.Tanh()(out[:, self.count])

            self.count += 1

        else:
            rotate = torch.zeros([out.shape[0]]).to(self.device)

        #        print(self.count)

        if self.shear:
            if self.Symmetric:
                shear_1 = self.shear_limit * nn.Tanh()(out[:, self.count])
                shear_2 = shear_1

                self.count += 1
            else:
                shear_1 = self.shear_limit * nn.Tanh()(out[:, self.count])
                shear_2 = self.shear_limit * nn.Tanh()(out[:, self.count + 1])

                self.count += 2
        else:
            shear_1 = torch.zeros([out.shape[0]]).to(self.device)
            shear_2 = torch.zeros([out.shape[0]]).to(self.device)

        # usually the 4d-stem has symmetric shear value, we make xy=yx, that's the reason we don't need shear2

        if self.translation:
            trans_1 = self.trans_limit * nn.Tanh()(out[:, self.count])
            trans_2 = self.trans_limit * nn.Tanh()(out[:, self.count + 1])

            self.count += 2
        else:
            trans_1 = torch.zeros([out.shape[0]]).to(self.device)
            trans_2 = torch.zeros([out.shape[0]]).to(self.device)

        # add one additional learnable parameter to adjust intensity of value in mask region
        if self.mask_intensity:
            mask_parameter = (
                self.adj_mask_para * nn.Tanh()(out[:, self.count : self.count + 1]) + 1
            )

        else:
            # this project doesn't need mask parameter to adjust value intensity in mask region, so we make it 1 here.
            mask_parameter = torch.ones([out.shape[0], 1])

        # reset count to 0 for next minibatch using
        self.count = 0

        a_1 = torch.cos(rotate)
        a_2 = torch.sin(rotate)
        a_4 = torch.ones([out.shape[0]]).to(self.device)
        a_5 = torch.zeros([out.shape[0]]).to(self.device)

        # combine shear and strain together
        c1 = torch.stack((scale_1, shear_1), dim=1).squeeze()
        c2 = torch.stack((shear_2, scale_2), dim=1).squeeze()
        c3 = torch.stack((a_5, a_5), dim=1).squeeze()
        scaler_shear = torch.stack((c1, c2, c3), dim=2)

        # Add the rotation after the shear and strain
        b1 = torch.stack((a_1, a_2), dim=1).squeeze()
        b2 = torch.stack((-a_2, a_1), dim=1).squeeze()
        b3 = torch.stack((a_5, a_5), dim=1).squeeze()
        rotation = torch.stack((b1, b2, b3), dim=2)

        # add translation after rotation
        d1 = torch.stack((a_4, a_5), dim=1).squeeze()
        d2 = torch.stack((a_5, a_4), dim=1).squeeze()
        d3 = torch.stack((trans_1, trans_2), dim=1).squeeze()
        translation = torch.stack((d1, d2, d3), dim=2)

        return scaler_shear, rotation, translation, mask_parameter


# narrow the range of the adjust parameter for the mask region, since it is not the noise free dataset,
# this will increase the background noise's influence to the MSE loss
#
class Encoder(nn.Module):
    def __init__(
        self,
        original_step_size,
        pool_list,
        conv_size,
        device,
        scale=True,
        shear=True,
        rotation=True,
        rotate_clockwise=True,
        translation=False,
        Symmetric=True,
        mask_intensity=True,
        num_base=2,
        fixed_mask=None,
        num_mask=1,
        interpolate=False,
        up_size=800,
        scale_limit=0.05,
        shear_limit=0.1,
        rotation_limit=0.1,
        trans_limit=0.15,
        adj_mask_para=0,
        radius=60,
        coef=1.5,
        reduced_size=20,
        interpolate_mode="bicubic",
        affine_mode="bicubic",
        emb_function='ktop',
        coords = None,
        r = None
    ):
        """_summary_

        Args:
            original_step_size (list of int): the x and y size of input image
            pool_list (list of int): the list of parameter for each 2D MaxPool layer
            embedding_size (int): the value for number of channels
            conv_size (int): the value of filters number goes to each block
            device (torch.device): set the device to run the model
            scale (bool): set to True if the model include scale affine transform
            shear (bool): set to True if the model include shear affine transform
            rotation (bool): set to True if the model include rotation affine transform
            rotate_clockwise (bool): set to True if the image should be rotated along one direction
            translation (bool): set to True if the model include translation affine transform
            Symmetric (bool): set to True if the shear affine transform is symmetric
            mask_intensity (bool):set to True if the intensity of the mask region is learnable
            num_base(int, optional): the value for number of base. Defaults to 2.
            fixed_mask (list of tensor, optional): The list of tensor with binary type. Defaults to None.
            num_mask (int, optional): the value for number of mask. Defaults to len(fixed_mask).
            interpolate (bool): set to determine if need to calculate loss value in interpolated version. Defaults to False.
            up_size (int, optional): the size of image to set for calculating MSE loss. Defaults to 800.
            scale_limit (float): set the range of scale. Defaults to 0.05.
            shear_limit (float): set the range of shear. Defaults to 0.1.
            rotation_limit (float): set the range of shear. Defaults to 0.1.
            trans_limit (float): set the range of translation. Defaults to 0.15.
            adj_mask_para (float): set the range of learnable parameter used to adjust pixel value in mask region. Defaults to 0.
            radius (int): set the radius of small square image for cropping. Defaults to 60.
            coef (float): set the threshold for COM operation. Defaults to 1.5.
            reduced_size (int): set the input length of K-top layer. Defaults 20.
            interpolate_size (string, optional): set the interpolate mode to function F.interpolate(). Defaults 'bicubic'.
            affine_mode (int): set the affine mode to function F.affine_grid(). Defaults 'bicubic'.

        """

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
        
        # update image size to to each convolutional block and identity block
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
        self.before = nn.Linear(input_size, reduced_size)
        self.embedding_size = 0
        if scale:
            self.embedding_size += 2
        if shear:
            if Symmetric:
                self.embedding_size += 1
            else:
                self.embedding_size += 2
        if rotation:
            self.embedding_size += 1
        if translation:
            self.embedding_size += 2
        if self.embedding_size == 0:
            print(" No affine transformation found in the model structure")
        self.mask_size = num_mask

        self.interpolate = interpolate
        self.interpolate_mode = interpolate_mode
        self.affine_mode = affine_mode
        self.up_size = up_size
        
        if fixed_mask is not None:
        # Set the mask_ to upscale mask if the interpolate mode is True
            if self.interpolate:
                mask_with_inp = []
                for mask_ in fixed_mask:
                    temp_mask = torch.tensor(
                        mask_.reshape(1, 1, self.input_size_0, self.input_size_1),
                        dtype=torch.float,
                    )
                    temp_mask = F.interpolate(
                        temp_mask,
                        size=(self.up_size, self.up_size),
                        mode=self.interpolate_mode,
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

        if mask_intensity:
            self.dense = nn.Linear(reduced_size + num_base, self.embedding_size + 1)
        else:
        # Set the all the adj parameter to be the same
            self.dense = nn.Linear(reduced_size + num_base, self.embedding_size)
            
        

        self.radius = radius
        self.coef = coef

        self.affine_matrix = Affine_Transform(
            device,
            scale,
            shear,
            rotation,
            rotate_clockwise,
            translation,
            Symmetric,
            mask_intensity,
            scale_limit,
            shear_limit,
            rotation_limit,
            trans_limit,
            adj_mask_para,
        ).to(device)
           
        if emb_function == 'ktop': 
            # set the number of base (number of cluster)
            self.for_k = nn.Linear(reduced_size, num_base)
            self.norm = nn.LayerNorm(num_base)
            self.softmax = nn.Softmax()
            
            # k is set to be 1 means one input only belongs to 1 cluster
            self.num_k_sparse = 1
            self.emb_func = self.ktop
            
        if emb_function == 'spots': 
            self.tanh=nn.Tanh()
            self.emb_dense = nn.Linear(reduced_size, num_base)
            self.norm = nn.LayerNorm(num_base)
            self.coords = coords
            if self.coords is None: self.coords = [(self.input_size_0 //2,self.input_size_0 //2) for i in range(num_base//2)]
            self.r = r
            if self.r is None: self.r = self.input_size_0//2
            self.emb_func = self.spots 
            
    # @property
    # def device(self):
    #     return self._device

    # @device.setter
    # def device(self, device):
    #     self._device = device
    #     self.to(device)
    #     self._set_submodules_device(self, device)

    # def _set_submodules_device(self, module, device):
    #     for submodule in module.children():
    #         submodule.to(device)
    #         self._set_submodules_device(submodule, device)
    #             # create K-sparse strategy for classification
                
    def ktop(self, x):
        """ktop function

        Args:
            x (torch.tensor): 1D vector

        Returns:
            torch.tensor: binary vector
        """
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

    def spots(self, x,):  
        """generates predicted coordinates of each spot in base

        Returns:
            x: input of shape (N, S*2) where S is the number of spots
        """        
        
        # self = self.to(self.device)
        # print('\t\t\tspots:', f'x:{x.device}', f'w:{self.dense.bias.device}')
                
        coords =  torch.tensor(self.coords).flatten().to(x.device) # numspots,2
        out = self.emb_dense(x)
        out = self.norm(out)
        # out = self.tanh(out)*self.r
        out = (self.tanh(out)*self.r/5 + coords)
        return out
 
    def forward(self, x, rotate_value=None):
        """forward function for nn.Module class

        Args:
            x (torch.tensor): input torch.tensor image
            rotate_value (float, optional): float value represents pretrained rotation angle. Defaults to None.
        """
        # make sure all devices the same for dataparallel
        # self.device = x.device
        # self = self.to(self.device)
        # print('\tenc:', f'x:{x.device}', f'w:{self.dense.bias.device}')
        
        
        # reshape the input into (minibatch, 1 , image_size)
        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        kout = self.before(out)
        # print('\t\tpre:', f'x:{x.device}', f'w:{self.dense.bias.device}')
        
        k_out = self.emb_func(kout)
        # print('\t\tpost:', f'x:{x.device}', f'w:{self.dense.bias.device}')
        
        # concatenate reduced dimensional vector and output vector of k-sparse function
        out = torch.cat((kout, k_out), dim=1).to(x.device)
        out = self.dense(out)

        scaler_shear, rotation, translation, mask_parameter = self.affine_matrix(
            out, rotate_value
        )

        # add affine transformation to input image
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
                x_inp, size=(self.up_size, self.up_size), mode=self.interpolate_mode
            )

            grid_1 = F.affine_grid(scaler_shear.to(self.device), x_inp.size()).to(
                self.device
            )
            out_sc_sh = F.grid_sample(x_inp, grid_1, mode=self.affine_mode)

            grid_2 = F.affine_grid(rotation.to(self.device), x_inp.size()).to(
                self.device
            )
            output = F.grid_sample(out_sc_sh, grid_2, mode=self.affine_mode)
            
        
        # apply inverse affine to each diffraction spot if interpolate is True
        if self.interpolate:
            # Test 1.5 is good for 5%-45% background noise, add to 2 for larger noise and rot512x512 4dstem
            out_revise = revise_size_on_affine_gpu(
                output,
                self.mask,
                x.shape[0],
                scaler_shear,
                self.device,
                adj_para=mask_parameter,
                radius=self.radius,
                coef=self.coef,
                affine_mode=self.affine_mode,
            )

            return out_revise, k_out, scaler_shear, rotation, mask_parameter, x_inp

        else:
            return output, k_out, scaler_shear, rotation, mask_parameter


class Decoder(nn.Module):
    def __init__(self, original_step_size, up_list, conv_size, device, num_base=2):
        """_summary_

        Args:
            original_step_size (list of int): the x and y size of input image
            up_list (list of int): the list of parameter for each 2D Upsample layer
            conv_size (int): the value of filters number goes to each block
            device (torch.device): set the device to run the model
            num_base (int): the value for number of base. Defaults to 2.
        """

        super(Decoder, self).__init__()

        self.device = device

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(num_base, original_step_size[0] * original_step_size[1])
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

        self.relu_1 = nn.LeakyReLU(0.001)

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): input tensor

        Returns:
            tensor: output tensor
        """
        # reconstruct image into original size
        out = self.dense(x)
        out = out.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = self.relu_1(out)

        return out


class Joint(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        device,
        radius=60,
        coef=1.5,
        interpolate_mode="bicubic",
        affine_mode="bicubic",
    ):
        """_summary_

        Args:
            encoder (torch. Module): the encoder of neural network
            decoder (torch.Module): the decoder of neural network
            device (torch.device): set the device to run the model
            radius (int): set the radius of small square image for cropping. Defaults to 60.
            coef (float): set the threshold for COM operation. Defaults to 1.5.
            interpolate_size (string, optional): set the interpolate mode to function F.interpolate(). Defaults 'bicubic'.
            affine_mode (int): set the affine mode to function F.affine_grid(). Defaults 'bicubic'.
        """
        super(Joint, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Load variable from encoder
        self.mask_size = encoder.mask_size
        self.mask = encoder.mask
        self.interpolate = encoder.interpolate
        self.up_size = encoder.up_size
        self.radius = radius
        self.coef = coef
        self.interpolate_mode = interpolate_mode
        self.affine_mode = affine_mode

    def rotate_mask(self):
        """function return the mask list

        Returns:
            list: list of torch.bool array
        """

        return self.mask

    # if have pretrained rotation, add to forward function
    def forward(self, x, rotate_value=None):
        """

        Args:
            x (torch.tensor): input torch.tensor image
            rotate_value (float, optional): float value represents pretrained rotation angle. Defaults to None.
        """

        if self.interpolate:
            (predicted_revise,
            k_out,
            scaler_shear,
            rotation,
            translation,
            adj_mask,
            x_inp,
            ) = self.encoder(x, rotate_value)

        else:
            (predicted_revise, 
             k_out, 
             scaler_shear, 
             rotation, 
             translation, 
             adj_mask 
            ) = self.encoder(x, rotate_value)
        # create identity matrix for computing inverse affine matrix 
        identity = (
            torch.tensor([0, 0, 1], dtype=torch.float)
            .reshape(1, 1, 3)
            .repeat(x.shape[0], 1, 1)
            .to(self.device)
        )

        new_theta_1 = torch.cat((scaler_shear, identity), axis=1).to(self.device)
        new_theta_2 = torch.cat((rotation, identity), axis=1).to(self.device)

        inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
        inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)

        predicted_base = self.decoder(k_out)

        # upgrid image is interpolate mode is True
        if self.interpolate:
            predicted_base_inp = F.interpolate(
                predicted_base,
                size=(self.up_size, self.up_size),
                mode=self.interpolate_mode,
            )

            grid_1 = F.affine_grid(
                inver_theta_1.to(self.device), predicted_base_inp.size()
            ).to(self.device)
            grid_2 = F.affine_grid(
                inver_theta_2.to(self.device), predicted_base_inp.size()
            ).to(self.device)

            predicted_rotate = F.grid_sample(
                predicted_base_inp, grid_2, mode=self.affine_mode
            )
            predicted_input = F.grid_sample(
                predicted_rotate, grid_1, mode=self.affine_mode
            )

        else:
            grid_1 = F.affine_grid(inver_theta_1.to(self.device), x.size()).to(
                self.device
            )
            grid_2 = F.affine_grid(inver_theta_2.to(self.device), x.size()).to(
                self.device
            )

            predicted_rotate = F.grid_sample(predicted_base, grid_2)

            predicted_input = F.grid_sample(predicted_rotate, grid_1)
            
        # create new mask list to save updated mask region with inverse affine transform

        new_list = []
        if self.mask is not None:
            for mask_ in self.mask:
                batch_mask = (
                    mask_.reshape(1, 1, mask_.shape[-2], mask_.shape[-1]
                                  )).repeat(x.shape[0],0)

                batch_mask = torch.tensor(batch_mask, dtype=torch.float).to(self.device)

                rotated_mask = F.grid_sample(batch_mask, grid_2)

                if self.interpolate:
            # Add reverse affine transform of scale and shear to make all spots in the mask region, crucial when mask region small
                    rotated_mask = F.grid_sample(rotated_mask, grid_1)

            # maintain the correct size of mask region after affine transformation
                rotated_mask[rotated_mask < 0.5] = 0
                rotated_mask[rotated_mask >= 0.5] = 1

                rotated_mask = (
                    torch.tensor(rotated_mask, dtype=torch.bool).squeeze().to(self.device)
                )

                new_list.append(rotated_mask)

        if self.interpolate:
         # apply inverse affine transform to recreate input image
            predicted_input_revise = revise_size_on_affine_gpu(
                predicted_input,
                new_list,
                x.shape[0],
                inver_theta_1,
                self.device,
                adj_para=adj_mask,
                radius=self.radius,
                coef=self.coef,
                pare_reverse=True,
                affine_mode=self.affine_mode,
            )

            # change predicted_base to predicted_base_inp, add new_list when interpolate mode is True
            return (
                predicted_revise,
                predicted_base_inp,
                predicted_input_revise,
                k_out,
                scaler_shear,
                rotation,
                adj_mask,
                new_list,
                x_inp,
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


class PV_Joint(Joint):
    def __init__(self,pv_model,coords,r,orig_size,
                 encoder,decoder,device,radius=60,coef=1.5,
                 interp_size=None,
                 interpolate_mode="bicubic",affine_mode="bicubic",
                 masking='None'):
        """_summary_

        Args:
            pv_model (_type_): model for fitting gaussian spots. returns 
            coords (_type_): _description_
            r (_type_): _description_
            orig_size (_type_): _description_
            encoder (_type_): _description_
            decoder (_type_): _description_
            device (_type_): _description_
            radius (int, optional): _description_. Defaults to 60.
            coef (float, optional): _description_. Defaults to 1.5.
            interp_size (_type_, optional): _description_. Defaults to None.
            interpolate_mode (str, optional): _description_. Defaults to "bicubic".
            affine_mode (str, optional): _description_. Defaults to "bicubic".
        """        
        super(PV_Joint, self).__init__(encoder,decoder,device,radius,coef,interpolate_mode,affine_mode)
        self.pv_model = pv_model
        self.device=device
        self.tile_coords = torch.tensor(coords).float()
        self.r = r
        self.orig_size = orig_size
        
        self.interp_size = self.orig_size if interp_size is None else interp_size
        self.interp_factor = interp_size[-1]/orig_size[-1]
        self.r_inp = int(self.r*self.interp_factor)
        self.interp_mode = 'bilinear'
        self.revise_affine = True
        
        self.masking=masking
        if self.masking=='threshold':
            self.mask_thresh=0.5
            self.num_pxs=100
        if self.masking=='circle':
            self.mask_r = self.r//1.25 #torch.nn.Parameter(torch.tensor([self.r]).float())
            
    # @property
    # def device(self):
    #     return self._device

    # @device.setter
    # def device(self, device):
    #     self._device = device
    #     self.to(device)
    #     self._set_submodules_device(self, device)

    # def _set_submodules_device(self, module, device):
    #     for submodule in module.children():
    #         submodule.to(device)
    #         self._set_submodules_device(submodule, device)
            
    def check_bounds(self,x1,x2,y1,y2,shape):
        if abs(x2-x1)!=shape[0]: x2 = x1+shape[0]   
        if abs(y2-y1)!=shape[1]: y2 = y1+shape[1]
        return x1,x2,y1,y2
    
    def dilate_mask(self, base_mask, kernel_size=3, pxs=1):
        dilation_kernel = torch.ones((1,1,kernel_size,kernel_size),dtype=torch.float32).to(self.device)
        for i in range(pxs):
            dilated = F.conv2d(base_mask,dilation_kernel,padding=(kernel_size//2,kernel_size//2))
            base_mask = (dilated > 0).float()
        return base_mask

    def erode_mask(self,base_mask, kernel_size=3, pxs=1):
        kernel = torch.ones((1,1,kernel_size,kernel_size),dtype=torch.float32).to(self.device)
        for _ in range(pxs):
            eroded = F.conv2d(base_mask, kernel, padding=kernel_size//2)
            base_mask = (eroded == kernel.numel()).float()
        return base_mask

    def interpolate_images(self,images):
        x_,y_ = images.shape[-2:]
        interp_images = F.interpolate(
                images.reshape(-1,1,x_,y_), 
                size=( int(x_*self.interp_factor), int(y_*self.interp_factor) ), 
                mode=self.interp_mode ).squeeze()
        return interp_images

    def interpolate_coords(self,coords):
        return torch.round(coords*self.interp_factor).int()

    def transform_coords(self,coords,thetas,b_,s_):
        try: 
            affine_coords = torch.cat([coords, torch.ones(s_,1).to(self.device)],axis=1).repeat(b_,1,1)
            for theta in thetas:
                affine_coords = torch.einsum( 'bse,bej->bsj',affine_coords,theta)
        except:
            affine_coords = torch.cat([coords, torch.ones(b_,s_,1).to(self.device)],axis=2)
            for theta in thetas:
                if theta.shape[1]==2:
                    theta = torch.cat([theta, torch.zeros(b_,1,3).to(self.device)],axis=1)
                    theta[:,2,:]=1
                affine_coords = torch.einsum( 'bse,bej->bsj', affine_coords, theta)
        return affine_coords
            
    def inverse_scaleshear(self, data, theta, coords, coords_, r, b_, s_, mask_r,
                           masking='None', expand=1):
        r_ = lambda r: int(r*expand)
        # get tiles from affine basis at ae predicted coords
        tiles = spot.get_tiles( data.squeeze(),coords.int(),r_(r) ) # shape spots,bsize,2 
                                
        try: 
            tiles = torch.stack([torch.stack(tile) for tile in tiles]).float().to(self.device)
        except:
            tiles=torch.stack(tiles).float().to(self.device)
            tiles = tiles.swapaxes(0,1)

        grid = F.affine_grid( theta.repeat(s_,1,1)[:,:2,:].to(self.device), 
                               (b_*s_,1,r*2,r*2) ).to(self.device)
        out = F.grid_sample(tiles.reshape(-1,1,r*2,r*2), grid, 
                            mode=self.affine_mode)
        if masking=='circle':
            cx,cy=coords_.reshape(-1,2)[:,0], coords_.reshape(-1,2)[:,1] # TODO:right?
            masks = spot.get_circle_mask(mask_r,cx,cy,
                        output_shape=(r*2,r*2),device=self.device)
            out*=masks.reshape(-1,1,r*2,r*2)
        
        out = spot.replace_spots(out.reshape(b_,s_,r*2,r*2), coords, 
                                 self.r_inp,  b_, s_,
                                 output_size=data.shape[-2:], 
                                 device=self.device)
        return out


    def set_device(self, device):
        self.device = device
        self.to(device)
        for name, submodule in self.named_children():
            submodule.to(device)
            
    def forward(self, x, rotate_value=None, unshear=True):
        full,tiles = x
        # self.device = full.device
        b_,s_,x_,y_ = tiles.shape
        # print('pv_joint:', x[0].device)
        # self = self.to(x[0].device)
        
        # encoder
        if self.interpolate:
            (affine_base, # this uses full from different devices
             ae_coords,
             scaler_shear,
             rotation,
             translation,
             adj_mask,
             full_imp,
            ) = self.encoder(full.reshape(b_,1,self.orig_size[-2],self.orig_size[-1]))
        else:
            (affine_base, 
             ae_coords, 
             scaler_shea, 
             rotation, 
             translation, 
             adj_mask 
            ) = self.encoder(full.reshape(b_,1,self.orig_size[-2],self.orig_size[-1]), rotate_value)
        
        ae_coords = ae_coords.reshape(b_,s_,2) # bsize, numspots, (x,y)
        
        # compute inverse affine matrix 
        identity = ( torch.tensor([0, 0, 1], dtype=torch.float).reshape(1, 1, 3)
                    .repeat(b_, 1, 1).to(self.device) )
        new_theta_1 = torch.cat((scaler_shear, identity), axis=1).to(self.device)
        new_theta_2 = torch.cat((rotation, identity), axis=1).to(self.device)
        new_theta_3 = torch.cat((translation, identity), axis=1).to(self.device)
        inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
        inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)
        inver_theta_3 = torch.linalg.inv(new_theta_3)[:, 0:2].to(self.device)
        # # Apply inverse_inverse affine to predicted ae_coords of basis. should be the true centers of the diff spots
        # ae_inv_affine_coords = torch.cat([ae_coords, torch.ones(b_,s_,1).to(self.device)],axis=2)
        # for theta in [inver_theta_1,inver_theta_2,inver_theta_3]:
        #     _ =  torch.cat([theta, torch.zeros(b_,1,3).to(self.device)],axis=1)
        #     _[:,2,:]=1
        #     ae_inv_affine_coords = torch.einsum( 'bse,bej->bsj', ae_inv_affine_coords, _)

        
        # find pv fit and mask
        tiles = tiles.reshape(b_*s_,x_,y_).to(self.device)
        pv_embedding, pv_pred = self.pv_model(tiles)
        # Find pv masks
        if self.masking=='threshold': # TODO: test this
            masks, mask_loss, mask_thresh, num_pxs = spot.get_mask_loss(tiles,pv_pred,mask_thresh=self.mask_thresh,i=self.num_pxs)
            self.mask_thresh=mask_thresh
            self.num_pxs=num_pxs
            masks = masks.reshape(-1, self.r*2, self.r*2) # batch*tiles, ksize, ksize
            dilated_masks = self.dilate_mask(masks.unsqueeze(1)).squeeze()
            masked_tiles = (tiles*dilated_masks).reshape(b_,s_, self.r*2, self.r*2)
            eroded_masks = self.erode_mask(dilated_masks.unsqueeze(1),pxs=2).squeeze()
            masks = self.dilate_mask(eroded_masks.unsqueeze(1)).squeeze()    
            centroids = spot.get_centroids(masks).reshape(b_,s_,2).to(self.device)            
            
            # superimpose mask on ktop TODO: change so the replacement is only done with orig shapes
            base_masks = spot.replace_spots(masks.reshape(b_,s_,x_,y_),ae_coords,self.r,
                                        new_centroids=centroids,
                                        orig_size=self.orig_size,
                                        interp_size=self.interp_size,
                                        interp_mode=self.interp_mode,
                                        device=self.device)
            base_masks[base_masks<0.5] = 0
            base_masks[base_masks>=0.5] = 1
            predicted_base = spot.replace_spots(tiles.reshape(b_,s_,x_,y_),ae_coords,self.r,
                                                new_centroids=centroids,
                                                orig_size=self.orig_size,
                                                interp_size=self.interp_size,
                                                interp_mode=self.interp_mode,
                                                device=self.device) # base
            predicted_base *= base_masks
        
        if self.masking=='circle': 
            # r = torch.nn.ReLU()(self.mask_r) +self.r/2 # should be true radius of spot
            centroids = (ae_coords-self.tile_coords.to(self.device))
            cy,cx = centroids.reshape(-1,2)[:,0],centroids.reshape(-1,2)[:,1]
            masks = spot.get_circle_mask(self.mask_r,cx,cy,output_shape=(self.r*2,self.r*2),
                        interp_factor=1,device=self.device)
            masks,mask_loss,mask_thresh,num_pxs = spot.get_mask_loss(tiles,pv_pred,
                                                                     masking='preset',masks=masks,
                                                                     device=self.device)
            masks_inp = spot.get_circle_mask(self.mask_r,output_shape=(x_,y_),
                                            interp_factor=self.interp_factor,
                                            device=self.device)
            base_masks = spot.replace_spots(masks_inp.float(), 
                                            self.interpolate_coords(ae_coords), 
                                            self.r_inp, b_, s_, 
                                            output_size=self.interp_size[-2:],
                                            device=self.device)
            
        # put raw tiles on ae predicted coords. Labelled location superimposed on predicted. Not the true centers
        ae_base = spot.replace_spots(self.interpolate_images(tiles).reshape(b_,s_,self.r_inp*2,self.r_inp*2), 
                                    self.interpolate_coords(ae_coords), 
                                    self.r_inp,b_,s_,
                                    output_size=self.interp_size[-2:],
                                    device=self.device).unsqueeze(1)
        # # put raw tiles on inverse affine * ae predicted coords
        # predicted_input = spot.replace_spots(tiles.reshape(b_,s_,x_,y_), ae_inv_affine_coords, self.r,
        #                                 orig_size=self.orig_size,
        #                                 interp_size=self.interp_size,
        #                                 interp_mode=self.interp_mode,
        #                                 device=self.device)
        # put masks on raw coords
        raw_masks = spot.replace_spots(masks_inp.float(), 
                                       self.interpolate_coords(self.tile_coords.repeat(b_,1,1)), 
                                       self.r_inp,b_,s_,
                                        output_size=self.interp_size[-2:],
                                        device=self.device)
        
        # contruct inverse affine grids. Used to apply transformation to predicted base
        # ae_base = predicted_base
        # ae_base = F.interpolate( predicted_base.unsqueeze(1),
        #     size=(self.up_size, self.up_size),
        #     mode=self.interpolate_mode )
        grid_1 = F.affine_grid(inver_theta_1.to(self.device), ae_base.size()).to(self.device)
        grid_2 = F.affine_grid(inver_theta_2.to(self.device), ae_base.size()).to(self.device)
        grid_3 = F.affine_grid(inver_theta_3.to(self.device), ae_base.size()).to(self.device)
        predicted_translation = F.grid_sample(ae_base, grid_3, mode=self.affine_mode)          
        predicted_rotate = F.grid_sample(predicted_translation, grid_2, mode=self.affine_mode)
        predicted_input = F.grid_sample(predicted_rotate, grid_1, mode=self.affine_mode)
            
        # TODO: do inv affine on predicted revise tiles so they match shape of input. make a bit bigger to avoid cropping
        # get tiles from affine basis at ae predicted coords
        # affine_tiles = spot.get_tiles( affine_base.squeeze(), 
        #                                 torch.round( ae_coords.swapaxes ).int(), # shape spots,bsize,2 
        #                                 int(self.r*1.5) ) 
        # affine_tiles = torch.stack(affine_tiles).reshape(b_,s_,self.r*3,self.r*3)

        # affine_coords = torch.cat([self.tile_coords, torch.ones(s_,1).to(self.device)],axis=1).repeat(b_,1,1)
        # for theta in [new_theta_1,new_theta_2,new_theta_3]:
        #     affine_coords = torch.einsum( 'bse,bej->bsj',affine_coords,theta)
        
        affine_coords = self.transform_coords(self.tile_coords.to(self.device),
                                              [new_theta_1,new_theta_2,new_theta_3],
                                              b_,s_)[:,:,:2]
        if unshear:
            affine_base = self.inverse_scaleshear(affine_base, inver_theta_1,
                                              self.interpolate_coords(affine_coords),
                                              self.interpolate_coords(ae_coords),
                                              self.r_inp,b_,s_,int(self.mask_r*self.interp_factor),
                                            #   masking=self.masking
                                              ).unsqueeze(1)     
        
        pred_coords = self.transform_coords(ae_coords,[inver_theta_1,inver_theta_2,inver_theta_3],b_,s_)[:,:,:2]
        if unshear:
            predicted_input = self.inverse_scaleshear(predicted_input,new_theta_1,
                                                  self.interpolate_coords(pred_coords),
                                                  self.interpolate_coords(torch.zeros_like(pred_coords)),
                                                  self.r_inp,b_,s_,int(self.mask_r*self.interp_factor),
                                                #   masking=self.masking
                                                  ).unsqueeze(1)
        
        # grid_1 = F.affine_grid(inver_theta_1.to(self.device), raw_tiles.size()).to(self.device)
        # grid_2 = F.affine_grid(inver_theta_2.to(self.device), raw_tiles.size()).to(self.device)
        # grid_3 = F.affine_grid(inver_theta_3.to(self.device), raw_tiles.size()).to(self.device)
        # predicted_translation = F.grid_sample(ae_base, grid_3, mode=self.affine_mode)          
        # predicted_rotate = F.grid_sample(predicted_translation, grid_2, mode=self.affine_mode)
        # predicted_input = F.grid_sample(predicted_rotate, grid_1, mode=self.affine_mode)
            
        # affine_tiles *= inver_theta_3 # return the tiles to original square shape
        
        # affine_base_inp = spot.replace_spots(raw_tiles, affine_coords,self.r,
        #                                 orig_size=self.orig_size,
        #                                 interp_size=self.interp_size,
        #                                 interp_mode=self.interp_mode,
        #                                 device=self.device)
        # # create new mask list to save updated mask region with inverse affine transform
        # new_list = []
        # for mask_ in [base_mask]:
        #     batch_mask = (mask_.reshape(-1, 1, mask_.shape[-2], mask_.shape[-1]))
        #     batch_mask = batch_mask.clone().detach().float().to(self.device)
        #     rotated_mask = F.grid_sample(batch_mask, grid_2)

        #     if self.interpolate:
        # # Add reverse affine transform of scale and shear to make all spots in the mask region, crucial when mask region small
        #         rotated_mask = F.grid_sample(rotated_mask, grid_1)

        # # maintain the correct size of mask region after affine transformation
        #     rotated_mask[rotated_mask < 0.5] = 0
        #     rotated_mask[rotated_mask >= 0.5] = 1
        #     rotated_mask = rotated_mask.clone().detach().squeeze().to(self.device)
        #     new_list.append(rotated_mask)

        # if self.interpolate:
        #  # apply inverse affine transform to recreate input image
        #     if self.revise_affine:
        #         predicted_input_revise = revise_size_on_affine_gpu(
        #             predicted_input,
        #             base_masks,
        #             b_,
        #             inver_theta_1,
        #             self.device,
        #             adj_para=adj_mask,
        #             radius=self.radius,
        #             coef=self.coef,
        #             pare_reverse=True,
        #             affine_mode=self.affine_mode,
        #         )
        #     else: predicted_input_revise = predicted_input
                
            # change predicted_base to ae_base, add new_list when interpolate mode is True
        return (    affine_base,
                    ae_base,
                    predicted_input,
                    ae_coords,
                    scaler_shear,
                    rotation,
                    translation,
                    adj_mask,
                    raw_masks,
                    base_masks,
                    full_imp, 
                ), (
                    pv_embedding, 
                    pv_pred,
                    mask_loss, 
                    mask_thresh, 
                num_pxs)


        
def make_model_fn(
    device,
    learning_rate=3e-5,
    en_original_step_size=[200, 200],
    de_original_step_size=[5, 5],
    pool_list=[5, 4, 2],
    up_list=[2, 4, 5],
    conv_size=128,
    scale=True,
    shear=True,
    rotation=True,
    rotate_clockwise=True,
    translation=False,
    Symmetric=True,
    mask_intensity=True,
    num_base=1,
    up_size=800,
    scale_limit=0.05,
    shear_limit=0.1,
    rotation_limit=0.1,
    trans_limit=0.15,
    adj_mask_para=0,
    radius=60,
    coef=1.5,
    reduced_size=20,
    interpolate_mode="bicubic",
    affine_mode="bicubic",
    num_mask=6,
    fixed_mask=None,
    interpolate=True,
):
    """_summary_

    Args:
        device (torch.device): set the device initialize model
        learning_rate (float): learning rate to optimizer. Defaults to 3e-5.
        en_original_step_size (list of int): the x and y size of input image to encoder
        de_original_step_size (list of int): the x and y size of input image to decoder
        pool_list (list of int): the list of parameter for each 2D MaxPool layer
        embedding_size (int): the value for number of channels
        conv_size (int): the value of filters number goes to each block
        device (torch.device): set the device to run the model
        scale (bool): set to True if the model include scale affine transform
        shear (bool): set to True if the model include shear affine transform
        rotation (bool): set to True if the model include rotation affine transform
        rotate_clockwise (bool): set to True if the image should be rotated along one direction
        translation (bool): set to True if the model include translation affine transform
        Symmetric (bool): set to True if the shear affine transform is symmetric
        mask_intensity (bool):set to True if the intensity of the mask region is learnable
        num_base(int, optional): the value for number of base. Defaults to 2.
        fixed_mask (list of tensor, optional): The list of tensor with binary type. Defaults to None.
        num_mask (int, optional): the value for number of mask. Defaults to len(fixed_mask).
        interpolate (bool): set to determine if need to calculate loss value in interpolated version. Defaults to False.
        up_size (int, optional): the size of image to set for calculating MSE loss. Defaults to 800.
        scale_limit (float): set the range of scale. Defaults to 0.05.
        shear_limit (float): set the range of shear. Defaults to 0.1.
        rotation_limit (float): set the range of shear. Defaults to 0.1.
        trans_limit (float): set the range of translation. Defaults to 0.15.
        adj_mask_para (float): set the range of learnable parameter used to adjust pixel value in mask region. Defaults to 0.
        radius (int): set the radius of small square image for cropping. Defaults to 60.
        coef (float): set the threshold for COM operation. Defaults to 1.5.
        reduced_size (int): set the input length of K-top layer. Defaults 20.
        interpolate_size (string, optional): set the interpolate mode to function F.interpolate(). Defaults 'bicubic'.
        affine_mode (int): set the affine mode to function F.affine_grid(). Defaults 'bicubic'.

    Returns:
        torch.Module: pytorch model and optimizer
    """

    encoder = Encoder(
        en_original_step_size,
        pool_list,
        conv_size,
        device,
        scale,
        shear,
        rotation,
        rotate_clockwise,
        translation,
        Symmetric,
        mask_intensity,
        num_base,
        fixed_mask,
        num_mask,
        interpolate,
        up_size,
        scale_limit,
        shear_limit,
        rotation_limit,
        trans_limit,
        adj_mask_para,
        radius,
        coef,
        reduced_size,
        interpolate_mode,
        affine_mode,
    ).to(device)

    decoder = Decoder(de_original_step_size, 
                      up_list, 
                      conv_size, 
                      device, 
                      num_base
                    ).to(device)

    join = Joint(
        encoder, decoder, device, radius, coef, interpolate_mode, affine_mode
    ).to(device)

    optimizer = optim.Adam(join.parameters(), lr=learning_rate)

    join = torch.nn.parallel.DataParallel(join)

    return encoder, decoder, join, optimizer

                

def imshow_tensor(x,title='show'):
    import matplotlib.pyplot as plt
    plt.imshow(x.detach().cpu().numpy());
    plt.colorbar()
    plt.title(title)
    plt.show()