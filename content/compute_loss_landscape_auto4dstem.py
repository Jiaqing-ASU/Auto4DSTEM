import os
import torch
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from cmcrameri import cm
from m3util.viz.text import labelfigs
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Make labelfigs available globally
import builtins

builtins.labelfigs = labelfigs

# from Auto4DSTEM.src.auto4dstem.nn.Train_Function import TrainClass
# from Auto4DSTEM.src.auto4dstem.Viz.util import mask_class
# from Auto4DSTEM.src.auto4dstem.Viz.viz import set_format_Auto4D, visualize_simulate_result, visual_performance_plot,normalized_strain_matrices
from m3util.util.IO import download_files_from_txt
from auto4dstem.nn.Train_Function import TrainClass
from auto4dstem.Viz.util import mask_class
from auto4dstem.Viz.viz import (
    set_format_Auto4D,
    visualize_simulate_result,
    visual_performance_plot,
    normalized_strain_matrices,
)


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Add memory management config for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

# Set device and GPU settings
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Explicitly use first GPU
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    n_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {n_gpus}")

    # Print available GPU memory
    for i in range(n_gpus):
        free_mem = torch.cuda.get_device_properties(
            i
        ).total_memory - torch.cuda.memory_allocated(i)
        print(f"GPU {i} free memory: {free_mem / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")
    n_gpus = 0

set_format = set_format_Auto4D()
pylab.rcParams.update(set_format)
warnings.filterwarnings("ignore")

folder_name = (
    "Simulated_4dstem/Extremely_Noisy_4DSTEM_Strain_Mapping_Using_CC_ST_AE_Simulated"
)
file_download = "simulated_label_weights_affine_para"


# set mask class
set_mask = mask_class()
# generate mask
mask_tensor, mask_list = set_mask.mask_ring(radius_1=50, radius_2=85)
# download_files_from_txt(file_download, folder_name)

print("set BKG level, load data and pretrained weights")
list_bkg_intensity = [
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.6,
    0.7,
]

for bkg_intensity in tqdm(list_bkg_intensity, desc="Processing background intensities"):
    bkg_str = format(int(bkg_intensity * 100), "02d")
    data_path = os.path.abspath(f"{folder_name}/polycrystal_output4D.mat")
    rotation_path = f"{folder_name}/{bkg_str}Percent_pretrained_rotation.npy"

    print("Initialize the training class from noise-free simulated dataset")

    # Initialize model with appropriate batch size for GPU memory
    tc = TrainClass(
        data_path,
        device=device,
        transpose=(1, 0, 3, 2),
        background_weight=bkg_intensity,
        learned_rotation=rotation_path,
        adjust_learned_rotation=0,
        num_base=1,
        up_size=800,
        scale_limit=0.05,
        shear_limit=0.1,
        rotation_limit=0.1,
        trans_limit=0.15,
        adj_mask_para=0,
        fixed_mask=mask_list,
        check_mask=None,
        interpolate=True,
        revise_affine=False,
        folder_path=folder_name,
        batch_size=8,  # Reduced batch size for GPU memory
    )

    # Initialize the model
    print("Initializing model...")
    tc.encoder, tc.decoder, tc.join, tc.optimizer = tc.reset_model()

    # Move model to GPU
    tc.join = tc.join.to(device)
    tc.encoder = tc.encoder.to(device)
    tc.decoder = tc.decoder.to(device)

    # Load pretrained weights
    print("Loading pretrained weights...")
    weight_path = (
        f"{folder_name}/{bkg_str}percent_noisy_simulated_4dstem_pretrained_weights.pkl"
    )
    tc.load_pretrained_weight(weight_path)

    # Clear GPU cache
    torch.cuda.empty_cache()

    tc.crop_one_image(clim=[0, 4e-5])
    tc.visual_noise(noise_level=[0, 0.25, 0.6], file_name="simulated")

    # Process predictions
    # print("Processing predictions...")
    # tc.predict(
    #     train_process="2",
    #     save_strain=True,
    #     save_rotation=True,
    #     file_name=bkg_intensity,
    #     num_workers=4
    # )
    torch.cuda.empty_cache()

    # tc.predict(
    #     train_process="2",
    #     save_strain=False,
    #     save_rotation=False,
    #     file_name=bkg_intensity,
    #     num_workers=4
    # )
    # torch.cuda.empty_cache()

    # Compute loss landscape
    print("Computing loss landscape...")
    try:
        # Define loss landscape parameters
        ll_distance = 0.75  # Distance parameter
        ll_points = 50  # Number of points
        ll_subset = 50  # Subset size

        loss_surface, x_coords, y_coords = tc.compute_loss_landscape(
            distance=ll_distance,
            n_points=ll_points,
            data_subset_size=ll_subset,
            normalization="filter",
        )

        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(folder_name, "loss_landscape_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create base filename with parameters
        base_filename = f"loss_landscape_bkg{bkg_intensity}_d{ll_distance}_n{ll_points}"
        plot_title = (
            f"Loss Landscape (bkg={bkg_intensity}, d={ll_distance}, n={ll_points})"
        )

        # Create 2D contour plot
        plt.figure(figsize=(12, 10))
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        contour = plt.contour(x_grid, y_grid, loss_surface, levels=40)
        plt.colorbar(contour, label="Loss")
        plt.xlabel("Direction 1")
        plt.ylabel("Direction 2")
        plt.title(f"{plot_title} - Contour")
        plt.plot([0], [0], "r.", markersize=10, label="Model Position")
        plt.legend()
        plt.savefig(
            os.path.join(plots_dir, f"{base_filename}_contour.svg"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # Create 3D surface plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            x_grid, y_grid, loss_surface, cmap="viridis", rcount=100, ccount=100
        )
        fig.colorbar(surf, label="Loss")
        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Loss")
        ax.set_title(f"{plot_title} - Surface")
        ax.scatter(
            [0],
            [0],
            [loss_surface[len(loss_surface) // 2, len(loss_surface) // 2]],
            color="red",
            s=100,
            label="Model Position",
        )
        ax.legend()
        plt.savefig(
            os.path.join(plots_dir, f"{base_filename}_3d.svg"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # Create filled contour plot
        plt.figure(figsize=(12, 10))
        contourf = plt.contourf(x_grid, y_grid, loss_surface, levels=40, cmap="viridis")
        plt.colorbar(contourf, label="Loss")
        plt.xlabel("Direction 1")
        plt.ylabel("Direction 2")
        plt.title(f"{plot_title} - Filled Contour")
        plt.plot([0], [0], "r.", markersize=10, label="Model Position")
        plt.legend()
        plt.savefig(
            os.path.join(plots_dir, f"{base_filename}_filled_contour.svg"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # Save the computed landscape data with parameters in filename
        np.savez(
            os.path.join(plots_dir, f"{base_filename}_data.npz"),
            loss_surface=loss_surface,
            x_coords=x_coords,
            y_coords=y_coords,
            distance=ll_distance,
            n_points=ll_points,
            subset_size=ll_subset,
            background_intensity=bkg_intensity,
        )

        print(f"Loss landscape visualization completed. Plots saved in {plots_dir}")

    except Exception as e:
        print(f"Error computing loss landscape: {str(e)}")
        raise
