import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import os

# Global configuration
STEPS = 5  # number of steps in each direction
DISTANCE = 0.5  # maximum distance from start point

def create_loss_landscape_plots(data_path, output_dir):
    """Create 2D and 3D visualizations of the loss landscape"""
    # Load the loss landscape data
    data = np.load(data_path)
    
    # Get the data arrays
    z = data['loss_surface']  # The loss values
    x = data['x_coordinates']  # x coordinates
    y = data['y_coordinates']  # y coordinates
    X, Y = np.meshgrid(x, y)
    
    # Verify dimensions match global settings
    if len(x) != STEPS or len(y) != STEPS:
        print(f"Warning: Data dimensions ({len(x)}x{len(y)}) don't match STEPS ({STEPS})")
    if abs(x[-1]) != DISTANCE or abs(y[-1]) != DISTANCE:
        print(f"Warning: Data range ({abs(x[-1])}) doesn't match DISTANCE ({DISTANCE})")
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(20, 8))
    
    # First subplot: Contour plot
    ax1 = fig.add_subplot(121)
    levels = np.logspace(np.log10(z.min()), np.log10(z.max()), 30)
    contour_filled = ax1.contourf(X, Y, z, 
                                levels=levels,
                                norm=LogNorm(),
                                cmap='RdYlBu_r')
    contour_lines = ax1.contour(X, Y, z,
                               levels=levels[::3],
                               colors='black',
                               linewidths=0.5,
                               alpha=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')
    colorbar = plt.colorbar(contour_filled, ax=ax1, label='Loss (log scale)', format='%.0e')
    colorbar.ax.tick_params(labelsize=10)
    ax1.set_xlabel('Direction 1', fontsize=12)
    ax1.set_ylabel('Direction 2', fontsize=12)
    ax1.set_title('Loss Landscape Contour', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.axis('equal')
    
    # Second subplot: 3D surface plot
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, z, 
                           cmap='RdYlBu_r',
                           norm=LogNorm(z.min(), z.max()),
                           linewidth=0,
                           antialiased=True)
    colorbar2 = plt.colorbar(surf, ax=ax2, label='Loss (log scale)', format='%.0e')
    colorbar2.ax.tick_params(labelsize=10)
    ax2.set_xlabel('Direction 1', fontsize=12)
    ax2.set_ylabel('Direction 2', fontsize=12)
    ax2.set_zlabel('Loss', fontsize=12)
    ax2.set_title('Loss Landscape 3D Surface', fontsize=14)
    ax2.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_landscape_2d_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, z,
                          cmap='RdYlBu_r',
                          norm=LogNorm(z.min(), z.max()),
                          linewidth=0,
                          antialiased=True)
    colorbar = plt.colorbar(surf, label='Loss (log scale)', format='%.0e')
    colorbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Direction 1', fontsize=12)
    ax.set_ylabel('Direction 2', fontsize=12)
    ax.set_zlabel('Loss', fontsize=12)
    ax.set_title('Loss Landscape 3D Surface', fontsize=14)
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(output_dir, 'loss_landscape_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Directory containing the loss landscape npz files
    landscapes_dir = 'loss_landscapes'
    
    # Process each npz file in the directory
    for filename in os.listdir(landscapes_dir):
        if filename.endswith('.npz'):
            # Verify file matches current settings
            if f"steps{STEPS}_dist{DISTANCE:.1f}" not in filename:
                print(f"Warning: File {filename} may have different parameters than current settings")
            
            # Create output directory based on npz filename
            output_name = filename[:-4]  # Remove .npz extension
            output_dir = os.path.join(landscapes_dir, output_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate plots
            data_path = os.path.join(landscapes_dir, filename)
            print(f"Processing {filename}...")
            create_loss_landscape_plots(data_path, output_dir)
            print(f"Plots saved in {output_dir}")

if __name__ == "__main__":
    main()
