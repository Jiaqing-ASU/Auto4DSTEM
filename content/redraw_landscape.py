import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import os

# Global configuration
# Extract steps and distance from filename (e.g. "25Percent_steps21_dist50.0_loss_landscape.npz")
def get_params_from_filename(filename):
    parts = filename.split('_')
    steps = int(parts[1].replace('steps', ''))
    dist = float(parts[2].replace('dist', ''))
    return steps, dist

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
    
    # Create a figure with three subplots
    fig = plt.figure(figsize=(25, 8))
    
    # First subplot: Contour plot
    ax1 = fig.add_subplot(131)
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
    colorbar = plt.colorbar(contour_filled, ax=ax1, label='Loss', format='%.2f')
    colorbar.ax.tick_params(labelsize=10)
    ax1.set_xlabel('Direction 1', fontsize=12)
    ax1.set_ylabel('Direction 2', fontsize=12)
    ax1.set_title('Loss Landscape Contour', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.axis('equal')
    
    # Second subplot: Heatmap
    ax2 = fig.add_subplot(132)
    heatmap = ax2.pcolormesh(X, Y, z, 
                            norm=LogNorm(),
                            cmap='RdYlBu_r')
    colorbar2 = plt.colorbar(heatmap, ax=ax2, label='Loss', format='%.2f')
    colorbar2.ax.tick_params(labelsize=10)
    ax2.set_xlabel('Direction 1', fontsize=12)
    ax2.set_ylabel('Direction 2', fontsize=12)
    ax2.set_title('Loss Landscape Heatmap', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.axis('equal')
    
    # Third subplot: 3D surface plot
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X, Y, z, 
                           cmap='RdYlBu_r',
                           norm=LogNorm(z.min(), z.max()),
                           linewidth=0,
                           antialiased=True)
    colorbar3 = plt.colorbar(surf, ax=ax3, label='Loss', format='%.2f')
    colorbar3.ax.tick_params(labelsize=10)
    ax3.set_xlabel('Direction 1', fontsize=12)
    ax3.set_ylabel('Direction 2', fontsize=12)
    ax3.set_zlabel('Loss', fontsize=12)
    ax3.set_title('Loss Landscape 3D Surface', fontsize=14)
    ax3.view_init(elev=30, azim=45)
    
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
    colorbar = plt.colorbar(surf, label='Loss', format='%.2f')
    colorbar.ax.tick_params(labelsize=10)
    ax.set_xlabel('Direction 1', fontsize=12)
    ax.set_ylabel('Direction 2', fontsize=12)
    ax.set_zlabel('Loss', fontsize=12)
    ax.set_title('Loss Landscape 3D Surface', fontsize=14)
    ax.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(output_dir, 'loss_landscape_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create 4 separate 3D plots with log scale on z-axis from different angles
    view_angles = [
        (30, 45),    # Default view
        (30, 135),   # Rotated 90 degrees
        (30, 225),   # Rotated 180 degrees
        (30, 315)    # Rotated 270 degrees
    ]

    for i, (elev, azim) in enumerate(view_angles):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        log_z = np.log10(z)  # Convert to log scale for shape
        surf = ax.plot_surface(X, Y, log_z,
                             cmap='RdYlBu_r',
                             linewidth=0,
                             antialiased=True)
        
        # Set up the colorbar with actual loss values
        colorbar = plt.colorbar(surf, label='Loss', format='%.2f')
        colorbar.ax.tick_params(labelsize=10)
        
        # Convert log ticks to actual values for z-axis
        log_ticks = ax.get_zticks()
        actual_ticks = 10 ** log_ticks
        ax.set_zticklabels([f'{x:.2f}' for x in actual_ticks])
        
        ax.set_xlabel('Direction 1', fontsize=12)
        ax.set_ylabel('Direction 2', fontsize=12)
        ax.set_zlabel('Loss', fontsize=12)
        ax.set_title(f'Loss Landscape 3D Surface - View {i+1}', fontsize=14)
        ax.view_init(elev=elev, azim=azim)
        
        # Convert colorbar ticks to actual values
        log_cticks = colorbar.ax.get_yticks()
        actual_cticks = 10 ** log_cticks
        colorbar.ax.set_yticklabels([f'{x:.2f}' for x in actual_cticks])
        
        plt.savefig(os.path.join(output_dir, f'loss_landscape_3d_log_view{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Directory containing the loss landscape npz files
    landscapes_dir = 'loss_landscapes'
    
    # Process each npz file in the directory
    for filename in os.listdir(landscapes_dir):
        if filename.endswith('.npz'):
            # Get parameters from filename
            global STEPS, DISTANCE
            STEPS, DISTANCE = get_params_from_filename(filename)
            
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
