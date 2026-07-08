import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_heatmap_plot(slice_z_cm=22.0):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    phantom_path = os.path.join(base_dir, 'phantom.npz')
    mesh_path = os.path.join(base_dir, 'mesh_results.npz')
    
    if not os.path.exists(phantom_path) or not os.path.exists(mesh_path):
        print("Required files (phantom.npz or mesh_results.npz) not found. Run simulation first.")
        return
        
    # Load phantom voxel array
    phantom_data = np.load(phantom_path)
    phantom = phantom_data['phantom']
    nz_p, ny_p, nx_p = phantom.shape
    
    # Load 3D mesh dose tally
    mesh_data = np.load(mesh_path)
    mesh_heating = mesh_data['mesh_heating']
    nz_m, ny_m, nx_m = mesh_heating.shape
    
    # Map height coordinate (0 to 40 cm) to indices
    idx_p = int((slice_z_cm / 40.0) * nz_p)
    idx_p = min(max(0, idx_p), nz_p - 1)
    
    idx_m = int((slice_z_cm / 40.0) * nz_m)
    idx_m = min(max(0, idx_m), nz_m - 1)
    
    print(f"Generating heatmap for slice Z = {slice_z_cm} cm (Phantom index {idx_p}, Mesh index {idx_m})")
    
    # Define custom organ colors
    organ_colors = [
        '#0b0f19', # Air (very dark slate)
        '#f1f5f9', # Skeleton (white/grey)
        '#38bdf8', # Lungs (sky blue)
        '#c084fc', # Brain (lavender purple)
        '#f43f5e', # Heart (rose red)
        '#b45309', # Liver (mahogany brown)
        '#f59e0b', # Kidneys (gold/orange)
        '#10b981', # Skin (emerald green)
        '#fda4af'  # Muscle (soft pink)
    ]
    cmap = mcolors.ListedColormap(organ_colors)
    
    # Physical coordinate bounds
    extent = [-10.0, 10.0, -6.0, 6.0]
    
    # Setup plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0f172a')
    
    # 1. Pure Dose Heatmap
    ax_heat = axes[0]
    ax_heat.set_facecolor('#0f172a')
    dose_slice = mesh_heating[idx_m, :, :]
    
    # Use logarithmic scale to visualize exponential radiation decay
    norm = mcolors.LogNorm(vmin=1e-1, vmax=max(1e1, dose_slice.max()))
    im_heat = ax_heat.imshow(dose_slice, cmap='inferno', origin='lower', extent=extent, norm=norm)
    ax_heat.set_title(f"Dose Heatmap Profile (Z = {slice_z_cm} cm)", color='white')
    ax_heat.set_xlabel("X coordinate (cm)", color='white')
    ax_heat.set_ylabel("Y coordinate (cm)", color='white')
    ax_heat.tick_params(colors='white')
    
    cbar1 = fig.colorbar(im_heat, ax=ax_heat, orientation='vertical')
    cbar1.ax.tick_params(colors='white')
    cbar1.set_label('Energy Deposited (eV / source particle)', color='white')
    
    # 2. Overlay Heatmap on Anatomy
    ax_over = axes[1]
    ax_over.set_facecolor('#0f172a')
    
    # Plot anatomy slice
    ax_over.imshow(phantom[idx_p, :, :], cmap=cmap, origin='lower', extent=extent, alpha=0.85)
    # Overlay dose heatmap with transparency
    im_over = ax_over.imshow(dose_slice, cmap='jet', origin='lower', extent=extent, norm=norm, alpha=0.45)
    
    ax_over.set_title(f"Anatomy and Dose Overlay (Z = {slice_z_cm} cm)", color='white')
    ax_over.set_xlabel("X coordinate (cm)", color='white')
    ax_over.set_ylabel("Y coordinate (cm)", color='white')
    ax_over.tick_params(colors='white')
    
    cbar2 = fig.colorbar(im_over, ax=ax_over, orientation='vertical')
    cbar2.ax.tick_params(colors='white')
    cbar2.set_label('Energy Deposited (eV / source particle)', color='white')
    
    plt.tight_layout()
    
    # Save output plot
    out_img_path = os.path.join(base_dir, 'dose_heatmap_slice.png')
    plt.savefig(out_img_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Heatmap image successfully saved to: {out_img_path}")

if __name__ == '__main__':
    generate_heatmap_plot(slice_z_cm=22.0)
