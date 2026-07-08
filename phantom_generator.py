import numpy as np
import json
import os

def generate_phantom():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratios_path = os.path.join(base_dir, 'scaling_ratios.json')
    
    # Load scaling factors
    if not os.path.exists(ratios_path):
        print("Scaling ratios not found. Running scale_generator.py first.")
        import scale_generator
        scale_generator.calculate_scaling()
        
    with open(ratios_path, 'r') as f:
        ratios = json.load(f)
        
    s_xy = ratios['scale_x']
    s_z = ratios['scale_z']
    
    print(f"Generating phantom with scaling factors: XY = {s_xy:.4f}, Z = {s_z:.4f}")
    
    # Define grid dimensions: 0.2 cm voxel size
    # Width (X): 20 cm => 100 voxels, from -10 cm to +10 cm
    # Depth (Y): 12 cm => 60 voxels, from -6 cm to +6 cm
    # Height (Z): 40 cm => 200 voxels, from 0 to 40 cm
    nx, ny, nz = 100, 60, 200
    voxel_size = 0.2
    
    # Coordinates of voxel centers
    x = -10.0 + (np.arange(nx) + 0.5) * voxel_size
    y = -6.0 + (np.arange(ny) + 0.5) * voxel_size
    z = (np.arange(nz) + 0.5) * voxel_size
    
    # Create 3D coordinate grids (shape: nz, ny, nx)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    # Initialize phantom array (0: Air)
    phantom = np.zeros((nz, ny, nx), dtype=np.uint8)
    
    # Target scaled radii
    torso_rx = 9.0 * s_xy
    torso_ry = 5.0 * s_xy
    
    neck_r = 3.0 * s_xy
    
    head_rx = 6.5 * s_xy
    head_ry = 7.5 * s_xy
    head_rz = 3.0 * s_z
    head_z0 = 37.0
    
    # 1. Torso body outline (z < 32 cm)
    torso_mask = (Z < 32.0) & (((X / torso_rx)**2 + (Y / torso_ry)**2) <= 1.0)
    
    # 2. Neck body outline (32 <= z < 34 cm)
    neck_mask = (Z >= 32.0) & (Z < 34.0) & ((X**2 + Y**2) <= neck_r**2)
    
    # 3. Head body outline (34 <= z < 40 cm)
    head_mask = (Z >= 34.0) & (Z < 40.0) & (((X / head_rx)**2 + (Y / head_ry)**2 + ((Z - head_z0) / head_rz)**2) <= 1.0)
    
    # Combine body masks (Muscle = 8)
    body_mask = torso_mask | neck_mask | head_mask
    phantom[body_mask] = 8
    
    # 4. Skin Layer (Skin = 7): boundary layer of body_mask
    # Find boundary voxels (any body voxel that has at least one air neighbor)
    eroded = body_mask.copy()
    # Check 6-connected neighbors by shifting
    eroded_z = eroded & np.roll(eroded, 1, axis=0) & np.roll(eroded, -1, axis=0)
    eroded_zy = eroded_z & np.roll(eroded, 1, axis=1) & np.roll(eroded, -1, axis=1)
    eroded_zyx = eroded_zy & np.roll(eroded, 1, axis=2) & np.roll(eroded, -1, axis=2)
    # Correct boundaries where roll wraps around
    eroded_zyx[0, :, :] = False
    eroded_zyx[-1, :, :] = False
    eroded_zyx[:, 0, :] = False
    eroded_zyx[:, -1, :] = False
    eroded_zyx[:, :, 0] = False
    eroded_zyx[:, :, -1] = False
    
    skin_mask = body_mask & ~eroded_zyx
    phantom[skin_mask] = 7
    
    # 5. Skeleton (Skeleton = 1)
    # A. Spine: vertical cylinder running from z = 0 to 34 cm
    spine_x0, spine_y0 = 0.0, -3.5 * s_xy
    spine_r = 1.0 * s_xy
    spine_mask = (Z < 34.0) & (((X - spine_x0)**2 + (Y - spine_y0)**2) <= spine_r**2) & body_mask
    phantom[spine_mask] = 1
    
    # B. Skull: shell in the head region (z >= 34 cm)
    skull_outer_mask = (Z >= 34.0) & (((X / head_rx)**2 + (Y / head_ry)**2 + ((Z - head_z0) / head_rz)**2) <= 1.0)
    skull_inner_rx = head_rx - 0.6 * s_xy
    skull_inner_ry = head_ry - 0.6 * s_xy
    skull_inner_rz = head_rz - 0.6 * s_z
    skull_inner_mask = (Z >= 34.0) & (((X / skull_inner_rx)**2 + (Y / skull_inner_ry)**2 + ((Z - head_z0) / skull_inner_rz)**2) <= 1.0)
    
    skull_shell_mask = skull_outer_mask & ~skull_inner_mask & body_mask
    phantom[skull_shell_mask] = 1
    
    # C. Rib cage: stylized ribs
    # Ribs are ellipses in horizontal planes at regular intervals
    rib_rx = torso_rx - 0.6 * s_xy
    rib_ry = torso_ry - 0.6 * s_xy
    for rib_z in [12, 15, 18, 21, 24, 27, 30]:
        # scaled rib heights
        scaled_z = rib_z * s_z
        rib_slice_mask = (Z >= scaled_z - 0.6) & (Z <= scaled_z + 0.6)
        # Rib loop
        rib_ring = rib_slice_mask & (((X / rib_rx)**2 + (Y / rib_ry)**2) >= 0.85) & (((X / rib_rx)**2 + (Y / rib_ry)**2) <= 1.0) & body_mask
        phantom[rib_ring] = 1
        
    # 6. Brain (Brain = 3)
    # Brain fills the inner skull cavity
    brain_mask = skull_inner_mask & body_mask
    phantom[brain_mask] = 3
    
    # 7. Lungs (Lungs = 2)
    # Two ellipsoids in the chest region (z from 16 to 30 cm)
    lung_z0 = 23.0 * s_z
    lung_rz = 6.0 * s_z
    lung_rx = 2.5 * s_xy
    lung_ry = 2.0 * s_xy
    lung_left_x0 = -4.0 * s_xy
    lung_right_x0 = 4.0 * s_xy
    lung_y0 = 0.5 * s_xy
    
    lung_l_mask = (Z >= 15.0 * s_z) & (Z <= 29.0 * s_z) & \
                  ((((X - lung_left_x0) / lung_rx)**2 + ((Y - lung_y0) / lung_ry)**2 + ((Z - lung_z0) / lung_rz)**2) <= 1.0)
    lung_r_mask = (Z >= 15.0 * s_z) & (Z <= 29.0 * s_z) & \
                  ((((X - lung_right_x0) / lung_rx)**2 + ((Y - lung_y0) / lung_ry)**2 + ((Z - lung_z0) / lung_rz)**2) <= 1.0)
    
    # Fill lungs but don't overwrite bone spine or ribs
    phantom[(lung_l_mask | lung_r_mask) & (phantom != 1)] = 2
    
    # 8. Heart (Heart = 4)
    # Ellipsoid in chest region (z from 18 to 25)
    heart_z0 = 21.0 * s_z
    heart_rz = 2.5 * s_z
    heart_rx = 2.0 * s_xy
    heart_ry = 1.8 * s_xy
    heart_x0 = -0.5 * s_xy
    heart_y0 = 1.0 * s_xy
    
    heart_mask = (Z >= 18.0 * s_z) & (Z <= 24.0 * s_z) & \
                 ((((X - heart_x0) / heart_rx)**2 + ((Y - heart_y0) / heart_ry)**2 + ((Z - heart_z0) / heart_rz)**2) <= 1.0)
    phantom[heart_mask & (phantom != 1)] = 4
    
    # 9. Liver (Liver = 5)
    # Right-lateral upper abdomen region (z from 10 to 16)
    liver_z0 = 13.0 * s_z
    liver_rz = 2.5 * s_z
    liver_rx = 3.5 * s_xy
    liver_ry = 3.0 * s_xy
    liver_x0 = 3.5 * s_xy
    liver_y0 = 0.5 * s_xy
    
    liver_mask = (Z >= 10.0 * s_z) & (Z <= 16.0 * s_z) & \
                 ((((X - liver_x0) / liver_rx)**2 + ((Y - liver_y0) / liver_ry)**2 + ((Z - liver_z0) / liver_rz)**2) <= 1.0)
    phantom[liver_mask & (phantom != 1) & (phantom != 4)] = 5
    
    # 10. Kidneys (Kidneys = 6)
    # Two small posterior ellipsoids in lower abdomen (z from 6 to 11)
    kidney_z0 = 8.5 * s_z
    kidney_rz = 2.0 * s_z
    kidney_rx = 1.5 * s_xy
    kidney_ry = 1.2 * s_xy
    kidney_left_x0 = -3.5 * s_xy
    kidney_right_x0 = 3.5 * s_xy
    kidney_y0 = -2.0 * s_xy
    
    kidney_l_mask = (Z >= 6.0 * s_z) & (Z <= 11.0 * s_z) & \
                    ((((X - kidney_left_x0) / kidney_rx)**2 + ((Y - kidney_y0) / kidney_ry)**2 + ((Z - kidney_z0) / kidney_rz)**2) <= 1.0)
    kidney_r_mask = (Z >= 6.0 * s_z) & (Z <= 11.0 * s_z) & \
                    ((((X - kidney_right_x0) / kidney_rx)**2 + ((Y - kidney_y0) / kidney_ry)**2 + ((Z - kidney_z0) / kidney_rz)**2) <= 1.0)
    phantom[(kidney_l_mask | kidney_r_mask) & (phantom != 1) & (phantom != 5)] = 6
    
    # Re-enforce skin layer where organs might have pushed to the edge
    # Only keep skin where skin_mask was defined
    phantom[skin_mask] = 7

    out_file = os.path.join(base_dir, 'phantom.npz')
    np.savez_compressed(out_file, phantom=phantom)
    print(f"Phantom generation complete!")
    print(f"  Voxel Grid Shape: {phantom.shape}")
    print(f"  Total Active Body Voxels: {np.sum(phantom > 0)}")
    print(f"  Organ Voxel Counts:")
    organ_names = ["Air", "Skeleton", "Lungs", "Brain", "Heart", "Liver", "Kidneys", "Skin", "Muscle"]
    for i, name in enumerate(organ_names):
        count = np.sum(phantom == i)
        vol = count * (voxel_size**3)
        print(f"    {name} (ID {i}): {count} voxels ({vol:.1f} cm³)")

if __name__ == '__main__':
    generate_phantom()
