import json
import math
import os

def calculate_scaling():
    # Load demographics
    base_dir = os.path.dirname(os.path.abspath(__file__))
    demo_path = os.path.join(base_dir, 'demographics.json')
    with open(demo_path, 'r') as f:
        data = json.load(f)
        
    bd = data['bangladesh']
    icrp = data['icrp_reference_male']
    
    # Calculate Z scaling
    sz = bd['height_cm'] / icrp['height_cm']
    
    # Calculate lateral scaling (X, Y) assuming isotropic scaling in XY plane
    # weight ratio = S_x * S_y * S_z = S_xy^2 * S_z
    weight_ratio = bd['weight_kg'] / icrp['weight_kg']
    sxy = math.sqrt(weight_ratio / sz)
    
    ratios = {
        "scale_x": sxy,
        "scale_y": sxy,
        "scale_z": sz,
        "weight_ratio": weight_ratio
    }
    
    out_path = os.path.join(base_dir, 'scaling_ratios.json')
    with open(out_path, 'w') as f:
        json.dump(ratios, f, indent=2)
        
    print(f"Calculated Scaling Ratios:")
    print(f"  Z-scale (Height): {sz:.4f}")
    print(f"  XY-scale (Lateral): {sxy:.4f}")
    print(f"  Saved to: {out_path}")

if __name__ == "__main__":
    calculate_scaling()
