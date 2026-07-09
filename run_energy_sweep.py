import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import json
import numpy as np
import openmc

def run_sweep():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    phantom_path = os.path.join(base_dir, 'phantom.npz')
    if not os.path.exists(phantom_path):
        raise FileNotFoundError("Phantom file not found. Run phantom_generator.py first.")
        
    data = np.load(phantom_path)
    phantom = data['phantom']
    
    # Define standard energy bins in keV
    energies_kev = [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    particles = 5000  # 5,000 histories per energy step for rapid execution
    
    sweep_results = {}
    organ_names = ["Skeleton", "Lungs", "Brain", "Heart", "Liver", "Kidneys", "Skin", "Muscle"]
    
    # Densities and masses of organs (material IDs 11 to 18)
    densities = [1.92, 0.26, 1.04, 1.06, 1.06, 1.05, 1.09, 1.05]
    voxel_volume = 0.2 * 0.2 * 0.2  # 0.008 cm3
    
    masses = []
    for i in range(1, 9):
        voxel_count = np.sum(phantom == i)
        masses.append(voxel_count * voxel_volume * densities[i-1])
        
    # Set cross sections environment variable dynamically
    default_cross_sec = '/Users/rafsunhossain/nndc_/cross_sections.xml'
    cross_sec_path = os.environ.get('OPENMC_CROSS_SECTIONS', default_cross_sec)
    if not os.path.exists(cross_sec_path) and os.path.exists(default_cross_sec):
        cross_sec_path = default_cross_sec
    os.environ['OPENMC_CROSS_SECTIONS'] = cross_sec_path
    
    for energy in energies_kev:
        print(f"Executing OpenMC simulation for energy = {energy} keV...")
        
        # Update settings for this energy bin
        settings = openmc.Settings()
        settings.run_mode = 'fixed source'
        settings.batches = 5
        settings.inactive = 0
        settings.particles = particles
        
        source = openmc.Source()
        source.space = openmc.stats.Box((-10.0, 5.8, 0.0), (10.0, 5.9, 40.0))
        source.angle = openmc.stats.Monodirectional((0.0, -1.0, 0.0))
        source.energy = openmc.stats.Discrete([energy * 1000.0], [1.0]) # eV
        source.particle = 'photon'
        
        settings.source = source
        settings.photon_transport = True
        settings.export_to_xml()
        
        # Run OpenMC
        openmc.run(openmc_exec='/opt/anaconda3/envs/openmc-env/bin/openmc')
        
        # Read tally results
        sp = openmc.StatePoint('statepoint.5.h5')
        tally_output = sp.get_tally(name='organ_dose')
        
        for i, name in enumerate(organ_names):
            if name not in sweep_results:
                sweep_results[name] = {"energies": [], "dose_ev_g": [], "uncertainty_rel": []}
                
            mat_id = 11 + i
            mean = float(tally_output.get_values(filters=[openmc.MaterialFilter], filter_bins=[(mat_id,)], scores=['heating'], value='mean')[0][0][0])
            std_dev = float(tally_output.get_values(filters=[openmc.MaterialFilter], filter_bins=[(mat_id,)], scores=['heating'], value='std_dev')[0][0][0])
            
            # Dose in eV/g
            dose = mean / masses[i]
            # Relative uncertainty in %
            rel_unc = (std_dev / mean * 100.0) if mean > 0.0 else 0.0
            
            sweep_results[name]["energies"].append(energy)
            sweep_results[name]["dose_ev_g"].append(dose)
            sweep_results[name]["uncertainty_rel"].append(rel_unc)
            
    # Export results
    sweep_path = os.path.join(base_dir, 'energy_sweep_results.json')
    with open(sweep_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
        
    print(f"Energy sweep completed. Output saved to: {sweep_path}")

if __name__ == '__main__':
    run_sweep()
