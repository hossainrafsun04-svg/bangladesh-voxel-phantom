import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import argparse
import json
import numpy as np
import openmc

def run_simulation(energy_kev=100.0, particles=10000):
    print(f"Setting up OpenMC simulation...")
    print(f"  Photon Energy: {energy_kev} keV")
    print(f"  Particle Count: {particles}")
    
    # 1. Load Phantom Voxel Grid
    base_dir = os.path.dirname(os.path.abspath(__file__))
    phantom_path = os.path.join(base_dir, 'phantom.npz')
    if not os.path.exists(phantom_path):
        raise FileNotFoundError(f"Phantom file {phantom_path} not found. Please run phantom_generator.py first.")
        
    data = np.load(phantom_path)
    phantom = data['phantom']
    nz, ny, nx = phantom.shape
    
    # Set cross sections environment variable dynamically
    default_cross_sec = '/Users/rafsunhossain/nndc_/cross_sections.xml'
    cross_sec_path = os.environ.get('OPENMC_CROSS_SECTIONS', default_cross_sec)
    if not os.path.exists(cross_sec_path) and os.path.exists(default_cross_sec):
        cross_sec_path = default_cross_sec
    os.environ['OPENMC_CROSS_SECTIONS'] = cross_sec_path
    print(f"  Cross-Sections path: {cross_sec_path}")
    
    # 2. Define Materials (ICRP/ICRU-116 standards)
    materials = {}
    
    # ID 0: Air
    materials[0] = openmc.Material(material_id=10, name='Air')
    materials[0].set_density('g/cm3', 0.0012)
    materials[0].add_element('N', 0.755, percent_type='wo')
    materials[0].add_element('O', 0.232, percent_type='wo')
    materials[0].add_element('Ar', 0.013, percent_type='wo')

    # ID 1: Skeleton (Cortical Bone)
    materials[1] = openmc.Material(material_id=11, name='Skeleton')
    materials[1].set_density('g/cm3', 1.92)
    materials[1].add_element('H', 0.034, percent_type='wo')
    materials[1].add_element('C', 0.155, percent_type='wo')
    materials[1].add_element('N', 0.042, percent_type='wo')
    materials[1].add_element('O', 0.435, percent_type='wo')
    materials[1].add_element('Na', 0.001, percent_type='wo')
    materials[1].add_element('Mg', 0.002, percent_type='wo')
    materials[1].add_element('P', 0.103, percent_type='wo')
    materials[1].add_element('S', 0.003, percent_type='wo')
    materials[1].add_element('Ca', 0.225, percent_type='wo')

    # ID 2: Lungs
    materials[2] = openmc.Material(material_id=12, name='Lungs')
    materials[2].set_density('g/cm3', 0.26)
    materials[2].add_element('H', 0.103, percent_type='wo')
    materials[2].add_element('C', 0.105, percent_type='wo')
    materials[2].add_element('N', 0.031, percent_type='wo')
    materials[2].add_element('O', 0.749, percent_type='wo')
    materials[2].add_element('Na', 0.002, percent_type='wo')
    materials[2].add_element('P', 0.002, percent_type='wo')
    materials[2].add_element('S', 0.003, percent_type='wo')
    materials[2].add_element('Cl', 0.003, percent_type='wo')
    materials[2].add_element('K', 0.002, percent_type='wo')

    # ID 3: Brain
    materials[3] = openmc.Material(material_id=13, name='Brain')
    materials[3].set_density('g/cm3', 1.04)
    materials[3].add_element('H', 0.107, percent_type='wo')
    materials[3].add_element('C', 0.145, percent_type='wo')
    materials[3].add_element('N', 0.022, percent_type='wo')
    materials[3].add_element('O', 0.712, percent_type='wo')
    materials[3].add_element('Na', 0.002, percent_type='wo')
    materials[3].add_element('P', 0.004, percent_type='wo')
    materials[3].add_element('S', 0.002, percent_type='wo')
    materials[3].add_element('Cl', 0.003, percent_type='wo')
    materials[3].add_element('K', 0.003, percent_type='wo')

    # ID 4: Heart
    materials[4] = openmc.Material(material_id=14, name='Heart')
    materials[4].set_density('g/cm3', 1.06)
    materials[4].add_element('H', 0.104, percent_type='wo')
    materials[4].add_element('C', 0.139, percent_type='wo')
    materials[4].add_element('N', 0.032, percent_type='wo')
    materials[4].add_element('O', 0.712, percent_type='wo')
    materials[4].add_element('Na', 0.001, percent_type='wo')
    materials[4].add_element('P', 0.002, percent_type='wo')
    materials[4].add_element('S', 0.002, percent_type='wo')
    materials[4].add_element('Cl', 0.002, percent_type='wo')
    materials[4].add_element('K', 0.003, percent_type='wo')
    materials[4].add_element('Fe', 0.003, percent_type='wo')

    # ID 5: Liver
    materials[5] = openmc.Material(material_id=15, name='Liver')
    materials[5].set_density('g/cm3', 1.06)
    materials[5].add_element('H', 0.102, percent_type='wo')
    materials[5].add_element('C', 0.139, percent_type='wo')
    materials[5].add_element('N', 0.030, percent_type='wo')
    materials[5].add_element('O', 0.713, percent_type='wo')
    materials[5].add_element('Na', 0.002, percent_type='wo')
    materials[5].add_element('P', 0.003, percent_type='wo')
    materials[5].add_element('S', 0.003, percent_type='wo')
    materials[5].add_element('Cl', 0.002, percent_type='wo')
    materials[5].add_element('K', 0.003, percent_type='wo')
    materials[5].add_element('Fe', 0.003, percent_type='wo')

    # ID 6: Kidneys
    materials[6] = openmc.Material(material_id=16, name='Kidneys')
    materials[6].set_density('g/cm3', 1.05)
    materials[6].add_element('H', 0.103, percent_type='wo')
    materials[6].add_element('C', 0.125, percent_type='wo')
    materials[6].add_element('N', 0.032, percent_type='wo')
    materials[6].add_element('O', 0.728, percent_type='wo')
    materials[6].add_element('Na', 0.002, percent_type='wo')
    materials[6].add_element('P', 0.002, percent_type='wo')
    materials[6].add_element('S', 0.002, percent_type='wo')
    materials[6].add_element('Cl', 0.002, percent_type='wo')
    materials[6].add_element('K', 0.002, percent_type='wo')
    materials[6].add_element('Fe', 0.003, percent_type='wo')

    # ID 7: Skin
    materials[7] = openmc.Material(material_id=17, name='Skin')
    materials[7].set_density('g/cm3', 1.09)
    materials[7].add_element('H', 0.100, percent_type='wo')
    materials[7].add_element('C', 0.199, percent_type='wo')
    materials[7].add_element('N', 0.042, percent_type='wo')
    materials[7].add_element('O', 0.647, percent_type='wo')
    materials[7].add_element('Na', 0.002, percent_type='wo')
    materials[7].add_element('P', 0.001, percent_type='wo')
    materials[7].add_element('S', 0.002, percent_type='wo')
    materials[7].add_element('Cl', 0.003, percent_type='wo')
    materials[7].add_element('K', 0.001, percent_type='wo')
    materials[7].add_element('Fe', 0.003, percent_type='wo')

    # ID 8: Muscle
    materials[8] = openmc.Material(material_id=18, name='Muscle')
    materials[8].set_density('g/cm3', 1.05)
    materials[8].add_element('H', 0.102, percent_type='wo')
    materials[8].add_element('C', 0.143, percent_type='wo')
    materials[8].add_element('N', 0.034, percent_type='wo')
    materials[8].add_element('O', 0.710, percent_type='wo')
    materials[8].add_element('Na', 0.001, percent_type='wo')
    materials[8].add_element('P', 0.002, percent_type='wo')
    materials[8].add_element('S', 0.003, percent_type='wo')
    materials[8].add_element('Cl', 0.001, percent_type='wo')
    materials[8].add_element('K', 0.004, percent_type='wo')
    materials[8].add_element('Fe', 0.001, percent_type='wo')

    # Export materials to XML
    openmc_materials = openmc.Materials(list(materials.values()))
    openmc_materials.export_to_xml()
    
    # 3. Setup Geometry & Lattice
    # Bounding planes
    x_min = openmc.XPlane(x0=-10.0, boundary_type='vacuum')
    x_max = openmc.XPlane(x0=10.0, boundary_type='vacuum')
    y_min = openmc.YPlane(y0=-6.0, boundary_type='vacuum')
    y_max = openmc.YPlane(y0=6.0, boundary_type='vacuum')
    z_min = openmc.ZPlane(z0=0.0, boundary_type='vacuum')
    z_max = openmc.ZPlane(z0=40.0, boundary_type='vacuum')
    
    phantom_box_cell = openmc.Cell(cell_id=999, name='Phantom Box Cell')
    phantom_box_cell.region = +x_min & -x_max & +y_min & -y_max & +z_min & -z_max
    
    # Create cells and universes for the lattice
    universes = {}
    for organ_id, mat in materials.items():
        cell = openmc.Cell(cell_id=100+organ_id, name=f'Cell for {mat.name}')
        cell.fill = mat
        univ = openmc.Universe(universe_id=100+organ_id, name=f'Univ for {mat.name}')
        univ.add_cell(cell)
        universes[organ_id] = univ
        
    lattice = openmc.RectLattice()
    lattice.lower_left = (-10.0, -6.0, 0.0)
    lattice.pitch = (0.2, 0.2, 0.2)
    
    # Assign universes to lattice cells
    lattice_universes = np.empty((nz, ny, nx), dtype=object)
    for organ_id, univ in universes.items():
        lattice_universes[phantom == organ_id] = univ
        
    # Flip the Y axis to match OpenMC's indexing convention (y=0 is max Y)
    lattice.universes = np.flip(lattice_universes, axis=1).tolist()
    lattice.outer = universes[0] # Default surrounding air
    
    phantom_box_cell.fill = lattice
    geometry = openmc.Geometry(root=[phantom_box_cell])
    geometry.export_to_xml()
    
    # 4. Settings Configuration (Fixed Source Photon Simulation)
    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.batches = 5
    settings.inactive = 0
    settings.particles = particles
    
    # Source: Planar beam at Y=5.9 cm, pointing along -Y (anterior to posterior exposure)
    source = openmc.Source()
    source.space = openmc.stats.Box((-10.0, 5.8, 0.0), (10.0, 5.9, 40.0))
    source.angle = openmc.stats.Monodirectional((0.0, -1.0, 0.0))
    source.energy = openmc.stats.Discrete([energy_kev * 1000.0], [1.0]) # convert keV to eV
    source.particle = 'photon'
    
    settings.source = source
    settings.photon_transport = True
    settings.export_to_xml()
    
    # 5. Tallies
    # Mapped heating score across organ indexes to evaluate energy deposition
    tally_materials = [materials[i] for i in range(1, 9)] # Skeleton, Lungs, Brain, Heart, Liver, Kidneys, Skin, Muscle
    material_filter = openmc.MaterialFilter(tally_materials)
    
    tally = openmc.Tally(name='organ_dose')
    tally.filters = [material_filter]
    tally.scores = ['heating']
    
    # 3D Mesh Tally for Heatmap
    mesh = openmc.RegularMesh()
    mesh.dimension = [50, 30, 100] # nx, ny, nz
    mesh.lower_left = [-10.0, -6.0, 0.0]
    mesh.upper_right = [10.0, 6.0, 40.0]
    mesh_filter = openmc.MeshFilter(mesh)
    
    mesh_tally = openmc.Tally(name='mesh_dose')
    mesh_tally.filters = [mesh_filter]
    mesh_tally.scores = ['heating']
    
    tallies = openmc.Tallies([tally, mesh_tally])
    tallies.export_to_xml()
    
    # 6. Run simulation
    print("Running OpenMC execution...")
    openmc.run()
    
    # 7. Process Results
    sp_filename = 'statepoint.5.h5'
    sp = openmc.StatePoint(sp_filename)
    tally_output = sp.get_tally(name='organ_dose')
    mesh_output = sp.get_tally(name='mesh_dose')
    
    # Save mesh dose array (nz = 100, ny = 30, nx = 50)
    mesh_heating = mesh_output.get_values(scores=['heating'], value='mean').reshape((100, 30, 50))
    np.savez_compressed(os.path.join(base_dir, 'mesh_results.npz'), mesh_heating=mesh_heating)
    
    results = []
    organ_names = ["Skeleton", "Lungs", "Brain", "Heart", "Liver", "Kidneys", "Skin", "Muscle"]
    voxel_volume = 0.2 * 0.2 * 0.2 # 0.008 cm3
    
    for i, name in enumerate(organ_names):
        organ_id = i + 1
        mat = materials[organ_id]
        density = mat.density
        
        # Calculate volume and mass from voxel counts
        voxel_count = int(np.sum(phantom == organ_id))
        volume = voxel_count * voxel_volume
        mass = volume * density
        
        # Query values for this material
        mean = float(tally_output.get_values(filters=[openmc.MaterialFilter], filter_bins=[(mat.id,)], scores=['heating'], value='mean')[0][0][0])
        std_dev = float(tally_output.get_values(filters=[openmc.MaterialFilter], filter_bins=[(mat.id,)], scores=['heating'], value='std_dev')[0][0][0])
        
        results.append({
            "organ_id": organ_id,
            "organ_name": name,
            "voxel_count": voxel_count,
            "volume_cc": volume,
            "density_g_cc": density,
            "mass_g": mass,
            "heating_mean_ev": mean,
            "heating_std_ev": std_dev
        })
        
    out_results_path = os.path.join(base_dir, 'simulation_results.json')
    with open(out_results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Simulation completed! Organ heating tallies saved to: {out_results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--energy', type=float, default=100.0, help='Photon energy in keV')
    parser.add_argument('--particles', type=int, default=10000, help='Number of histories per batch')
    args = parser.parse_args()
    run_simulation(args.energy, args.particles)
