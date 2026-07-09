import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import os
import subprocess
import time
import pandas as pd
import sys

# Set page design
st.set_page_config(
    page_title="Bangladeshi Voxel Phantom Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (premium light mode with black/dark charcoal text)
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .sidebar .sidebar-content {
        background-color: #f1f5f9;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        color: #0f172a;
    }
    h1, h2, h3 {
        color: #0f172a !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    /* Explicitly force all paragraph, list, label, and markdown text to be dark slate/black */
    .stMarkdown, p, span, label, li, div {
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Workspace Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMOGRAPHICS_PATH = os.path.join(BASE_DIR, "demographics.json")
RATIOS_PATH = os.path.join(BASE_DIR, "scaling_ratios.json")
PHANTOM_PATH = os.path.join(BASE_DIR, "phantom.npz")
RESULTS_PATH = os.path.join(BASE_DIR, "simulation_results.json")
MESH_PATH = os.path.join(BASE_DIR, "mesh_results.npz")
SWEEP_PATH = os.path.join(BASE_DIR, "energy_sweep_results.json")

# Ensure demographics exists
if not os.path.exists(DEMOGRAPHICS_PATH):
    default_demo = {
        "bangladesh": {"height_cm": 165.0, "weight_kg": 60.0},
        "icrp_reference_male": {"height_cm": 176.0, "weight_kg": 73.0}
    }
    with open(DEMOGRAPHICS_PATH, 'w') as f:
        json.dump(default_demo, f, indent=2)

# Load data helper functions
def load_scaling_ratios():
    if os.path.exists(RATIOS_PATH):
        with open(RATIOS_PATH, 'r') as f:
            return json.load(f)
    return {"scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0, "weight_ratio": 1.0}

def load_phantom():
    if os.path.exists(PHANTOM_PATH):
        data = np.load(PHANTOM_PATH)
        return data['phantom']
    return None

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            return json.load(f)
    return None

def load_mesh_results():
    if os.path.exists(MESH_PATH):
        data = np.load(MESH_PATH)
        return data['mesh_heating']
    return None

def load_sweep_results():
    if os.path.exists(SWEEP_PATH):
        with open(SWEEP_PATH, 'r') as f:
            return json.load(f)
    return None

# Custom Colormap for Phantom Visualization
organ_colors = [
    '#0b0f19', # Air (very dark slate)
    '#f1f5f9', # Skeleton (white/grey)
    '#38bdf8', # Lungs (pale sky blue)
    '#c084fc', # Brain (lavender purple)
    '#f43f5e', # Heart (rose red)
    '#b45309', # Liver (mahogany amber)
    '#f59e0b', # Kidneys (gold/orange)
    '#10b981', # Skin (emerald green boundary)
    '#fda4af'  # Muscle (soft pink)
]
organ_names = ["Air", "Skeleton", "Lungs", "Brain", "Heart", "Liver", "Kidneys", "Skin", "Muscle"]
cmap = mcolors.ListedColormap(organ_colors)

# Dashboard Layout
st.title("🇧🇩 Bangladeshi Voxel Anthropomorphic Phantom")
st.subheader("Monte Carlo Simulation & Radiation Dosimetry Dashboard (OpenMC)")

# User Guide & Medical Physics Comments
with st.expander("📖 User Guide & Medical Physics Comments (Click to Expand)", expanded=False):
    st.markdown("""
    ### 🔬 Scientific & Clinical Context
    *   **Custom Demographic Scaling**: Standard reference phantoms (like the ICRP Reference Man) represent Western averages (176 cm, 73 kg). This pipeline scales the coordinates of the 3D voxel array to match custom demographics (default: 165 cm, 60 kg for the Bangladeshi population profile).
    *   **Size-Specific Dose Enhancement (SSDE)**: Because absorbed dose is defined as energy deposited per unit mass ($D = E/m$), smaller organs absorb a higher specific dose. For the average Bangladeshi patient, this leads to a **~21.7% increase** in absorbed specific dose relative to ICRP standards for the same radiation exposure.
    *   **Simulation Engine**: The simulation runs the OpenMC Monte Carlo transport code with a planar photon beam entering from the anterior face ($Y = 5.9$ cm) and moving along the $-Y$ direction.
    
    ### 🛠️ How to Use this Dashboard
    1.  **Adjust Demographics**: Use the sidebar sliders to change the target height and weight. The phantom geometry will automatically recalculate and redraw.
    2.  **Inspect Anatomy**: In the **Phantom Geometry** tab, drag the Z, Y, and X slice sliders to inspect the 2D cross-sections of the 8 major organs (Skeleton, Lungs, Brain, Heart, Liver, Kidneys, Skin, and Muscle).
    3.  **Run Simulation**: In the **Dosimetry Run & Comparison** tab, set the incident photon energy (in keV) and particle history count, then click **Run Simulation Pipeline**. The log will stream in real-time. Once complete, you will see a size comparison bar chart and table against ICRP baselines, along with the **Root Mean Square Error (RMSE)**.
    4.  **Visualize Heatmaps**: In the **Dose Heatmaps** tab, look at the spatial dose profile and overlay heatmaps to see the beam penetration and attenuation.
    5.  **Energy Sweep**: In the **Energy Sweep Heatmaps** tab, run a sweep to see how organ doses and uncertainties change across standard diagnostic energy levels.
    """)

# Sidebar Configuration
st.sidebar.header("Demographics & Scaling Configuration")
with open(DEMOGRAPHICS_PATH, 'r') as f:
    demo_data = json.load(f)

# Sidebar inputs
height = st.sidebar.slider("Target Bangladeshi Height (cm)", 150.0, 185.0, float(demo_data['bangladesh']['height_cm']), step=0.5)
weight = st.sidebar.slider("Target Bangladeshi Weight (kg)", 45.0, 95.0, float(demo_data['bangladesh']['weight_kg']), step=0.5)

# Trigger scale and phantom regeneration if parameters changed
if (height != demo_data['bangladesh']['height_cm']) or (weight != demo_data['bangladesh']['weight_kg']):
    demo_data['bangladesh']['height_cm'] = height
    demo_data['bangladesh']['weight_kg'] = weight
    with open(DEMOGRAPHICS_PATH, 'w') as f:
        json.dump(demo_data, f, indent=2)
        
    # Re-generate scaling ratios
    subprocess.run([sys.executable, "scale_generator.py"], cwd=BASE_DIR)
    # Re-generate phantom geometry
    subprocess.run([sys.executable, "phantom_generator.py"], cwd=BASE_DIR)
    st.sidebar.success("Phantom model updated and rescaled successfully!")

# Load current scaling factors
ratios = load_scaling_ratios()
st.sidebar.markdown("### Spatial Scaling Ratios")
st.sidebar.markdown(f"**Lateral ($S_{{xy}}$):** `{ratios['scale_x']:.4f}`")
st.sidebar.markdown(f"**Height ($S_{{z}}$):** `{ratios['scale_z']:.4f}`")
st.sidebar.markdown(f"**Volume / Mass Ratio:** `{ratios.get('weight_ratio', height*weight/(176.0*73.0)):.4f}`")

# Main Page - Tabs
tab_geometry, tab_simulation, tab_heatmap, tab_energy_sweep = st.tabs(["🩻 Phantom Geometry", "⚛️ Dosimetry Run & Comparison", "🔥 Dose Heatmaps", "📈 Energy Sweep Heatmaps"])

with tab_geometry:
    st.markdown("### 🩻 Anatomical Voxel Phantom Preview")
    phantom = load_phantom()
    
    if phantom is not None:
        nz, ny, nx = phantom.shape
        st.write(f"Grid Size: **{nx} x {ny} x {nz}** (0.2 cm resolution, total **{nx*ny*nz:,}** voxels)")
        
        # Slices Sliders
        transverse_idx = st.slider("Transverse Slice View (Z-axis)", 0, nz-1, int(nz * 0.55))
        coronal_idx = st.slider("Coronal Slice View (Y-axis)", 0, ny-1, int(ny * 0.5))
        sagittal_idx = st.slider("Sagittal Slice View (X-axis)", 0, nx-1, int(nx * 0.5))
        
        # Plotting cross-sections
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='#0f172a')
        
        # Transverse
        axes[0].imshow(phantom[transverse_idx, :, :], cmap=cmap, origin='lower', vmin=0, vmax=8)
        axes[0].set_title(f"Transverse (Z={transverse_idx})", color='white')
        axes[0].set_xlabel("X (Width)", color='white')
        axes[0].set_ylabel("Y (Depth)", color='white')
        axes[0].tick_params(colors='white')
        axes[0].axis('off')
        
        # Coronal
        axes[1].imshow(phantom[:, coronal_idx, :], cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=8)
        axes[1].set_title(f"Coronal (Y={coronal_idx})", color='white')
        axes[1].set_xlabel("X (Width)", color='white')
        axes[1].set_ylabel("Z (Height)", color='white')
        axes[1].tick_params(colors='white')
        axes[1].axis('off')
        
        # Sagittal
        axes[2].imshow(phantom[:, :, sagittal_idx], cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=8)
        axes[2].set_title(f"Sagittal (X={sagittal_idx})", color='white')
        axes[2].set_xlabel("Y (Depth)", color='white')
        axes[2].set_ylabel("Z (Height)", color='white')
        axes[2].tick_params(colors='white')
        axes[2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Legend
        st.markdown("#### Tissue Legend")
        legend_cols = st.columns(len(organ_names))
        for idx, name in enumerate(organ_names):
            legend_cols[idx].markdown(
                f"<div style='display: flex; align-items: center; gap: 0.4rem;'>"
                f"<div style='width: 15px; height: 15px; background-color: {organ_colors[idx]}; border-radius: 3px; border: 1px solid #ffffffaa;'></div>"
                f"<span style='font-size: 0.85rem;'>{name}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.error("Voxel phantom array not loaded. Please generate it first.")

with tab_simulation:
    st.markdown("### ⚛️ Monte Carlo Run Controls")
    
    # Simulation Parameters
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        energy_kev = st.number_input("Photon Energy (keV)", 10.0, 1000.0, 100.0, step=10.0, key="sim_energy")
    with sim_col2:
        particles = st.number_input("Particle Count (Histories)", 1000, 100000, 10000, step=1000, key="sim_histories")
        
    st.markdown("---")
    
    # Run Button
    run_btn = st.button("Run Simulation Pipeline", key="run_sim_btn")
    log_area = st.empty()
    
    if run_btn:
        st.info("Initializing OpenMC validation pipeline...")
        log_text = ""
        
        # Launch subprocess pointing to the OpenMC python executable
        cmd = [
            sys.executable, 
            "openmc_simulation.py", 
            "--energy", str(energy_kev), 
            "--particles", str(particles)
        ]
        
        process = subprocess.Popen(
            cmd, 
            cwd=BASE_DIR, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        
        # Stream the logs in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                log_text += line
                log_area.code(log_text[-3000:], language="bash")
                time.sleep(0.01)
                
        rc = process.poll()
        if rc == 0:
            st.success("Simulation finished successfully!")
        else:
            st.error(f"Simulation failed with exit code: {rc}")
            
    # Load and Render Results
    results = load_results()
    if results is not None:
        st.markdown("### 📊 Size-Specific Dosimetry Analysis & ICRP Comparison")
        
        df = pd.DataFrame(results)
        
        # Calculate specific energy deposition / absorbed dose (eV / gram)
        # Note: Absorbed Dose = Energy / Mass
        df['dose_mean_ev_g'] = df['heating_mean_ev'] / df['mass_g']
        df['dose_std_ev_g'] = df['heating_std_ev'] / df['mass_g']
        
        # Generate theoretical baseline comparison with standard unscaled ICRP Reference Male
        # Because ICRP is larger, its organ mass is scaled up by the inverse of our volume scale:
        vol_scale = ratios['scale_x']**2 * ratios['scale_z']
        df['icrp_mass_g'] = df['mass_g'] / vol_scale
        df['icrp_volume_cc'] = df['volume_cc'] / vol_scale
        
        # Theoretical absorbed dose for reference male assuming same overall energy deposition profile:
        # Since ICRP mass is higher, the specific dose (eV/g) is lower!
        df['icrp_dose_mean_ev_g'] = df['heating_mean_ev'] / df['icrp_mass_g']
        df['dose_increase_percent'] = ((df['dose_mean_ev_g'] - df['icrp_dose_mean_ev_g']) / df['icrp_dose_mean_ev_g'].replace(0, np.nan)) * 100
        df['dose_increase_percent'] = df['dose_increase_percent'].fillna(0)
        
        # Render Comparison chart
        fig2, ax2 = plt.subplots(figsize=(10, 4.5), facecolor='#0f172a')
        ax2.set_facecolor('#1e293b')
        
        x_indices = np.arange(len(df['organ_name']))
        width = 0.35
        
        ax2.bar(x_indices - width/2, df['dose_mean_ev_g'], width, label='Bangladeshi Phantom (Customized)', color='#10b981', edgecolor='white')
        ax2.bar(x_indices + width/2, df['icrp_dose_mean_ev_g'], width, label='ICRP Reference Male (Standard)', color='#64748b', edgecolor='white')
        
        ax2.set_ylabel("Specific Absorbed Dose (eV / g per source photon)", color='white')
        ax2.set_title("Size-Specific Absorbed Dose: Bangladeshi vs. ICRP Standard Reference Male", color='white')
        ax2.set_xticks(x_indices)
        ax2.set_xticklabels(df['organ_name'], rotation=25)
        ax2.tick_params(colors='white')
        ax2.legend()
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3, color='white')
        ax2.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Scientific narrative explaining accuracy
        st.markdown(f"""
        > [!NOTE]
        > **Dosimetric Accuracy & Size Effects**: 
        > The custom Bangladeshi Phantom uses demographics scaled to a height of `{height} cm` and weight of `{weight} kg`. 
        > The physical volume of all internal organs is scaled down by a factor of **{vol_scale:.3f}** relative to the standard ICRP Male Reference.
        > Because absorbed dose is defined as energy deposited per unit mass ($D = E/m$), the smaller organ masses in the Bangladeshi population profile lead to a **{(1/vol_scale - 1)*100:.1f}% increase** in the average absorbed dose relative to standard international phantoms for the same energy deposition profile.
        """)
        
        # Raw Data Table
        st.markdown("#### Tabular Dosimetry & Size Comparison Report")
        comparison_df = pd.DataFrame({
            'Organ': df['organ_name'],
            'BD Volume (cm³)': df['volume_cc'].round(2),
            'ICRP Volume (cm³)': df['icrp_volume_cc'].round(2),
            'BD Mass (g)': df['mass_g'].round(2),
            'ICRP Mass (g)': df['icrp_mass_g'].round(2),
            'BD Dose (eV/g)': df['dose_mean_ev_g'].round(2),
            'ICRP Dose (eV/g)': df['icrp_dose_mean_ev_g'].round(2),
            'Dose Difference (%)': df['dose_increase_percent'].round(1).apply(lambda val: f"+{val}%" if val > 0 else "0.0%")
        })
        st.dataframe(comparison_df, use_container_width=True)
        
        # Calculate RMSE (Root Mean Square Error) of absorbed dose
        diff_sq = (df['dose_mean_ev_g'] - df['icrp_dose_mean_ev_g'])**2
        rmse = np.sqrt(diff_sq.mean())
        
        # Calculate Relative RMSE
        rel_diff_sq = (((df['dose_mean_ev_g'] - df['icrp_dose_mean_ev_g']) / df['icrp_dose_mean_ev_g'].replace(0, np.nan)).fillna(0))**2
        rrmse = np.sqrt(rel_diff_sq.mean()) * 100
        
        st.markdown("### 📈 Statistical Deviation (RMSE Analysis)")
        rms_col1, rms_col2 = st.columns(2)
        with rms_col1:
            st.metric(label="Root Mean Square Error (RMSE)", value=f"{rmse:.3f} eV/g")
        with rms_col2:
            st.metric(label="Relative RMSE (R-RMSE)", value=f"{rrmse:.2f}%")
    else:
        st.info("No simulation results found. Please run the simulation to compile results.")

with tab_heatmap:
    st.markdown("### 🔥 2D Radiation Dose Heatmap Overlay")
    
    phantom = load_phantom()
    mesh_heating = load_mesh_results()
    
    if phantom is not None and mesh_heating is not None:
        nz_p, ny_p, nx_p = phantom.shape
        nz_m, ny_m, nx_m = mesh_heating.shape
        
        # Heatmap slice selector (mapped to mesh coordinates)
        slice_z = st.slider("Select Transverse Height slice (cm)", 0.0, 40.0, 22.0, step=0.4)
        
        # Map slice height to phantom and mesh indices
        idx_p = int((slice_z / 40.0) * nz_p)
        idx_p = min(max(0, idx_p), nz_p - 1)
        
        idx_m = int((slice_z / 40.0) * nz_m)
        idx_m = min(max(0, idx_m), nz_m - 1)
        
        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0f172a')
        
        # Physical coordinates box
        extent = [-10.0, 10.0, -6.0, 6.0]
        
        # Left plot: Pure Dose Heatmap
        ax_heat = axes3[0]
        ax_heat.set_facecolor('#0f172a')
        # Log scale norm for visualization since radiation field decay is exponential
        dose_slice = mesh_heating[idx_m, :, :]
        norm = mcolors.LogNorm(vmin=1e-1, vmax=max(1e1, dose_slice.max()))
        
        im_heat = ax_heat.imshow(dose_slice, cmap='inferno', origin='lower', extent=extent, norm=norm)
        ax_heat.set_title(f"Dose Distribution at Z = {slice_z} cm", color='white')
        ax_heat.set_xlabel("X (cm)", color='white')
        ax_heat.set_ylabel("Y (cm)", color='white')
        ax_heat.tick_params(colors='white')
        
        cbar = fig3.colorbar(im_heat, ax=ax_heat, orientation='vertical')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.tick_params(labelcolor='white')
        cbar.set_label('Energy Deposited (eV / particle)', color='white')
        
        # Right plot: Overlay of Dose Heatmap on Anatomy
        ax_over = axes3[1]
        ax_over.set_facecolor('#0f172a')
        
        # Plot anatomy
        ax_over.imshow(phantom[idx_p, :, :], cmap=cmap, origin='lower', extent=extent, alpha=0.85)
        # Plot dose overlay
        im_over = ax_over.imshow(dose_slice, cmap='jet', origin='lower', extent=extent, norm=norm, alpha=0.45)
        
        ax_over.set_title(f"Anatomy and Dose Overlay (Z = {slice_z} cm)", color='white')
        ax_over.set_xlabel("X (cm)", color='white')
        ax_over.set_ylabel("Y (cm)", color='white')
        ax_over.tick_params(colors='white')
        
        cbar2 = fig3.colorbar(im_over, ax=ax_over, orientation='vertical')
        cbar2.ax.yaxis.set_tick_params(color='white')
        cbar2.ax.tick_params(labelcolor='white')
        cbar2.set_label('Energy Deposited (eV / particle)', color='white')
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        st.markdown("""
        #### Heatmap Interpretation Guide:
        *   **Left Plot (Dose Profile)**: Displays the local energy deposition profile using the `inferno` colormap (bright yellow/white indicates high energy deposition, dark purple/black indicates zero dose).
        *   **Right Plot (Overlay)**: Overlays the logarithmic dose distribution (jet scale) directly on top of the organ anatomy. The monodirectional photon source enters from the top edge ($Y = 6.0$, anterior face) and attenuates as it travels downwards, demonstrating local organ shielding and depth-dose characteristics.
        """)
        
    else:
        st.warning("Please run a simulation on the previous tab to generate the 3D dose mesh results first.")

with tab_energy_sweep:
    st.markdown("### 📈 Energy-Dependent Organ Dosimetry & Uncertainty Heatmaps")
    
    # Sweep controls
    run_sweep_btn = st.button("Run Energy Sweep (6 energy bins)", key="run_sweep_btn")
    sweep_log_area = st.empty()
    
    if run_sweep_btn:
        st.info("Executing background energy sweep simulation pipeline...")
        log_text = ""
        process = subprocess.Popen(
            [sys.executable, "run_energy_sweep.py"], 
            cwd=BASE_DIR, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1
        )
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                log_text += line
                sweep_log_area.code(log_text[-3000:], language="bash")
                time.sleep(0.01)
        rc = process.poll()
        if rc == 0:
            st.success("Energy sweep completed successfully!")
            st.rerun()
        else:
            st.error("Energy sweep failed.")
            
    sweep_data = load_sweep_results()
    if sweep_data is not None:
        st.markdown("#### Organ Doses & Uncertainties across Energy Bins")
        
        organs = list(sweep_data.keys())
        energies = sweep_data[organs[0]]["energies"]
        
        # Build matrices
        dose_matrix = np.zeros((len(organs), len(energies)))
        unc_matrix = np.zeros((len(organs), len(energies)))
        
        for i, organ in enumerate(organs):
            dose_matrix[i, :] = sweep_data[organ]["dose_ev_g"]
            unc_matrix[i, :] = sweep_data[organ]["uncertainty_rel"]
            
        fig4, (ax_dose, ax_unc) = plt.subplots(2, 1, figsize=(10, 10), facecolor='#0f172a')
        
        # Top Plot: Dose Heatmap
        ax_dose.set_facecolor('#1e293b')
        norm_dose = mcolors.LogNorm(vmin=max(1e-2, dose_matrix.min()), vmax=max(1e1, dose_matrix.max()))
        im_dose = ax_dose.imshow(dose_matrix, cmap='viridis', aspect='auto', norm=norm_dose)
        ax_dose.set_title("Organ-Absorbed Dose (eV / g per source photon)", color='white', fontsize=12)
        ax_dose.set_yticks(np.arange(len(organs)))
        ax_dose.set_yticklabels(organs, color='white')
        ax_dose.set_xticks(np.arange(len(energies)))
        ax_dose.set_xticklabels([f"{e} keV" for e in energies], color='white')
        ax_dose.tick_params(colors='white')
        
        cbar_dose = fig4.colorbar(im_dose, ax=ax_dose, orientation='vertical')
        cbar_dose.ax.tick_params(labelcolor='white')
        cbar_dose.set_label('Absorbed Dose (eV/g)', color='white')
        
        # Annotate values
        for i in range(len(organs)):
            for j in range(len(energies)):
                val = dose_matrix[i, j]
                ax_dose.text(j, i, f"{val:.1f}", ha='center', va='center', 
                             color='white' if val < dose_matrix.max()*0.1 else 'black', fontsize=9)
                             
        # Bottom Plot: Uncertainty Heatmap
        ax_unc.set_facecolor('#1e293b')
        im_unc = ax_unc.imshow(unc_matrix, cmap='plasma', aspect='auto')
        ax_unc.set_title("Relative Statistical Uncertainty (U, %)", color='white', fontsize=12)
        ax_unc.set_yticks(np.arange(len(organs)))
        ax_unc.set_yticklabels(organs, color='white')
        ax_unc.set_xticks(np.arange(len(energies)))
        ax_unc.set_xticklabels([f"{e} keV" for e in energies], color='white')
        ax_unc.tick_params(colors='white')
        
        cbar_unc = fig4.colorbar(im_unc, ax=ax_unc, orientation='vertical')
        cbar_unc.ax.tick_params(labelcolor='white')
        cbar_unc.set_label('Relative Uncertainty (%)', color='white')
        
        # Annotate values
        for i in range(len(organs)):
            for j in range(len(energies)):
                val = unc_matrix[i, j]
                ax_unc.text(j, i, f"{val:.1f}%", ha='center', va='center', 
                            color='white' if val < unc_matrix.max()*0.5 else 'black', fontsize=9)
                            
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.markdown("""
        #### Physics Interpretation:
        *   **Energy Dependence**: Note how the energy deposition peaks at lower energies (e.g. 50 keV) due to photoelectric absorption predominance, and decreases at higher energies (e.g. 1000 keV) where Compton scattering dominates and photons are highly penetrating (leaking out of the phantom).
        *   **Relative Uncertainty (U)**: The relative uncertainty is higher in smaller internal organs (like Kidneys, Heart) at low energies because fewer photons reach the interior depth without attenuation, resulting in sparse statistics.
        """)
        
    else:
        st.warning("No energy sweep results found. Click the button above to execute the 6-energy bin simulation sweep (takes ~15 seconds).")
