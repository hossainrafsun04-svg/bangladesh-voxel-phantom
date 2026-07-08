# Bangladeshi Voxel Anthropomorphic Phantom & Simulation Dashboard

A complete, end-to-end framework to build, scale, and run a customized Bangladeshi Voxel Anthropomorphic Phantom using the OpenMC Monte Carlo simulator.

---

## 📂 Codebase File Structure

*   `demographics.json`: Stores target Bangladeshi adult demographics (Height: 165 cm, Weight: 60 kg) vs. standard ICRP reference male profiles.
*   `scale_generator.py`: Calculates lateral and vertical spatial scaling ratios ($S_{xy} \approx 0.9363$ and $S_z \approx 0.9375$).
*   `phantom_generator.py`: Generates the 3D voxel array (200 x 60 x 100 lattice, 0.2 cm resolution) representing 8 major organs. Saves to `phantom.npz`.
*   `openmc_simulation.py`: Builds materials with ICRP/ICRU-116 elemental fractions and densities, creates OpenMC XML configuration architecture, maps the voxel grid to a `RectLattice`, and runs photon transport.
*   `run_energy_sweep.py`: Automates running simulations across 6 standard energy levels (10 keV to 1000 keV) to evaluate dosimetry vs. uncertainty.
*   `generate_heatmap.py`: Standalone script to plot the spatial dose distribution heatmap overlaid on the phantom anatomy.
*   `dashboard.py`: Interactive Streamlit web interface displaying cross-sections, running simulations, showing size comparison tables, and plotting 2D/3D heatmaps.

---

## ⚙️ Prerequisites & Setup

This application runs within a Python environment containing OpenMC and Streamlit.

### 1. Environment Activation
```bash
conda activate openmc-env
```

### 2. Launching the Web Dashboard
```bash
streamlit run dashboard.py --server.port 8501
```
Open **http://localhost:8501** in your web browser.

---

## 🔬 Physics Summary

Due to demographic scaling, customized Bangladeshi organs are volumetric scaled down by a factor of **0.822** relative to Western-based ICRP reference phantoms. Under the same absolute radiation exposure, this mass reduction yields a **21.7% increase in absorbed specific dose** ($D = E/m$) for the target population, demonstrating the necessity of size-specific clinical dose planning (SSDE).

---

## ☁️ Permanent Free Cloud Deployment (Streamlit Community Cloud)

To make the website run **permanently** with **exclusive administrative access for you** (while allowing other people to use the app whenever they want), deploy the app directly to Streamlit Community Cloud:

### 1. Push Code to GitHub
1. Create a new repository on your GitHub account (e.g., `bangladesh-voxel-phantom`).
2. Push all the codebase files (including the newly created `environment.yml` and `requirements.txt`) to your repository.
> [!NOTE]
> Since you own the GitHub repository, **only you** have the permission to modify the code, edit settings, or update the phantom calculations. Other users will have read-only execution access through the website.

### 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository (`bangladesh-voxel-phantom`), branch (`main`), and main file path (`dashboard.py`).
4. Click **"Deploy"**.

Streamlit Cloud will read the `environment.yml` file, configure the Conda environment, install `openmc` from Conda-Forge, and host your application permanently on a custom sub-domain (e.g., `https://bangladesh-voxel-phantom.streamlit.app`) for free.

