FROM --platform=linux/amd64 condaforge/miniforge3:latest

# Set working directory
WORKDIR /app

# Copy all codebase files into the container
COPY . /app

# Install openmc and other dependencies using mamba (pre-installed and fast)
RUN mamba install -y \
    python=3.11 \
    openmc \
    streamlit \
    numpy \
    matplotlib \
    pandas \
    && conda clean --all --yes

# Install openmc_data_downloader and download cross-sections for target elements (both neutron and photon)
RUN pip install --no-cache-dir openmc_data_downloader \
    && openmc_data_downloader -l ENDFB-7.1-NNDC -e H C N O Na Mg P S Cl K Ca Fe Ar -d /app/nndc_data -p neutron photon

# Set environment paths to point to the downloaded cross-sections
ENV OPENMC_CROSS_SECTIONS="/app/nndc_data/cross_sections.xml"

# Default port, Render/Railway will override this using the PORT env variable
EXPOSE 8501

# Run streamlit app dynamically binding to PORT
CMD streamlit run dashboard.py --server.port ${PORT:-8501} --server.address 0.0.0.0
