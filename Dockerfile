FROM condaforge/miniforge3:latest

# Set working directory
WORKDIR /app

# Copy all codebase files into the container
COPY . /app

# Install openmc and other dependencies using mamba (pre-installed and fast)
RUN mamba install -y \
    openmc \
    streamlit \
    numpy \
    matplotlib \
    pandas \
    && conda clean --all --yes

# Set environment paths
ENV OPENMC_CROSS_SECTIONS="/opt/conda/share/openmc/cross_sections.xml"

# Default port, Render/Railway will override this using the PORT env variable
EXPOSE 8501

# Run streamlit app dynamically binding to PORT
CMD streamlit run dashboard.py --server.port ${PORT:-8501} --server.address 0.0.0.0
