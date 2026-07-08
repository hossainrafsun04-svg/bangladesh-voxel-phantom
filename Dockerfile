FROM mambaorg/micromamba:1.5.8-ubuntu22.04

# Copy all codebase files into the container
COPY --chown=micromamba:micromamba . /app
WORKDIR /app

# Install openmc and dependencies using micromamba (100x faster than conda)
RUN micromamba install -y -n base -c conda-forge \
    openmc \
    streamlit \
    numpy \
    matplotlib \
    pandas \
    && micromamba clean --all --yes

# Set environment path for base environment
ENV PATH="/opt/conda/bin:${PATH}"
ENV OPENMC_CROSS_SECTIONS="/opt/conda/share/openmc/cross_sections.xml"

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

# Command to run the dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
