FROM mambaorg/micromamba:jammy

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

# Default port, Render will automatically override this using the PORT env variable
EXPOSE 8501

# Run streamlit app dynamically binding to PORT
CMD streamlit run dashboard.py --server.port ${PORT:-8501} --server.address 0.0.0.0
