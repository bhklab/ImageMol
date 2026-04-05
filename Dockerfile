# ==========================================
# Build Stage
# ==========================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget bzip2 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Create the environment and install everything
ENV PATH=/opt/conda/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -y -n imagemol python=3.10 && \
    conda install -y -n imagemol -c conda-forge rdkit
RUN /opt/conda/envs/imagemol/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    /opt/conda/envs/imagemol/bin/pip install --no-build-isolation torch-cluster torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.9.1+cpu.html && \
    /opt/conda/envs/imagemol/bin/pip install google-cloud-storage && \
    conda clean -afy


# Install the rest of the dependencies from requirements.txt into the conda environment
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/imagemol/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# clean conda cache
RUN conda clean -afy

# ==========================================
# Runtime Stage
# ==========================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire conda environment from the builder stage
COPY --from=builder /opt/conda /opt/conda

# Set environment paths
ENV PATH=/opt/conda/envs/imagemol/bin:/opt/conda/bin:$PATH

WORKDIR /workspace

# Copy over training code
COPY . /workspace

# Start the container with the finetuning script
ENTRYPOINT ["python", "finetune.py"]

# Start the container with a config file if desired
# ENTRYPOINT ["python", "finetune.py", "--config", "hyperparam_config.yaml"]