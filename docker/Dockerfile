# docker/Dockerfile
# syntax=docker/dockerfile:1
FROM rocker/r-ver:4.5.2

ARG DEBIAN_FRONTEND=noninteractive
ARG SEURAT_VERSION=5.3.1

# -----------------------------
# 0) OS deps (same as LLM-scCurator)
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    build-essential patch pkg-config cmake gfortran \
    bzip2 xz-utils unzip \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libicu-dev \
    libhdf5-dev zlib1g-dev libbz2-dev liblzma-dev libzstd-dev \
    libfftw3-dev \
    libcairo2-dev libxt-dev libx11-dev libxext-dev libxrender-dev \
    libudunits2-dev libgdal-dev libgeos-dev libproj-dev \
    libglpk-dev libgmp-dev \
    libopenblas-dev liblapack-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev \
    libpng-dev libtiff-dev libjpeg-dev \
    libmagick++-dev libgit2-dev \
    libgl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

ENV RGL_USE_NULL=TRUE
ENV MPLBACKEND=Agg

# Reproducibility-ish knobs (reduce non-determinism from threading)
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTHONHASHSEED=0

RUN echo 'options(timeout = 777, repos=c(CRAN="https://cloud.r-project.org"))' \
  >> /usr/local/lib/R/etc/Rprofile.site

# -----------------------------
# 1) micromamba (same)
# -----------------------------
ENV MAMBA_ROOT_PREFIX=/opt/conda

RUN ARCH="$(uname -m)" && \
    case "$ARCH" in \
      aarch64|arm64) MAMBA_PLATFORM="linux-aarch64" ;; \
      *)             MAMBA_PLATFORM="linux-64" ;; \
    esac && \
    mkdir -p /tmp/micromamba && \
    cd /tmp/micromamba && \
    curl -L "https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/latest" | tar -xj && \
    install -m 0755 ./bin/micromamba /usr/local/bin/micromamba && \
    cd / && rm -rf /tmp/micromamba

RUN micromamba create -y -n py -c conda-forge \
      python=3.12 pip \
      jupyterlab ipykernel \
      numpy pandas scipy \
      scanpy anndata h5py \
      python-igraph leidenalg \
      matplotlib \
      scikit-learn \
      scikit-misc \
      adjusttext \
    && micromamba clean -a -y

ENV RETICULATE_PYTHON=/opt/conda/envs/py/bin/python

RUN micromamba run -n py python -m ipykernel install --sys-prefix \
    --name python3 --display-name "Python (LLM-PathwayCurator)"

# -----------------------------
# 2) R packages (Fig2: DE + GSEA; keep Seurat optional)
# -----------------------------
RUN /usr/local/bin/R --vanilla <<'EOF'
options(repos = c(CRAN="https://cloud.r-project.org"))

install.packages(c(
  "remotes",
  "reticulate",
  "dplyr","tidyr",
  "data.table",
  "ggplot2","patchwork",
  "pheatmap",
  "hdf5r",
  "Matrix","igraph",
  "zip",
  "IRkernel"
))

if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(version="3.22", ask=FALSE)

# Baseline Bioc deps you already used elsewhere (keep if you want sc compatibility)
BiocManager::install(c(
  "BiocGenerics","Biobase",
  "multtest",
  "qvalue",
  "SingleCellExperiment","SummarizedExperiment",
  "SingleR","celldex",
  "zellkonverter",
  "scran","scater","scuttle",
  "DropletUtils","BiocParallel",
  "SpatialExperiment","SpatialExperimentIO"
), ask=FALSE, update=FALSE)

# Fig2 required: DEG + rank-based enrichment
BiocManager::install(c(
  "DESeq2",
  "edgeR",
  "limma",
  "fgsea"
), ask=FALSE, update=FALSE)

# MSigDB access (CRAN)
install.packages(c("msigdbr"))

# Small utils
install.packages(c("qqconf","mutoss","metap"))

q(save="no")
EOF

# Optional: Seurat (parity with your template; safe to keep)
RUN /usr/local/bin/R -q -e ' \
  options(repos=c(CRAN="https://cloud.r-project.org")); \
  if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes"); \
  remotes::install_version("Seurat", version=Sys.getenv("SEURAT_VERSION", unset="5.3.1"), dependencies=NA, repos="https://cloud.r-project.org")' \
  && true

# IRkernel visible in Jupyter
RUN PATH=/opt/conda/envs/py/bin:$PATH /usr/local/bin/R -q -e \
  'IRkernel::installspec(user=FALSE, prefix="/opt/conda/envs/py")'

# -----------------------------
# 3) Python deps (docker/requirements.txt) + your package
# -----------------------------
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV R_HOME=/usr/local/lib/R

WORKDIR /opt/LLM-PathwayCurator
COPY . /opt/LLM-PathwayCurator

# Install Python deps for paper/bench + backends
RUN micromamba run -n py python -m pip install -U pip \
 && micromamba run -n py python -m pip install -r docker/requirements.txt

# Install your package (core deps come from pyproject.toml)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN micromamba run -n py python -m pip install -vvv -e "."

# -----------------------------
# 4) Sanity checks (light)
# -----------------------------
RUN /usr/local/bin/R -q -e ' \
  library(reticulate); \
  library(DESeq2); library(edgeR); library(limma); library(fgsea); library(msigdbr); \
  TRUE' \
 && micromamba run -n py python -c \
 "import pandas as pd; import numpy as np; import scipy; import synapseclient; import yaml; print('OK')"

ENV PATH=/opt/conda/envs/py/bin:$PATH

# -----------------------------
# 5) Jupyter
# -----------------------------
WORKDIR /work
EXPOSE 8888
CMD ["/bin/sh","-lc","micromamba run -n py jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"]
