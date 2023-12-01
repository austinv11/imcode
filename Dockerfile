FROM gpuci/miniforge-cuda:11.8-devel-ubuntu20.04
# Build a dev environment for the project

# Use mamba solver
RUN conda update -y -n base conda && conda install -y -n base -c conda-forge conda-libmamba-solver zstandard && conda config --set solver libmamba

ENV LD_LIBRARY_PATH=/opt/conda/lib/:$LD_LIBRARY_PATH
ENV TZ="America/New_York"

COPY environment.yml .
RUN conda env update -n base --file environment.yml

# Ensure we have gpu-capable pytorch
RUN conda install -c pytorch -c nvidia -y -n base pytorch torchvision torchaudio pytorch-cuda=11.8 cuda-nvcc

USER root

SHELL ["/bin/bash", "-c"]

RUN conda init bash

# Wrapper for python to automatically activate the conda environment
RUN echo "#!/bin/bash --login" > /usr/bin/python-conda && \
    echo "conda activate base" >> /usr/bin/python-conda && \
    echo "exec python \$@" >> /usr/bin/python-conda && \
#    echo "conda run --live-stream -n base python \$@" >> /usr/bin/python-conda && \
    chmod +x /usr/bin/python-conda

# Copy files to the container
#COPY . /workspace
#WORKDIR /workspace

ENTRYPOINT ["python-conda"]