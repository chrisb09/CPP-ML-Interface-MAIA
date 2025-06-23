#!/usr/local_rwth/bin/zsh

module purge
module --ignore_cache load foss/2024a          # GCC/11.3.0 >= 9.1 
                                # OpenMPI 4.1.4 >= 4.0
module --ignore_cache load FFTW.MPI/3.3.10     # >= 3.3.2

module --ignore_cache load PnetCDF/1.14.0      # >= 1.9
module --ignore_cache load HDF5/1.14.5
#module --ignore_cache load Szip
module --ignore_cache load Python/3.12.3
module --ignore_cache load CMake/4.0.2
module --ignore_cache load imkl

module --ignore_cache load Score-P/8.4
#-CUDA-12.6.3

# for AIxeleratorService
#module load CUDA/11.8.0
module --ignore_cache load CUDA/12.6.3
#module load cuDNN/8.6.0.163-CUDA-11.8.0
module --ignore_cache load cuDNN/9.7.0.66-CUDA-12.6.3
# with LibTorch Version 2.1.0-cuda-11.8

export PYTHONPATH=$PYTHONPATH:/work/thes1961/ai4hpc
