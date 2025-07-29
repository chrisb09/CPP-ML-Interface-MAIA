#!/usr/local_rwth/bin/zsh

module purge

module load foss/2024a
module load FFTW.MPI/3.3.10
module load PnetCDF/1.14.0
module load HDF5/1.14.5
module load Python/3.12.3
module load CMake/4.0.2
module load imkl/2024.2.0

module load CUDA/12.6.3
module load cuDNN/9.7.0.66-CUDA-12.6.3

module load Score-P/9.0-CUDA-12.6.3