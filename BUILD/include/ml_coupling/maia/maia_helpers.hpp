#include <iostream>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

namespace GridHelpers {
    void computeSmoothness(const double* field, int Nx, int Ny, int Nz);

    void computeHistogram(const double* field, size_t size, int num_bins); 

    void computeStats(const double* field, size_t size);

    void voxelcluster(const double* field, int Nx, int Ny, int Nz);

    std::vector<double> compute_laplacian(const double* field, int Nx, int Ny, int Nz);

    void smoothField(double* field, int Nx, int Ny, int Nz, int kernel_size);
    
    void computeGaussianKernel(std::vector<double>& kernel, int kernel_size, double sigma);

    void gaussianSmooth1D(double* in, double* out, int Nx, int Ny, int Nz, const std::vector<double>& kernel, int axis);

    void smoothFieldGaussian(double* field, int Nx, int Ny, int Nz, int kernel_size, double sigma);
}