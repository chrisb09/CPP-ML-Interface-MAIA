#include "ml_coupling/maia/helpers/maia_helpers.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

namespace GridHelpers {
    void computeSmoothness(const double* field, int Nx, int Ny, int Nz) {
        auto index = [=](int x, int y, int z) {
            return z * Ny * Nx + y * Nx + x;
        };

        double sum_grad = 0.0;
        double max_grad = 0.0;
        size_t count = 0;

        for (int z = 1; z < Nz - 1; ++z) {
            for (int y = 1; y < Ny - 1; ++y) {
                for (int x = 1; x < Nx - 1; ++x) {
                    double dx = (field[index(x+1, y, z)] - field[index(x-1, y, z)]) * 0.5;
                    double dy = (field[index(x, y+1, z)] - field[index(x, y-1, z)]) * 0.5;
                    double dz = (field[index(x, y, z+1)] - field[index(x, y, z-1)]) * 0.5;

                    double grad_mag = std::sqrt(dx*dx + dy*dy + dz*dz);
                    sum_grad += grad_mag;
                    if (grad_mag > max_grad) max_grad = grad_mag;
                    count++;
                }
            }
        }

        std::cout << "Gradient magnitude stats:" << std::endl;
        std::cout << "  Avg gradient = " << (sum_grad / count) << std::endl;
        std::cout << "  Max gradient = " << max_grad << std::endl;
    }

    void computeHistogram(const double* field, size_t size, int num_bins = 20) {
        double min_val = *std::min_element(field, field + size);
        double max_val = *std::max_element(field, field + size);
        double bin_width = (max_val - min_val) / num_bins;

        std::vector<int> bins(num_bins, 0);

        for (size_t i = 0; i < size; ++i) {
            int bin = static_cast<int>((field[i] - min_val) / bin_width);
            if (bin == num_bins) bin--; // Handle max edge
            bins[bin]++;
        }

        std::cout << "Histogram:" << std::endl;
        for (int i = 0; i < num_bins; ++i) {
            double bin_start = min_val + i * bin_width;
            double bin_end = bin_start + bin_width;
            std::cout << "[" << bin_start << ", " << bin_end << "): " << bins[i] << std::endl;
        }
    }

    void computeStats(const double* field, size_t size) {
        double sum = 0.0, sq_sum = 0.0;
        double min_val = field[0];
        double max_val = field[0];

        for (size_t i = 0; i < size; ++i) {
            double val = field[i];
            sum += val;
            sq_sum += val * val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        double mean = sum / size;
        double stdev = std::sqrt(sq_sum / size - mean * mean);

        std::cout << "Statistics:" << std::endl;
        std::cout << "  Min   = " << min_val << std::endl;
        std::cout << "  Max   = " << max_val << std::endl;
        std::cout << "  Mean  = " << mean << std::endl;
        std::cout << "  Stddev= " << stdev << std::endl;
    }

    void voxelcluster(const double* field, int Nx, int Ny, int Nz){
        int count_clustered = 0;
        for (int z = 1; z < Nz - 1; ++z) {
            for (int y = 1; y < Ny - 1; ++y) {
                for (int x = 1; x < Nx - 1; ++x) {
                    int idx = z * Ny * Nx + y * Nx + x;
                    if (field[idx] < 0.0) {
                        // Check if neighbors are also negative
                        int neighbors = 0;
                        for (int dz = -1; dz <= 1; ++dz)
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dx = -1; dx <= 1; ++dx)
                                    if (dx || dy || dz) {
                                        int nidx = (z+dz) * Ny * Nx + (y+dy) * Nx + (x+dx);
                                        if (field[nidx] < 0.0) neighbors++;
                                    }
                        if (neighbors >= 6) count_clustered++;
                    }
                }
            }
        }
        std::cout << "Clustered negative voxels (>=6 neighbors): " << count_clustered << std::endl;
    }

    std::vector<double> compute_laplacian(const double* field, int Nx, int Ny, int Nz) {
        std::vector<double> laplacian(Nx * Ny * Nz, 0.0);
        auto idx = [&](int x, int y, int z) {
            return z * Ny * Nx + y * Nx + x;
        };

        for (int z = 1; z < Nz - 1; ++z) {
            for (int y = 1; y < Ny - 1; ++y) {
                for (int x = 1; x < Nx - 1; ++x) {
                    int i = idx(x, y, z);
                    laplacian[i] = 
                        -6.0 * field[i]
                        + field[idx(x+1, y, z)] + field[idx(x-1, y, z)]
                        + field[idx(x, y+1, z)] + field[idx(x, y-1, z)]
                        + field[idx(x, y, z+1)] + field[idx(x, y, z-1)];
                }
            }
        }

        return laplacian;
    }

    void smoothField(double* field, int Nx, int Ny, int Nz, int kernel_size = 3) {
        std::vector<double> temp(field, field + Nx * Ny * Nz);
        auto idx = [&](int x, int y, int z) { return z * Ny * Nx + y * Nx + x; };

        int r = kernel_size / 2;
        for (int z = r; z < Nz - r; ++z) {
            for (int y = r; y < Ny - r; ++y) {
                for (int x = r; x < Nx - r; ++x) {
                    double sum = 0.0;
                    int count = 0;
                    for (int dz = -r; dz <= r; ++dz)
                        for (int dy = -r; dy <= r; ++dy)
                            for (int dx = -r; dx <= r; ++dx) {
                                sum += temp[idx(x + dx, y + dy, z + dz)];
                                count++;
                            }
                    field[idx(x, y, z)] = sum / count;
                }
            }
        }
    }

    void computeGaussianKernel(std::vector<double>& kernel, int kernel_size, double sigma) {
        int r = kernel_size / 2;
        double sum = 0.0;
        kernel.resize(kernel_size);
        for (int i = -r; i <= r; ++i) {
            double val = std::exp(-(i * i) / (2 * sigma * sigma));
            kernel[i + r] = val;
            sum += val;
        }
        // Normalize
        for (auto& v : kernel) v /= sum;
    }

    void gaussianSmooth1D(double* in, double* out, int Nx, int Ny, int Nz,
                        const std::vector<double>& kernel, int axis) {
        int r = kernel.size() / 2;
        auto idx = [&](int x, int y, int z) { return z * Ny * Nx + y * Nx + x; };

        for (int z = 0; z < Nz; ++z) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    double sum = 0.0;
                    for (int k = -r; k <= r; ++k) {
                        int xi = x, yi = y, zi = z;
                        if (axis == 0) xi = std::clamp(x + k, 0, Nx - 1);
                        else if (axis == 1) yi = std::clamp(y + k, 0, Ny - 1);
                        else if (axis == 2) zi = std::clamp(z + k, 0, Nz - 1);

                        sum += in[idx(xi, yi, zi)] * kernel[k + r];
                    }
                    out[idx(x, y, z)] = sum;
                }
            }
        }
    }

    void smoothFieldGaussian(double* field, int Nx, int Ny, int Nz, int kernel_size = 5, double sigma = 1.0) {
        std::vector<double> kernel;
        computeGaussianKernel(kernel, kernel_size, sigma);

        size_t N = Nx * Ny * Nz;
        std::vector<double> temp1(field, field + N);
        std::vector<double> temp2(N);

        // X axis
        gaussianSmooth1D(temp1.data(), temp2.data(), Nx, Ny, Nz, kernel, 0);
        // Y axis
        gaussianSmooth1D(temp2.data(), temp1.data(), Nx, Ny, Nz, kernel, 1);
        // Z axis
        gaussianSmooth1D(temp1.data(), field, Nx, Ny, Nz, kernel, 2);
    }
}


//#include "ml_coupling/maia/ml_coupling_maia.hpp"
//#include "ml_coupling/maia/helpers/maia_helpers.hpp"



   /* void baseSetup(
        const std::vector<int>& nCells,
        const std::vector<int>& nOffsetCells,
        int cubeD,
        const std::vector<int>& activeCells,
        int nFields,
        int nGhostLayers,
        int activeFieldSize,
        int sequenceLen
    ){
        this->nCells = nCells;
        this->nOffsetCells = nOffsetCells;
        this->cubeD = cubeD;
        this->activeCells = activeCells;
        this->nFields = nFields;
        this->nGhostLayers = nGhostLayers;
        this->activeFieldSize = activeFieldSize;
        this->sequenceLen = sequenceLen;

        
        // Compute number of cubes based on interior only
        int concatX = (activeCells[2] + cubeD - 1) / cubeD;
        int concatY = (activeCells[1] + cubeD - 1) / cubeD;
        int concatZ = (activeCells[0] + cubeD - 1) / cubeD;

        // === Inline linspace logic for Z ===
        std::vector<int> zs(concatZ);
        if (concatZ == 1) {
            zs[0] = 0;
        } else {
            double step = static_cast<double>(activeCells[0]  - cubeD) / (concatZ - 1);
            for (int i = 0; i < concatZ; ++i) {
                zs[i] = static_cast<int>(std::round(i * step));
            }
        }
        zs.insert(zs.begin(), 0); // Add origin cube
        //for (int& z : zs) z += nGhostLayers; // offset to interior position in full domain

        // === Inline linspace logic for Y ===
        std::vector<int> ys(concatY);
        if (concatY == 1) {
            ys[0] = 0;
        } else {
            double step = static_cast<double>(activeCells[1] - cubeD) / (concatY - 1);
            for (int i = 0; i < concatY; ++i) {
                ys[i] = static_cast<int>(std::round(i * step));
            }
        }
        ys.insert(ys.begin(), 0); // Add origin cube
        //for (int& y : ys) y += nGhostLayers;

        // === Inline linspace logic for X ===
        std::vector<int> xs(concatX);
        if (concatX == 1) {
            xs[0] = 0;
        } else {
            double step = static_cast<double>(activeCells[2] - cubeD) / (concatX - 1);
            for (int i = 0; i < concatX; ++i) {
                xs[i] = static_cast<int>(std::round(i * step));
            }
        }
        xs.insert(xs.begin(), 0); // Add origin cube
        //for (int& x : xs) x += nGhostLayers;

        // === Count total cubes and calculate total elements ===
        this->num_cubes = zs.size() * ys.size() * xs.size();
        this->cube_volume = cubeD * cubeD * cubeD;
        this->total_elements = this->sequenceLen * this->nFields * num_cubes * cube_volume;
    }*/

/*    

    //Save where the inputs and also outputs are in maia


    coupling_strategy->setup(
        this->nCells, 
        this->nOffsetCells, 
        cubeD,
        nActiveCells,
        nFields, 
        this->nGhostLayers, 
        activeFieldSize,
        sequenceLen
    );
}*/
    