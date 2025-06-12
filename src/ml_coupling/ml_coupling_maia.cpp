#include "ml_coupling_maia.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>



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

// Constructor and destructor
MLCouplingMaia::MLCouplingMaia() = default;

MLCouplingMaia::~MLCouplingMaia() {
    finalize();
}

// Initializes the coupling strategy.
void MLCouplingMaia::init(int strategy_id) {
    coupling_strategy_id = strategy_id;
    #ifdef WITH_AIX
    if (coupling_strategy_id == 1) {
        coupling_strategy = new MLCouplingStrategyAix();
        std::cout << "Created Aix Coupling\n";
    } 
    #endif
    #ifdef WITH_PHYDLL
    if (coupling_strategy_id == 2) {
        coupling_strategy = new MLCouplingStrategyPhyDll();
        std::cout << "Created PhyDLL Coupling\n";
    }
    #endif
    if (!coupling_strategy) {
        std::cerr << "ERROR: Unknown coupling_strategy_id = " << coupling_strategy_id << "!\n";
    }

    coupling_strategy->init();
    
    this->app_comm = coupling_strategy->getComm();
}

void MLCouplingMaia::setup(
        std::vector<double*> input_fields_ptr, 
        std::vector<double*> output_fields_ptr,
        const std::string& modelPath,
        int batchSize,
        const std::vector<int>& nCells,
        const std::vector<int>& nOffsetCells,
        int nGhostLayers
        ) 
{
    nFields = input_fields_ptr.size();
    fieldSize = std::accumulate(nCells.begin(), nCells.end(), 1,  std::multiplies());
    sequenceLen = 5;

    model_path = modelPath;
    batch_size = batchSize;

    this->nCells.resize(3);
    this->nOffsetCells.resize(3);
    this->nCells[0] = nCells[0]; 
    this->nCells[1] = nCells[1];
    this->nCells[2] = nCells[2];
    this->nOffsetCells[0] =  nOffsetCells[0];
    this->nOffsetCells[1] =  nOffsetCells[1];
    this->nOffsetCells[2] =  nOffsetCells[2];     
    this->nGhostLayers = nGhostLayers;
    
    concatX = static_cast<int>(std::ceil(static_cast<double>(this->nCells[2]) / cubeD));
    concatY = static_cast<int>(std::ceil(static_cast<double>(this->nCells[1]) / cubeD));
    concatZ = static_cast<int>(std::ceil(static_cast<double>(this->nCells[1]) / cubeD));
    
    //Save where the inputs and also outputs are in maia
    input_fields.clear();
    output_fields.clear();
    for(size_t i = 0; i < nFields; i++){
        input_fields.push_back(input_fields_ptr[i]);  
        std::cout << "input_fields_ptr ptr " << input_fields_ptr[i] << std::endl;
        std::cout << "input_fields " << input_fields[i] << std::endl;
        
        output_fields.push_back(output_fields_ptr[i]); 
        std::cout << "outputfield ptr " << output_fields_ptr[i] << std::endl;
        std::cout << "outputfield " << output_fields[i] << std::endl;
    }

    coupling_strategy->setup(this->nCells, this->nOffsetCells, nFields, this->nGhostLayers, fieldSize, this->app_comm, sequenceLen);
}

void MLCouplingMaia::preprocess_input(
    std::vector<double*>& input_fields, 
    std::vector<std::vector<double>>& input_fields_pre)
{
    std::vector<std::vector<double>> field_cubes(3); // [field][num_cubes * cubeD³]
    int zSize = nCells[0] - 2 * nGhostLayers;
    int ySize = nCells[1] - 2 * nGhostLayers;
    int xSize = nCells[2] - 2 * nGhostLayers;
    
    double sum = 0.0;
    for (double* vol : input_fields) {
        sum += std::accumulate(vol, vol + xSize * ySize * zSize, 0.0);
    }
    std::cout << "Checksum of input_fields: " << sum << std::endl;

    int concatX = (xSize + cubeD - 1) / cubeD;
    int concatY = (ySize + cubeD - 1) / cubeD;
    int concatZ = (zSize + cubeD - 1) / cubeD;
    std::cout << "input_fields is: " << input_fields.size() << std::endl;
    // Extract cubes for u, v, w
    for (int f = 0; f < 3; ++f) {
        std::vector<double> trimmed_data;
        trimmed_data.reserve(xSize * ySize * zSize);
        // Flattened index access: input_fields[f] is of size nCells[0]*nCells[1]*nCells[2]
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    size_t idx = (z * nCells[1] * nCells[2]) + (y * nCells[2]) + x;
                    trimmed_data.push_back(input_fields[f][idx]);
                }
            }
        }
        auto cubes = extract_cubes(trimmed_data.data(), xSize, ySize, zSize, cubeD, concatX, concatY, concatZ);
        for (const auto& cube : cubes) {
            field_cubes[f].insert(field_cubes[f].end(), cube.begin(), cube.end());
        }
    }

    // Interleave into [num_cubes * 3 * cubeD³]
    size_t num_cubes = field_cubes[0].size() / (cubeD * cubeD * cubeD);
    std::vector<double> flat;
    flat.reserve(num_cubes * 3 * cubeD * cubeD * cubeD);
    for (size_t i = 0; i < num_cubes; ++i) {
        for (int f = 0; f < 3; ++f) {
            flat.insert(flat.end(),
                        field_cubes[f].begin() + i * cubeD * cubeD * cubeD,
                        field_cubes[f].begin() + (i + 1) * cubeD * cubeD * cubeD);
        }
    }

    input_fields_pre.push_back(std::move(flat));
    std::cout << "input_fields_pre is: " << input_fields_pre.size() << std::endl;

}

void MLCouplingMaia::inference(
    std::vector<std::vector<double>>& input_fields_pre, 
    std::vector<double>& output_fields_post)
{
    if (!coupling_strategy) {
        std::cerr << "ERROR: No coupling strategy set!\n";
        return;
    }
    
    coupling_strategy->inference(input_fields_pre, output_fields_post);
}

void MLCouplingMaia::postprocess_output(
    std::vector<double>& output_fields_post,  // Flat, interleaved data (all cubes)
    std::vector<double*>& output_fields)        // Output: three reconstructed full volumes
{
    for(size_t i = 0; i < nFields; i++){
        std::cout << "outputfield " << output_fields[i] << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "output_fields[" << 0 << "] address: " << static_cast<void*>(output_fields[0]) << std::endl;
    std::cout << "Sample from output_fields_post:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), output_fields_post.size()); ++i) {
        std::cout << output_fields_post[i] << " ";
    }
    double sum = std::accumulate(output_fields_post.begin(), output_fields_post.end(), 0.0);
    std::cout << "Checksum of output_fields_post: " << sum << std::endl;
    std::cout << std::endl;
    std::cout << "post process outputfields_post front: " << output_fields_post.front() << std::endl;
    std::cout << "post process outputfields_post back: " << output_fields_post.back() << std::endl;
    std::cout << "post process output_fields front: " << *(output_fields.front()) << std::endl;
    std::cout << "post process output_fields back: " << *(output_fields.back()) << std::endl;
    // === Parameters assumed available as member variables ===
    // cubeD: The cube edge length.
    // nCells: An int array (or vector) with { Nz, Ny, Nx }
    //
    // Note: In extract_cubes we used:
    //   Nx = nCells[2], Ny = nCells[1], Nz = nCells[0]
    
    int NxFull = nCells[2];
    int NyFull = nCells[1];
    int NzFull = nCells[0];

    int Nx = NxFull - 2 * nGhostLayers;
    int Ny = NyFull - 2 * nGhostLayers;
    int Nz = NzFull - 2 * nGhostLayers;

    size_t volumeSize = static_cast<size_t>(Nx * Ny * Nz);
    const int numFields = 3;  // for example, u, v, w

    // ------------------------------------------------------------------------
    // Step 1. Deinterleave the flat vector into 3 separate vectors (one per field)
    // ------------------------------------------------------------------------
    size_t cubeSize = static_cast<size_t>(cubeD * cubeD * cubeD);
    size_t groupSize = numFields * cubeSize; // data for one cube across all fields
    size_t numCubes = output_fields_post.size() / groupSize;

    /*for (int f = 0; f < numFields; ++f)
    {
        computeStats(output_fields[f], volumeSize);
        computeHistogram(output_fields[f], volumeSize, 30);
        computeSmoothness(output_fields[f], Nx, Ny, Nz);
        voxelcluster(output_fields[f], Nx, Ny, Nz);

        std::vector<double> lap = compute_laplacian(output_fields[f], Nx, Ny, Nz);
        double lap_sum = std::accumulate(lap.begin(), lap.end(), 0.0);
        double lap_min = *std::min_element(lap.begin(), lap.end());
        double lap_max = *std::max_element(lap.begin(), lap.end());
        std::cout << "Laplacian stats:\n";
        std::cout << "  Sum:   " << lap_sum << "\n";
        std::cout << "  Min:   " << lap_min << "\n";
        std::cout << "  Max:   " << lap_max << "\n";

        double* vol = output_fields[f];
        int in_neg_count = 0;
        double sum = 0;
        double negsum = 0;
        double zero_count = 0;
        for (size_t i = 0; i < volumeSize; ++i)
        {
            if (vol[i] < 0.0){
                //vol[i] = 0.0;
                in_neg_count++;
                negsum += vol[i];
            }
            if (vol[i] == 0.0){
                zero_count++;
            }
            sum += vol[i];
        }
        std::cout << "Negsum in in " << f << " is: " << negsum << std::endl;
        std::cout << "Checksum of inputfield " << f << " is: " << sum << std::endl;
        std::cout << "number of input negative values in field " << f <<" is: " << in_neg_count << std::endl;
        std::cout << "number of input zero values in field " << f <<" is: " << zero_count << std::endl;
    }*/

    // Create three temporary vectors to store concatenated cube data for each field.
    std::vector<std::vector<double>> field_cubes(numFields);
    for (int f = 0; f < numFields; ++f)
    {
        field_cubes[f].reserve(numCubes * cubeSize);
    }

    // Iterate through each cube (as interleaved groups) and extract each field’s cube.
    for (size_t cubeIndex = 0; cubeIndex < numCubes; ++cubeIndex)
    {
        for (int f = 0; f < numFields; ++f)
        {
            size_t startIndex = cubeIndex * groupSize + f * cubeSize;
            field_cubes[f].insert(
                field_cubes[f].end(),
                output_fields_post.begin() + startIndex,
                output_fields_post.begin() + startIndex + cubeSize
            );
        }
    }
    sum = std::accumulate(field_cubes[0].begin(), field_cubes[0].end(), 0.0);
    std::cout << "Checksum of field_cubes[0]: " << sum << std::endl;
    std::cout << "First 5 values of each field after deinterleaving:\n";
    for (int f = 0; f < numFields; ++f) {
        for (int i = 0; i < 5 && i < field_cubes[f].size(); ++i)
            std::cout << field_cubes[f][i] << " ";
        std::cout << std::endl;
    }
    // ------------------------------------------------------------------------
    // Step 2. Reconstruct full volumes (for each field)
    // ------------------------------------------------------------------------

    // Prepare the weight grid (to count contributions at each voxel)
    std::vector<double> weight(NxFull * NyFull * NzFull, 0.0);

    // Recompute the extraction starting positions (xs, ys, zs)
    //
    // For the extraction we computed coordinates using linspace:
    //   auto xs = linspace(0, Nx - cubeD, concatX);
    //   auto ys = linspace(0, Ny - cubeD, concatY);
    //   auto zs = linspace(0, Nz - cubeD, concatZ);
    //
    // and then inserted a 0 at the beginning of each.
    int concatX = (Nx + cubeD - 1) / cubeD;
    int concatY = (Ny + cubeD - 1) / cubeD;
    int concatZ = (Nz + cubeD - 1) / cubeD;
    auto xs = linspace(0, Nx - cubeD, concatX);  // assumed to return std::vector<int>
    auto ys = linspace(0, Ny - cubeD, concatY);
    auto zs = linspace(0, Nz - cubeD, concatZ);
    xs.insert(xs.begin(), 0);
    ys.insert(ys.begin(), 0);
    zs.insert(zs.begin(), 0);  
    
    //for (int& x : xs) x += nGhostLayers;
    //for (int& y : ys) y += nGhostLayers;
    //for (int& z : zs) z += nGhostLayers;

    // Build the weight grid by “painting” one cube at each starting position.
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // For the current cube, increment the contribution counter for each voxel.
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int global_x = x0 + dx + nGhostLayers;
                            int global_y = y0 + dy + nGhostLayers;
                            int global_z = z0 + dz + nGhostLayers;
                            if (global_x < NxFull && global_y < NyFull && global_z < NzFull) {
                                int vol_index = global_z * (NyFull * NxFull) + global_y * NxFull + global_x;
                                weight[vol_index] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int f = 0; f < numFields; ++f)
    {
        // Use the preallocated pointer from output_fields[i] (already set in setup())
        double* vol = output_fields[f];  // pointer already points to pvariables[i]
        std::fill(vol, vol + (NxFull * NyFull * NzFull), 0.0);
        // Do not push_back again! You're already using it.
    }
    sum = std::accumulate(field_cubes[0].begin(), field_cubes[0].end(), 0.0);
    std::cout << "Checksum of field_cubes[0]: " << sum << std::endl;

    // For each field, “stitch” the cubes back into the full volume.
    // The cubes were extracted (and later deinterleaved) in the same order as defined by
    // iterating over z, then y, then x coordinates given by zs, ys, xs.
    for (int f = 0; f < numFields; ++f)
    {
        size_t cubeIndex = 0;
        // Loop over the cube starting positions in the same order as extraction.
        for (int z0 : zs) {
            for (int y0 : ys) {
                for (int x0 : xs) {
                    // For each cube, loop over its local voxel coordinates.
                    for (int dz = 0; dz < cubeD; ++dz) {
                        for (int dy = 0; dy < cubeD; ++dy) {
                            for (int dx = 0; dx < cubeD; ++dx)
                            {
                                int global_x = x0 + dx + nGhostLayers;
                                int global_y = y0 + dy + nGhostLayers;
                                int global_z = z0 + dz + nGhostLayers;
                                if (global_x < NxFull && global_y < NyFull && global_z < NzFull) {
                                    int vol_index = global_z * (NyFull * NxFull) + global_y * NxFull + global_x;
                                    int cube_offset = dz * cubeD * cubeD + dy * cubeD + dx;
                                    if (vol_index >= NxFull * NyFull * NzFull) {
                                        std::cerr << "ERROR: vol_index out of bounds: " << vol_index << " / " << NxFull * NyFull * NzFull << std::endl;
                                    }
                                    size_t flat_index = cubeIndex * cubeSize + cube_offset;
                                    if (flat_index >= field_cubes[f].size()) {
                                        std::cerr << "ERROR: field_cubes[" << f << "] index out of bounds: " << flat_index << " / " << field_cubes[f].size() << std::endl;
                                    }
                                    static int log_count = 0;
                                    if (log_count < 10 && field_cubes[f][flat_index] != 0.0) {
                                        std::cout << "Writing to output_fields[" << f << "][" << vol_index << "] = " << field_cubes[f][flat_index] << std::endl;
                                        log_count++;
                                    }
                                    // Accumulate the cube’s contribution.
                                    output_fields[f][vol_index] += 
                                        field_cubes[f][cubeIndex * cubeSize + cube_offset];
                                        
                                    static int log_count_1 = 0;
                                    if (log_count_1 < 10 && output_fields[f][vol_index] != 0.0) {
                                        std::cout << "Writing to output_fields[" << f << "][" << vol_index << "] = " << output_fields[f][vol_index] << std::endl;
                                        log_count_1++;
                                    }
                                }
                            }
                        }
                    }
                    ++cubeIndex;
                }
            }
        }
        // Normalize the full volume by dividing each voxel by its contribution count.
        for (size_t i = 0; i < NxFull * NyFull * NzFull; ++i)
        {
            if (weight[i] > 0.0)
                output_fields[f][i] /= weight[i];
        }
    }
    sum = std::accumulate(field_cubes[0].begin(), field_cubes[0].end(), 0.0);
    std::cout << "Checksum of field_cubes[0]: " << sum << std::endl;
    sum = 0.0;
    for (double* vol : output_fields) {
        sum += std::accumulate(vol, vol + NxFull * NyFull * NzFull, 0.0);
    }
    std::cout << "Checksum of output_fields: " << sum << std::endl;
    
    for(size_t i = 0; i < nFields; i++){
        std::cout << "outputfield " << output_fields[i] << std::endl;
    }
    /*
    // TEST #################################
    // Clip negative values to zero
    for (int f = 0; f < numFields; ++f)
    {
        computeStats(output_fields[f], volumeSize);
        computeHistogram(output_fields[f], volumeSize, 30);
        computeSmoothness(output_fields[f], Nx, Ny, Nz);
        voxelcluster(output_fields[f], Nx, Ny, Nz);

        
        std::vector<double> lap = compute_laplacian(output_fields[f], Nx, Ny, Nz);
        double lap_sum = std::accumulate(lap.begin(), lap.end(), 0.0);
        double lap_min = *std::min_element(lap.begin(), lap.end());
        double lap_max = *std::max_element(lap.begin(), lap.end());
        std::cout << "Laplacian stats:\n";
        std::cout << "  Sum:   " << lap_sum << "\n";
        std::cout << "  Min:   " << lap_min << "\n";
        std::cout << "  Max:   " << lap_max << "\n";

        double* vol = output_fields[f];
        int neg_count = 0;
        double sum = 0;
        double negsum = 0;
        double zero_count = 0;
        
        //smoothField(output_fields[f], Nx, Ny, Nz, 7);
        //smoothFieldGaussian(output_fields[f], Nx, Ny, Nz, 3, 2.0); 
        for (size_t i = 0; i < volumeSize; ++i)
        {
            if (vol[i] < 0.0){
                vol[i] = 0.0; //hard clipping
                //vol[i] * 0.1; // Soft clipping
                neg_count++;
                negsum += vol[i];
            }
            if (vol[i] == 0.0){
                zero_count++;
            }
            sum += vol[i];
        }
        smoothField(output_fields[f], Nx, Ny, Nz);
        std::cout << "Negsum pred in " << f << " is: " << negsum << std::endl;
        std::cout << "Checksum of predictedfield " << f << " is: " << sum << std::endl;
        std::cout << "number of predicted negative values in field " << f <<" is: " << neg_count << std::endl;
        std::cout << "number of predicted zero values in field " << f <<" is: " << zero_count << std::endl;


        std::cout << "SMOOTHING FIELD NOW :\n";
        for (size_t i = 0; i < volumeSize; ++i)//Small bias add
        {
            vol[i] += 0.00000001;
        }
        lap = compute_laplacian(output_fields[f], Nx, Ny, Nz);
        lap_sum = std::accumulate(lap.begin(), lap.end(), 0.0);
        lap_min = *std::min_element(lap.begin(), lap.end());
        lap_max = *std::max_element(lap.begin(), lap.end());
        std::cout << "Laplacian stats:\n";
        std::cout << "  Sum:   " << lap_sum << "\n";
        std::cout << "  Min:   " << lap_min << "\n";
        std::cout << "  Max:   " << lap_max << "\n";
        std::cout << "------------------------------------------------------------------ :\n";
    }
   // TEST #################################*/
    std::cout << "post process output_fields front: " << *(output_fields.front()) << std::endl;
    std::cout << "post process output_fields back: " << *(output_fields.back()) << std::endl;
}

void MLCouplingMaia::finalize() {
    if (coupling_strategy) {
        coupling_strategy->finalize();
        delete coupling_strategy;
        coupling_strategy = nullptr;
    }

    // free the base-class pointers if allocated
    /*if (input_fields_pre) {
        delete[] input_fields_pre;
        input_fields_pre = nullptr;
    }
    if (output_fields_post) {
        delete[] output_fields_post;
        output_fields_post = nullptr;
    }*/
}

MPI_Comm MLCouplingMaia::getComm(){
    return coupling_strategy->getComm();
}

void MLCouplingMaia::ml_step(){
    //With this we ensure that we gather seqLen (4) timesteps and only then infer
    preprocess_input(input_fields, input_fields_pre);   
    if (iter < sequenceLen-1) {
        iter++;
    }else{
        inference(input_fields_pre, output_fields_post);
        postprocess_output(output_fields_post, output_fields);
        iter = 0;
        
        input_fields_pre.clear();//Empty buffer
        output_fields_post.clear();
    }
}


std::vector<int> MLCouplingMaia::linspace(int start, int end, int count) {
    std::vector<int> result(count);
    double step = (end - start) / static_cast<double>(std::max(count - 1, 1));
    for (int i = 0; i < count; ++i) {
        result[i] = static_cast<int>(start + std::round(i * step));
    }
    return result;
}

std::vector<std::vector<double>> MLCouplingMaia::extract_cubes(
    const double* data, int Nx, int Ny, int Nz, int cubeD, int concatX, int concatY, int concatZ)
{
    std::vector<std::vector<double>> cubes;
    auto xs = linspace(0, Nx - cubeD, concatX);
    auto ys = linspace(0, Ny - cubeD, concatY);
    auto zs = linspace(0, Nz - cubeD, concatZ);

    // Optional: add origin cube like Python does
    zs.insert(zs.begin(), 0);
    ys.insert(ys.begin(), 0);
    xs.insert(xs.begin(), 0);

    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                std::vector<double> cube(cubeD * cubeD * cubeD);
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int src_idx = (z0 + dz) * Ny * Nx + (y0 + dy) * Nx + (x0 + dx);
                            int cube_idx = dz * cubeD * cubeD + dy * cubeD + dx;
                            cube[cube_idx] = data[src_idx];
                        }
                    }
                }
                cubes.push_back(std::move(cube));
            }
        }
    }

    return cubes;
}
