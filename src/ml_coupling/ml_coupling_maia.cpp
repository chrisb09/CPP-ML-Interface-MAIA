#include "ml_coupling_maia.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>

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
        const std::vector<int>& nOffsetCells
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

    
    concatX = static_cast<int>(std::ceil(static_cast<double>(this->nCells[2]) / cubeD));
    concatY = static_cast<int>(std::ceil(static_cast<double>(this->nCells[1]) / cubeD));
    concatZ = static_cast<int>(std::ceil(static_cast<double>(this->nCells[1]) / cubeD));
    
    //Save where the inputs and also outputs are in maia
    input_fields.clear();
    output_fields.clear();
    for(size_t i = 0; i < nFields; i++){
        input_fields.push_back(input_fields_ptr[i]);  
        output_fields.push_back(output_fields_ptr[i]);  
    }

    coupling_strategy->setup(this->nCells, this->nOffsetCells, nFields, nGhostLayers, fieldSize, this->app_comm, sequenceLen);
}

void MLCouplingMaia::preprocess_input(
    std::vector<double*>& input_fields, 
    std::vector<std::vector<double>>& input_fields_pre)
{
    std::vector<std::vector<double>> field_cubes(3); // [field][num_cubes * cubeD³]
    int concatX = (nCells[2] + cubeD - 1) / cubeD;
    int concatY = (nCells[1] + cubeD - 1) / cubeD;
    int concatZ = (nCells[0] + cubeD - 1) / cubeD;
    std::cout << "input_fields is: " << input_fields.size() << std::endl;
    // Extract cubes for u, v, w
    for (int f = 0; f < 3; ++f) {
        auto cubes = extract_cubes(input_fields[f], nCells[2], nCells[1], nCells[0], cubeD, concatX, concatY, concatZ);
        for (const auto& cube : cubes) {
            field_cubes[f].insert(field_cubes[f].end(), cube.begin(), cube.end());
        }
    }

    // Interleave into [num_cubes * 3 * cubeD³]
    size_t num_cubes = field_cubes[0].size() / (cubeD * cubeD * cubeD);
    std::vector<double> flat;

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
    // === Parameters assumed available as member variables ===
    // cubeD: The cube edge length.
    // nCells: An int array (or vector) with { Nz, Ny, Nx }
    //
    // Note: In extract_cubes we used:
    //   Nx = nCells[2], Ny = nCells[1], Nz = nCells[0]
    
    int Nx = nCells[2];
    int Ny = nCells[1];
    int Nz = nCells[0];
    size_t volumeSize = static_cast<size_t>(Nx * Ny * Nz);
    const int numFields = 3;  // for example, u, v, w

    // ------------------------------------------------------------------------
    // Step 1. Deinterleave the flat vector into 3 separate vectors (one per field)
    // ------------------------------------------------------------------------
    size_t cubeSize = static_cast<size_t>(cubeD * cubeD * cubeD);
    size_t groupSize = numFields * cubeSize; // data for one cube across all fields
    size_t numCubes = output_fields_post.size() / groupSize;

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

    // ------------------------------------------------------------------------
    // Step 2. Reconstruct full volumes (for each field)
    // ------------------------------------------------------------------------

    // Prepare the weight grid (to count contributions at each voxel)
    std::vector<double> weight(volumeSize, 0.0);

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

    // Build the weight grid by “painting” one cube at each starting position.
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // For the current cube, increment the contribution counter for each voxel.
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int global_x = x0 + dx;
                            int global_y = y0 + dy;
                            int global_z = z0 + dz;
                            if (global_x < Nx && global_y < Ny && global_z < Nz) {
                                int vol_index = global_z * (Ny * Nx) + global_y * Nx + global_x;
                                weight[vol_index] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // Allocate memory for each reconstructed full volume as a contiguous array.
    // (The output_fields vector holds pointers to these arrays.)
    output_fields.clear();
    for (int f = 0; f < numFields; ++f)
    {
        double* vol = new double[volumeSize];
        // Initialize the volume to zero.
        std::fill(vol, vol + volumeSize, 0.0);
        output_fields.push_back(vol);
    }

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
                                int global_x = x0 + dx;
                                int global_y = y0 + dy;
                                int global_z = z0 + dz;
                                if (global_x < Nx && global_y < Ny && global_z < Nz) {
                                    int vol_index = global_z * (Ny * Nx) + global_y * Nx + global_x;
                                    int cube_offset = dz * cubeD * cubeD + dy * cubeD + dx;
                                    // Accumulate the cube’s contribution.
                                    output_fields[f][vol_index] += 
                                        field_cubes[f][cubeIndex * cubeSize + cube_offset];
                                }
                            }
                        }
                    }
                    ++cubeIndex;
                }
            }
        }
        // Normalize the full volume by dividing each voxel by its contribution count.
        for (size_t i = 0; i < volumeSize; ++i)
        {
            if (weight[i] > 0.0)
                output_fields[f][i] /= weight[i];
        }
    }
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
    }
}

    // Helper: linspace equivalent in C++ to generate integer evenly spaced points like numpy.linspace(0, end, count)
std::vector<int> MLCouplingMaia::linspace_int(int start, int end, int count) {
    std::vector<int> result;
    if (count <= 1) {
        result.push_back(start);
        return result;
    }
    double step = static_cast<double>(end - start) / (count - 1);
    for (int i = 0; i < count; ++i) {
        result.push_back(static_cast<int>(std::round(start + i * step)));
    }
    return result;
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