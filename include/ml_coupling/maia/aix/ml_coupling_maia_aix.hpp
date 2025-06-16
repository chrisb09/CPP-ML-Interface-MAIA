#ifndef ML_COUPLING_MAIA_AIX_HPP
#define ML_COUPLING_MAIA_AIX_HPP

#include "ml_coupling/maia/ml_coupling_maia.hpp"  // Defines the templated base class MLCoupling
#include "ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"
#include <mpi.h>  // For MPI_Comm

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>
#include <math.h>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cassert>

// Include strategy headers instead of .cpp files.
#ifdef WITH_AIX
#include "ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"
#endif

class MLCouplingMaiaAix : public MLCouplingMaia<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>{
public:
    MLCouplingMaiaAix();
    ~MLCouplingMaiaAix();

    void init();
    
    void setup(
        std::vector<double*> input_fields_ptr, 
        std::vector<double*> output_fields_ptr,
        const std::string& param_model_path,
        const std::vector<int>& param_nCells,
        const std::vector<int>& param_nOffsetCells,
        int param_nGhostLayers
    ) override;

    MPI_Comm getComm() override;

    void finalize();

protected:
    //Strategy Object
    MLCouplingStrategyAix<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>* couplingStrategy;

    //Internal ML pipeline steps
    void preprocess_input();
    void inference();
    void postprocess_output();

    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    int batchSize;
};


// Constructor and destructor
inline MLCouplingMaiaAix::MLCouplingMaiaAix() = default;

inline MLCouplingMaiaAix::~MLCouplingMaiaAix() {
    finalize();
}

inline void MLCouplingMaiaAix::init() {
    couplingStrategy = new MLCouplingStrategyAix<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>();
    couplingStrategy->init();
}

inline void MLCouplingMaiaAix::setup(
    std::vector<double*> input_fields_ptr, 
    std::vector<double*> output_fields_ptr,
    const std::string& param_model_path,
    const std::vector<int>& param_nCells,
    const std::vector<int>& param_nOffsetCells,
    int param_nGhostLayers
){
    MLCouplingMaia::setup(input_fields_ptr, output_fields_ptr, param_model_path, param_nCells, param_nOffsetCells, param_nGhostLayers);

    // Durch scripting erwartet jetzt [batchdim = nfields*numCubes][seqlen = 5][cubeD^3 = 8^3 = 512]
    inputShape = {nFields * numCubes, sequenceLen, cubeD * cubeD * cubeD};
    // Output ist dann [batchdim = nFields*numCubes][forecastwindow = 2][cubeD^3 = 512]
    outputShape = {nFields * numCubes, 2, cubeD * cubeD * cubeD};
    batchSize = inputShape[0]; //Since batch first = True; = nFields * num_cubes

    couplingStrategy->setup(
        model_path,
        inputShape,
        input_fields_pre,
        outputShape,
        output_fields_post,
        batchSize,
        app_comm
    );
}

//In: [sequenceLen][field][num_cubes * cubeD³]
//Out: [batchdim = nfields*numCubes][seqlen = 5][cubeD^3 = 8^3 = 512]
inline void MLCouplingMaiaAix::preprocess_input(){
    // Desired batch dimension = nFields * numCubes.
    int requiredBatch = nFields * numCubes;
    
    // Instead of having the outer vector sized by the sequence (time), we now want it sized by batch.
    if (input_fields_pre.size() != static_cast<size_t>(requiredBatch)) {
        input_fields_pre.clear();
        input_fields_pre.resize(requiredBatch);
        // Note: The inner vectors (the time series for each batch element) will be built incrementally.
    }
    
    // Process each field separately.
    for (int f = 0; f < nFields; ++f) {
        std::vector<double> trimmed_data;
        trimmed_data.reserve(activeFieldCells);
        // Same as before: extract the “trimmed” data for field f.
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    int idx = (z * nCells[1] * nCells[2]) + (y * nCells[2]) + x;
                    trimmed_data.push_back(input_fields[f][idx]);
                }
            }
        }
        
        // Extract cubes from the trimmed data.
        // Here, extract_cubes returns a vector of cubes for the given field.
        // Each cube is a vector<double> of length cubeD*cubeD*cubeD.
        std::vector<std::vector<double>> cubes = extract_cubes(trimmed_data.data());
        
        // Optional: Check that the number of cubes is as expected.
        assert(cubes.size() == static_cast<size_t>(numCubes));
        
        // Now, instead of storing the cubes under the time step index,
        // we place each cube into its proper "batch" slot.
        // The batch index for field f and cube cube_idx is: f * numCubes + cube_idx.
        for (size_t cube_idx = 0; cube_idx < cubes.size(); ++cube_idx) {
            int batch_index = f * numCubes + static_cast<int>(cube_idx);
            // Instead of overwriting an entire vector (as before) we push_back a new time step.
            // Each call to preprocess_input adds one new time step for each batch element.
            input_fields_pre[batch_index].push_back(std::move(cubes[cube_idx]));
        }
    }
}


//In: [batchdim = nfields*numCubes][seqlen = 5][cubeD^3 = 8^3 = 512]
//Out: [batchdim = nFields*numCubes][forecastwindow = 2][cubeD^3 = 512]
inline void MLCouplingMaiaAix::inference(){
    couplingStrategy->inference();
}

//In: [batchdim = nFields*numCubes][forecastwindow = 2][cubeD^3 = 512]
//Out: [forecastwindow][field][num_cubes * cubeD³]
inline void MLCouplingMaiaAix::postprocess_output(){
    // For clarity, assume:
    //   - forecastWindow is a member variable (e.g., forecastWindow == 2)
    //   - cubeSize = cubeD³
    //   - nFields, numCubes, nCells, nGhostLayers, fullFieldCells, xs, ys, zs, cubeD are known members.
    const int forecastWindow = 2;  

    // -----------------------------------------------------------------------------------
    // STEP 1: Regroup the network output for each field.
    // For each field, we need to collect its numCubes cubes.
    // The batch dimension is arranged as:
    //   batch_index = f * numCubes + cube_idx
    // And each batch element contains forecastWindow time steps; we take the last one.
    std::vector<std::vector<double>> field_cubes(nFields);
    for (int f = 0; f < nFields; ++f) {
        // Allocate storage for all cubes of field f.
        field_cubes[f].resize(numCubes * cubeSize);
        for (int cube = 0; cube < numCubes; ++cube) {
            int batch_index = f * numCubes + cube;
            // Get the predicted cube from the last forecast time step.
            const std::vector<double>& predCube = output_fields_post[batch_index][forecastWindow - 1];
            // Copy the cube data into the proper offset.
            for (int i = 0; i < cubeSize; ++i) {
                field_cubes[f][static_cast<int>(cube) * cubeSize + i] = predCube[i];
            }
        }
    }
    // Save these per-field cube arrays (for later comparison, if needed).
    m_postFieldCubes = field_cubes;

    // -----------------------------------------------------------------------------------
    // STEP 2: Compute the weight map for contributions.
    // Create a weight vector to count how many cubes contribute to each voxel.
    std::vector<double> weight(fullFieldCells, 0.0);
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int global_x = x0 + dx + nGhostLayers;
                            int global_y = y0 + dy + nGhostLayers;
                            int global_z = z0 + dz + nGhostLayers;
                            if (global_x < nCells[2] && global_y < nCells[1] && global_z < nCells[0]) {
                                int vol_index = global_z * (nCells[1] * nCells[2])
                                              + global_y * nCells[2] + global_x;
                                weight[vol_index] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------
    // STEP 3: Clear the output full volumes.
    // output_fields holds three full volumes (one per field); set the inner voxels to 0.
    for (int f = 0; f < nFields; ++f) {
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    int full_idx = z * (nCells[1] * nCells[2]) + y * nCells[2] + x;
                    output_fields[f][full_idx] = 0;
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------
    // STEP 4: Reconstruct the full volumes from the field cubes.
    // For each field, loop over the cube starting positions (ordered as in extraction)
    // and add in the cube’s contributions to the full volume.
    for (int f = 0; f < nFields; ++f) {
        int cubeIndex = 0;
        for (int z0 : zs) {
            for (int y0 : ys) {
                for (int x0 : xs) {
                    // For each extracted cube starting position, add its contribution.
                    for (int dz = 0; dz < cubeD; ++dz) {
                        for (int dy = 0; dy < cubeD; ++dy) {
                            for (int dx = 0; dx < cubeD; ++dx) {
                                int global_x = x0 + dx + nGhostLayers;
                                int global_y = y0 + dy + nGhostLayers;
                                int global_z = z0 + dz + nGhostLayers;
                                if (global_x < nCells[2] && global_y < nCells[1] && global_z < nCells[0]) {
                                    int vol_index = global_z * (nCells[1] * nCells[2])
                                                  + global_y * nCells[2] + global_x;
                                    int cube_offset = dz * cubeD * cubeD + dy * cubeD + dx;
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
        // Normalize each voxel by the number of contributions.
        for (int i = 0; i < fullFieldCells; ++i) {
            if (weight[i] > 0.0) {
                output_fields[f][i] /= weight[i];
            }
        }
    }

    //Empty buffers
    input_fields_pre.clear();
    output_fields_post.clear();
}

inline MPI_Comm MLCouplingMaiaAix::getComm(){
    return app_comm;
}

inline void MLCouplingMaiaAix::finalize() {
    if (couplingStrategy) {
        couplingStrategy->finalize();
        delete couplingStrategy;
        couplingStrategy = nullptr;
    }
}

#endif