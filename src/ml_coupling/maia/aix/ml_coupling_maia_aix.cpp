#include "ml_coupling/maia/aix/ml_coupling_maia_aix.hpp"
#include "ml_coupling/maia/ml_coupling_maia.hpp"
#include "ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <mpi.h>
#include <cstring>
#include <math.h>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cassert>

// Constructor and destructor
MLCouplingMaiaAix::MLCouplingMaiaAix() = default;

MLCouplingMaiaAix::~MLCouplingMaiaAix() {
    finalize();
}

void MLCouplingMaiaAix::init() {
    couplingStrategy = new MLCouplingStrategyAix<float, float>();
    couplingStrategy->init();
}

void MLCouplingMaiaAix::setup(
    std::vector<double*> input_fields_ptr, 
    std::vector<double*> output_fields_ptr,
    const std::string& param_model_path,
    const std::vector<int>& param_nCells,
    const std::vector<int>& param_nOffsetCells,
    int param_nGhostLayers
){
    MLCouplingMaia::setup(input_fields_ptr, output_fields_ptr, param_model_path, param_nCells, param_nOffsetCells, param_nGhostLayers);

    // Precompute strides in the original (ghost-including) input.
    yzStride = nCells[1] * nCells[2];
    rowStride = nCells[2];

    // Durch scripting erwartet jetzt [batchdim = nfields*numCubes][seqlen = 5][cubeD^3 = 8^3 = 512]
    inputShape = {nFields * numCubes, sequenceLen, cubeD * cubeD * cubeD};
    // Output ist dann [batchdim = nFields*numCubes][forecastwindow = 2][cubeD^3 = 512]
    outputShape = {nFields * numCubes, 2, cubeD * cubeD * cubeD};
    batchSize = inputShape[0]; //Since batch first = True; = nFields * num_cubes
    
    input_fields_pre = new float[totalElements];
    output_fields_post = new float[outputShape[0] * outputShape[1] * outputShape[2]];

    couplingStrategy->setup(
        model_path,
        inputShape,
        input_fields_pre,
        outputShape,
        output_fields_post,
        batchSize,
        app_comm
    );


    cubeSrcBases.resize(numCubes);
    int idx = 0;
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // For a cube starting at (x0, y0, z0) in the trimmed region,
                // the corresponding global (input_fields) base offset is:
                int base = ( (z0 + nGhostLayers) * yzStride ) +
                           ( (y0 + nGhostLayers) * nCells[2] ) +
                           ( x0 + nGhostLayers );
                cubeSrcBases[idx++] = base;
            }
        }
    }

    cubeDestBases.resize(nFields * numCubes);
    for (int f = 0; f < nFields; ++f) {
        for (int c = 0; c < numCubes; ++c) {
            int batch_index = f * numCubes + c;
            cubeDestBases[batch_index] = batch_index * sequenceLen * cubeSize;
        }
    }

    cubeOffsets.resize(cubeSize);
    // Precompute relative offsets inside a cube.
    int offsetIdx = 0;
    for (int dz = 0; dz < cubeD; ++dz) {
        for (int dy = 0; dy < cubeD; ++dy) {
            for (int dx = 0; dx < cubeD; ++dx) {
                // For an element at local coordinate (dz, dy, dx) in the cube,
                // its offset in a full volume is:
                cubeOffsets[offsetIdx++] = dz * yzStride + dy * nCells[2] + dx;
            }
        }
    }

    // Precompute base offsets for every cube extracted.
    cubeBaseOffsets.resize(numCubes);
    idx = 0;
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // The cube's base is the (global) offset in the full volume for its (0,0,0) element.
                // Add the ghost layer offset.
                int base = (z0 + nGhostLayers) * yzStride 
                        + (y0 + nGhostLayers) * nCells[2] 
                        + (x0 + nGhostLayers);
                cubeBaseOffsets[idx++] = base;
            }
        }
    }
    

    // The geometry never changes, precompute the weight map only once and store it.
     // cached weight map; computed only during first call.
    if (weight.size() != static_cast<size_t>(fullFieldCells)) {
        weight.assign(fullFieldCells, 0.0);
        for (int base : cubeBaseOffsets) {
            // Simply add one contribution per voxel in the cube.
            for (int off : cubeOffsets) {
                weight[base + off] += 1.0;
            }
        }
    }
}

//In: [sequenceLen][field][num_cubes * cubeD³]
//Out: [batchdim = nfields*numCubes][seqlen = 5][cubeD^3 = 8^3 = 512] flat
void MLCouplingMaiaAix::preprocess_input(){
    // Loop over each field.
    std::cout << "test -2" << std::endl;
    for (int f = 0; f < nFields; ++f) {
        // Pointer to this field's input volume.
        const double* srcField = input_fields[f];
        for (int c = 0; c < numCubes; ++c) {
            int batch_index = f * numCubes + c;
            // Compute destination offset for the current time step:
            int destOffset = cubeDestBases[batch_index] + iter * cubeSize;
            float* destPtr = input_fields_pre + destOffset;

            // The top–left–front element of the cube in the input volume:
            int srcBase = cubeSrcBases[c];

            // For each layer (dz) and each row (dy) within the cube, copy cubeD elements.
            // Destination cube is stored contiguously with row stride = cubeD and plane stride = cubeD * cubeD.
            for (int dz = 0; dz < cubeD; ++dz) {
                // Compute the offset for the current cube layer in input.
                int srcLayerOffset = srcBase + dz * yzStride;
                // Compute the offset for the current cube layer in the flat destination:
                int destLayerOffset = dz * (cubeD * cubeD);
                for (int dy = 0; dy < cubeD; ++dy) {
                    int srcRowOffset = srcLayerOffset + dy * rowStride;
                    int destRowOffset = destLayerOffset + dy * cubeD;
                    // Copy a contiguous row of cubeD doubles.
                    //std::memcpy(destPtr + destRowOffset,
                    //              srcField + srcRowOffset,
                    //              cubeD * sizeof(double));

                    // Copy each element with conversion from double to float.
                    for (int i = 0; i < cubeD; ++i) {
                        destPtr[destRowOffset + i] = static_cast<float>(srcField[srcRowOffset + i]);
                    }
                }
            }
        }
    }
    std::cout << "test -1" << std::endl;
}

//In: [batchdim = nfields*numCubes][seqlen = 5][cubeD^3 = 8^3 = 512] flat
//Out: [batchdim = nFields*numCubes][forecastwindow = 2][cubeD^3 = 512] flat
void MLCouplingMaiaAix::inference(){
    couplingStrategy->inference();
}

//In: [batchdim = nFields*numCubes][forecastwindow = 2][cubeD^3 = 512] flat
//Out: [forecastwindow][field][num_cubes * cubeD³]
void MLCouplingMaiaAix::postprocess_output(){    
    std::cout << "test 1" << std::endl;
    // If output_fields[f] is a vector containing the full volume for field f, clear it.
    for (int f = 0; f < nFields; ++f) {
        // Only clear the interior region (leave ghost layers unchanged if needed)
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            int base_z = z * yzStride;
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                int start = base_z + y * rowStride + nGhostLayers;
                std::fill(output_fields[f] + start,
                          output_fields[f] + start + nActiveCells[2],
                          0.0);
            }
        }
    }
    std::cout << "test 2" << std::endl;
    
    // Reconstruct the full volumes directly from the flat output array.
    // The flat array 'output_fields_post' has the layout:
    //    [batch dimension: (f * numCubes + cube)][time: FORECAST_WINDOW][cubeSize]
    // For each field and each cube, we need the predicted cube from time step FORECAST_WINDOW–1.
    // Since shape is fixed, we can compute offsets directly.
    for (int f = 0; f < nFields; ++f) {
        // "cubeCounter" indexes the cube for a given field.
        int cubeCounter = 0;
        for (int base : cubeBaseOffsets) {
            // The global batch index is f * numCubes + cubeCounter.
            int batch_index = f * numCubes + cubeCounter;
            // Each batch element has FORECAST_WINDOW time steps.
            // The predicted cube is located at time step FORECAST_WINDOW - 1.
            // Therefore, its starting offset in output_fields_post is:
            int src_offset = ((batch_index * 2) + (2 - 1)) * cubeSize;
            // Instead of copying cube data into an intermediate vector, add its contribution directly.
            // Pointer arithmetic makes inner loops efficient.
            const float* cubeData = &output_fields_post[src_offset];
            double* fullField   = output_fields[f];
            for (int i = 0; i < cubeSize; ++i) {
                fullField[base + cubeOffsets[i]] += static_cast<double>(cubeData[i]);
            }
            ++cubeCounter;
        }
    std::cout << "test 3" << std::endl;

        // Normalize the reconstructed full volume using the precomputed weight map.
        for (int i = 0; i < fullFieldCells; ++i) {
            // Only divide when there is a contribution.
            if (weight[i] > 0.0)
                output_fields[f][i] /= weight[i];
        }
    }
}

MPI_Comm MLCouplingMaiaAix::getComm(){
    return app_comm;
}

void MLCouplingMaiaAix::finalize() {
    if (couplingStrategy) {
        couplingStrategy->finalize();
        delete couplingStrategy;
        couplingStrategy = nullptr;
    }
    // Free the allocated arrays.
    delete[] input_fields_pre;
    input_fields_pre = nullptr;
    delete[] output_fields_post;
    output_fields_post = nullptr;
}