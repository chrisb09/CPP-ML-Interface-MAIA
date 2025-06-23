#include "ml_coupling/maia/phydll/ml_coupling_maia_phydll.hpp"
#include "ml_coupling/maia/ml_coupling_maia.hpp"
#include "ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.hpp"

#include <vector>
#include <string>
#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <math.h>
#include <numeric>
#include <fstream>
#include <sstream>

#ifdef WITH_SCOREP
#include <scorep/SCOREP_User.h>
#endif


MLCouplingMaiaPhyDLL::MLCouplingMaiaPhyDLL() = default;

MLCouplingMaiaPhyDLL::~MLCouplingMaiaPhyDLL() {
    finalize();
}

void MLCouplingMaiaPhyDLL::init() {
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(initRegion);
        SCOREP_USER_REGION_BEGIN(initRegion, "init", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

    couplingStrategy = new MLCouplingStrategyPhyDLL<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>();
    couplingStrategy->init();
    
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(initRegion);
    #endif
}


void MLCouplingMaiaPhyDLL::setup(
    std::vector<double*> input_fields_ptr, 
    std::vector<double*> output_fields_ptr,
    const std::string& param_model_path,
    const std::vector<int>& param_nCells,
    const std::vector<int>& param_nOffsetCells,
    int param_nGhostLayers,
    int param_start,
    int param_sequenceLen,
    int param_interval,
    int param_increment,
    int param_hdfOutputInterval
){
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(setupRegion);
        SCOREP_USER_REGION_BEGIN(setupRegion, "setup", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

    // Setup internal base class variables
    MLCouplingMaia::setup(input_fields_ptr, output_fields_ptr, param_model_path, param_nCells, param_nOffsetCells, param_nGhostLayers, param_start, param_sequenceLen, param_interval, param_increment, param_hdfOutputInterval);

    // Setup PhyDLL comm
    couplingStrategy->setup(true, 1, 1, nFields, numCubes * cubeSize);

    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(metaInfoComm);
        SCOREP_USER_REGION_BEGIN(metaInfoComm, "MetaInfoComm", SCOREP_USER_REGION_TYPE_CODE);
    #endif
   
    // Meta Info communication
    int* metaInfo = (int*) malloc(3 * sizeof(int));
    metaInfo[0] =  this->sequenceLen;
    metaInfo[1] =  this->cubeD;
    metaInfo[2] =  this->totalElements;

    int ndest = couplingStrategy->getNDest();
    int* dest = couplingStrategy->getDest();

    //Send meta information to python
    int own_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);
    for(int i = 0; i < ndest; ++i){
        // Send int metadata
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        MPI_Send(metaInfo, 3, MPI_INT, dest[i], own_rank, MPI_COMM_WORLD);
        #pragma GCC diagnostic pop
    }
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(metaInfoComm);
    #endif


    if (input_fields_pre.size() != 5){
        input_fields_pre.resize(5);
    }
    
    if (output_fields_post.size() != 3){
        output_fields_post.resize(3);
        for (int f = 0; f < nFields; f++){   
            if (output_fields_post[f].size() != numCubes * cubeSize) {
                output_fields_post[f].resize(numCubes * cubeSize);
            }  
        }
    }

    // Precompute the relative offsets for a cube.
    cubeOffsets.reserve(cubeSize);
    for (int dz = 0; dz < cubeD; ++dz) {
        for (int dy = 0; dy < cubeD; ++dy) {
            for (int dx = 0; dx < cubeD; ++dx){
                int offset = dz * (nCells[1] * nCells[2]) + dy * nCells[2] + dx;
                cubeOffsets.push_back(offset);
            }
        }
    }

    cubeVolumeIndices.resize(numCubes); // numCubes = zs.size() * ys.size() * xs.size()
    int cubeIndex = 0;
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // For each cube, precompute the mapping from local cube index to full volume index.
                std::vector<int> mapping(cubeSize);
                int localIdx = 0;
                for (int dz = 0; dz < cubeD; ++dz) {
                    int global_z = z0 + dz + nGhostLayers;
                    for (int dy = 0; dy < cubeD; ++dy) {
                        int global_y = y0 + dy + nGhostLayers;
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int global_x = x0 + dx + nGhostLayers;
                            mapping[localIdx++] = global_z * (nCells[1] * nCells[2])
                                + global_y * nCells[2] + global_x;
                        }
                    }
                }
                cubeVolumeIndices[cubeIndex++] = std::move(mapping);
            }
        }
    }

    weight.assign(fullFieldCells, 0.0);
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                for (int dz = 0; dz < cubeD; ++dz) {
                    int global_z = z0 + dz + nGhostLayers;
                    for (int dy = 0; dy < cubeD; ++dy) {
                        int global_y = y0 + dy + nGhostLayers;
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int global_x = x0 + dx + nGhostLayers;
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
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(setupRegion);
    #endif
}

MPI_Comm MLCouplingMaiaPhyDLL::getComm() {
    return couplingStrategy->getComm();
}

/**
 * Preprocessing function
 */
void MLCouplingMaiaPhyDLL::preprocess_input(){    
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(preprocessRegion);
        SCOREP_USER_REGION_BEGIN(preprocessRegion, "preprocess_input", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

    if (input_fields_pre.size() != 5){
        input_fields_pre.resize(5);
    }   

    input_fields_pre[iter].resize(nFields);
    for (int f = 0; f < nFields; ++f) {
        input_fields_pre[iter][f].resize(numCubes * cubeSize);
        // For each cube, copy data using precomputed mapping.
        for (size_t cube = 0; cube < cubeVolumeIndices.size(); ++cube) {
            int outOffset = cube * cubeSize;
            const std::vector<int>& mapping = cubeVolumeIndices[cube];
            for (int j = 0; j < cubeSize; ++j) {
                input_fields_pre[iter][f][outOffset + j] = input_fields[f][mapping[j]];
            }
        }
    }
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(preprocessRegion);
    #endif
}

void MLCouplingMaiaPhyDLL::inference(){    
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(inferenceRegion);
        SCOREP_USER_REGION_BEGIN(inferenceRegion, "inference", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

    // input_fields_pre: [sequenceLen][field][numCubes * cubeD³]
    for (int t = 0; t < sequenceLen; ++t) { 
        for(int f = 0; f < nFields; f++){         
            double* ptr = input_fields_pre[t][f].data();  
            std::cout << "Sending " << input_fields_pre[t].size() << " doubles "<< std::endl;
            couplingStrategy->setField(&ptr, (char*)"Python-DL-FIELD-INPUT-0");
            couplingStrategy->setField(&ptr, (char*)"Python-DL-FIELD-INPUT-1");
            couplingStrategy->setField(&ptr, (char*)"Python-DL-FIELD-INPUT-2");
        }   
        couplingStrategy->sendFields();
    }

    //ML does work here in Python

    couplingStrategy->receiveFields();

    
    for(int f = 0; f < nFields; f++){                
        double* ptr = output_fields_post[f].data();

        // Create a writable buffer for the label
        constexpr int label_size = 128;  // or LL_CHAR if defined
        char label[label_size] = {0};

        // Initialize the label with the literal string
        std::string fieldlabel = "Python-DL-FIELD-OUTPUT" + std::to_string(f);
        strncpy(label, fieldlabel.c_str(), label_size - 1);
        label[label_size - 1] = '\0'; // null terminate to be safe

        couplingStrategy->getField(&ptr, label); // now label is writable
    }
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(inferenceRegion);
    #endif
}

void MLCouplingMaiaPhyDLL::postprocess_output()  {      // Output: three reconstructed full volumes
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(postprocessRegion);
        SCOREP_USER_REGION_BEGIN(postprocessRegion, "postprocess_output", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

    // Initialize the destination full volumes in the active region to zero.
    for (int f = 0; f < nFields; ++f) {
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    int full_idx = z * (nCells[1] * nCells[2])
                                 + y * nCells[2] + x;
                    output_fields[f][full_idx] = 0.0;
                }
            }
        }
    }
    
    // Stitch the cubes from output_fields_post directly into output_fields.
    for (int f = 0; f < nFields; ++f) {
        int numCubes = cubeVolumeIndices.size();
        for (int cube = 0; cube < numCubes; ++cube) {
            int cubeBase = cube * cubeSize;
            const std::vector<int>& mapping = cubeVolumeIndices[cube];
            for (int j = 0; j < cubeSize; ++j) {
                output_fields[f][mapping[j]] += output_fields_post[f][cubeBase + j];
            }
        }        
        // Finally, normalize the full volume writes by dividing by the voxel weight.
        for (int i = 0; i < fullFieldCells; ++i) {
            if (weight[i] > 0.0) {
                output_fields[f][i] /= weight[i];
            }
        }
    }
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(postprocessRegion);
    #endif
}


void MLCouplingMaiaPhyDLL::finalize() {
    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_DEFINE(finalizeRegion);
        SCOREP_USER_REGION_BEGIN(finalizeRegion, "finalize", SCOREP_USER_REGION_TYPE_FUNCTION);
    #endif

    if (couplingStrategy) {
        couplingStrategy->finalize();
        delete couplingStrategy;
        couplingStrategy = nullptr;
    }

    #ifdef WITH_SCOREP
        SCOREP_USER_REGION_END(finalizeRegion);
    #endif
}