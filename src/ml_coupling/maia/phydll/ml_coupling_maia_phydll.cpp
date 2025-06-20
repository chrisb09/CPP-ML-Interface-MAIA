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


MLCouplingMaiaPhyDLL::MLCouplingMaiaPhyDLL() = default;

MLCouplingMaiaPhyDLL::~MLCouplingMaiaPhyDLL() {
    finalize();
}

void MLCouplingMaiaPhyDLL::init() {
    couplingStrategy = new MLCouplingStrategyPhyDLL<std::vector<std::vector<std::vector<double>>>, double*>();
    couplingStrategy->init();
}


void MLCouplingMaiaPhyDLL::setup(
    std::vector<double*> input_fields_ptr, 
    std::vector<double*> output_fields_ptr,
    const std::string& param_model_path,
    const std::vector<int>& param_nCells,
    const std::vector<int>& param_nOffsetCells,
    int param_nGhostLayers
){
    // Setup internal base class variables
    MLCouplingMaia::setup(input_fields_ptr, output_fields_ptr, param_model_path, param_nCells, param_nOffsetCells, param_nGhostLayers);
    // Setup PhyDLL comm
    couplingStrategy->setup(true, 1, 1, nFields, numCubes * cubeSize);

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

    outputShape = {nFields * numCubes, 2, cubeD * cubeD * cubeD};
    output_fields_post = new double[outputShape[0] * outputShape[1] * outputShape[2]];

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

MPI_Comm MLCouplingMaiaPhyDLL::getComm() {
    return couplingStrategy->getComm();
}

/**
 * Preprocessing function
 */
void MLCouplingMaiaPhyDLL::preprocess_input(){ 
    if (input_fields_pre.size() != 5){
        input_fields_pre.resize(5);
    }
    std::vector<std::vector<double>> field_cubes(nFields); // [field][numCubes * cubeD³]
    input_fields_pre[iter].resize(nFields);
    // Extract cubes for u, v, w
    for (int f = 0; f < nFields; ++f) {
        std::vector<double> trimmed_data;
        trimmed_data.reserve(activeFieldCells);
        // Flattened index access: input_fields[f] is of size nCells[0]*nCells[1]*nCells[2]
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    int idx = (z * nCells[1] * nCells[2]) + (y * nCells[2]) + x;
                    trimmed_data.push_back(input_fields[f][idx]);
                }
            }
        }
        auto cubes = extract_cubes(trimmed_data.data());       

        for (const auto& cube : cubes) {
            field_cubes[f].insert(field_cubes[f].end(), cube.begin(), cube.end());
        }
    }
    
    for (int f = 0; f < nFields; ++f) {
        input_fields_pre[iter][f].resize(activeFieldCells);
        input_fields_pre[iter][f] = (std::move(field_cubes[f]));
    }
    //input_fields_pre.push_back(std::move(flat));

    // Interleave into [numCubes * nFields * cubeD³]
    /*size_t numCubes = field_cubes[0].size() / (cubeSize);
    std::vector<double> flat;
    flat.reserve(numCubes * nFields * cubeSize);
    for (size_t i = 0; i < numCubes; ++i) {
        for (int f = 0; f < nFields; ++f) {
            flat.insert(flat.end(),
                        field_cubes[f].begin() + i * cubeSize,
                        field_cubes[f].begin() + (i + 1) * cubeSize);
        }
    }
    input_fields_pre.push_back(std::move(flat));*/
}

void MLCouplingMaiaPhyDLL::inference(){
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

    //ML does work here

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
}

void MLCouplingMaiaPhyDLL::postprocess_output()  {      // Output: three reconstructed full volumes
       // -------------------------------------------------------------------
    // STEP 1: Create the weight map to count the number of contributions
    // -------------------------------------------------------------------
    std::vector<double> weight(fullFieldCells, 0.0);
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                for (int dz = 0; dz < cubeD; ++dz) {
                    int global_z = z0 + dz + nGhostLayers;
                    if (global_z >= nCells[0])
                        continue;
                    int zOffset = global_z * yzStride;
                    for (int dy = 0; dy < cubeD; ++dy) {
                        int global_y = y0 + dy + nGhostLayers;
                        if (global_y >= nCells[1])
                            continue;
                        int yOffset = global_y * rowStride;
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int global_x = x0 + dx + nGhostLayers;
                            if (global_x >= nCells[2])
                                continue;
                            int vol_index = zOffset + yOffset + global_x;
                            weight[vol_index] += 1.0;
                        }
                    }
                }
            }
        }
    }
    
    // -----------------------------------------------------------
    // STEP 2: Clear the full field output volumes (for each field)
    // -----------------------------------------------------------
    for (int f = 0; f < nFields; ++f) {
        // Only clear the interior region (leave ghost layers unchanged if needed)
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            int base_z = z * yzStride;
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                int start = base_z + y * rowStride + nGhostLayers;
                int width = nCells[2] - 2 * nGhostLayers;
                std::fill(output_fields[f] + start,
                          output_fields[f] + start + width,
                          0.0);
            }
        }
    }
    
    // -------------------------------------------------------------------
    // STEP 3: Reconstruct the full volumes by “pasting” each predicted cube.
    // -------------------------------------------------------------------
    // The cube predictions were produced in the same order as extraction:
    // For each field f and cube index (from 0 to numCubes - 1):
    //   batch_index = f * numCubes + cube_index
    // We now map each cube back into the full volume.
    const int numX = static_cast<int>(xs.size());
    const int numY = static_cast<int>(ys.size());
    // (Assuming xs, ys, zs are ordered and their product equals numCubes.)
    for (int f = 0; f < nFields; ++f) {
        for (int cube = 0; cube < numCubes; ++cube) {
            int batch_index = f * numCubes + cube;
            // Compute the flat offset in flatArrayOut for the last forecast step:
            int flat_idx = (batch_index * 2 + (2 - 1)) * cubeSize;
            const double* cubePtr = &output_fields_post[flat_idx];
            
            // Determine the cube starting indices (using the same nested‑loop order as in preprocessing).
            int cube_index = cube;  // 0 <= cube < numCubes
            int idxZ = cube_index / (numX * numY);
            int rem  = cube_index % (numX * numY);
            int idxY = rem / numX;
            int idxX = rem % numX;
            int x0 = xs[idxX];
            int y0 = ys[idxY];
            int z0 = zs[idxZ];
            
            // “Paste” the cube into the full volume for field f.
            for (int dz = 0; dz < cubeD; ++dz) {
                int global_z = z0 + dz + nGhostLayers;
                if (global_z >= nCells[0])
                    continue;
                int zOffset = global_z * yzStride;
                for (int dy = 0; dy < cubeD; ++dy) {
                    int global_y = y0 + dy + nGhostLayers;
                    if (global_y >= nCells[1])
                        continue;
                    int yOffset = global_y * rowStride;
                    int dest_base = zOffset + yOffset + (x0 + nGhostLayers);
                    int cube_row_offset = dz * (cubeD * cubeD) + dy * cubeD;
                    for (int dx = 0; dx < cubeD; ++dx) {
                        int global_x = x0 + dx + nGhostLayers;
                        if (global_x >= nCells[2])
                            continue;
                        int vol_index = dest_base + dx;
                        output_fields[f][vol_index] += cubePtr[cube_row_offset + dx];
                    }
                }
            }
        }
        // Normalize by dividing each voxel by the number of contributions.
        for (int i = 0; i < fullFieldCells; ++i) {
            if (weight[i] > 0.0)
                output_fields[f][i] /= weight[i];
        }
    }
    




    //#ifdef OUTPUT_FIELDS
    /*{
        int target_y = 2;  // Global y you want
        int target_y_trimmed = target_y - nGhostLayers;

        int trimmed_z = nCells[0] - 2 * nGhostLayers;
        int trimmed_y = nCells[1] - 2 * nGhostLayers;
        int trimmed_x = nCells[2] - 2 * nGhostLayers;

        int nCubesZ = trimmed_z / cubeD;
        int nCubesY = trimmed_y / cubeD;
        int nCubesX = trimmed_x / cubeD;

        int cubeSize = cubeD * cubeD * cubeD;
        int numCubes = output_fields_post.size() / (nFields * cubeSize);

        std::ostringstream filename;
        filename << "cubes_received_y_slice_.csv";
        std::ofstream csv(filename.str());
        csv << "cube_id,field,x,z,value\n";

        for (int cube_id = 0; cube_id < numCubes; ++cube_id) {
            int cube_idx = cube_id;

            int zc = cube_id / (nCubesX * nCubesY);
            int yc = (cube_id / nCubesX) % nCubesY;
            int xc = cube_id % nCubesX;

            int y_start = yc * cubeD;
            int y_end = y_start + cubeD;

            if (target_y_trimmed >= y_start && target_y_trimmed < y_end) {
                int local_y = target_y_trimmed - y_start;

                for (int f = 0; f < nFields; ++f) {
                    const double* cube_ptr = &output_fields_post[(cube_id * nFields + f) * cubeSize];

                    for (int z = 0; z < cubeD; ++z) {
                        for (int x = 0; x < cubeD; ++x) {
                            size_t idx = z * cubeD * cubeD + local_y * cubeD + x;
                            double val = cube_ptr[idx];
                            csv << cube_id << "," << f << "," << x << "," << z << "," << val << "\n";
                        }
                    }
                }
            }
        }

        csv.close();


        std::ofstream csv_end("u_slice_y2_post_start.csv");
        if (!csv_end.is_open()) {
            std::cerr << "Error opening file u_slice_y2_post_start.csv for writing." << std::endl;
        } else {
            for (int z = 0; z < nCells[0]; ++z) {
                for (int x = 0; x < nCells[2]; ++x) {
                    int idx = z * nCells[1] * nCells[2] + 2 * nCells[2] + x;
                    csv_end << output_fields[0][idx];
                    if (x != nCells[2] - 1)
                        csv_end << ",";
                }
                csv_end << "\n";
            }
            csv_end.close();
        }
    }*/
    //#endif

    // ------------------------------------------------------------------------
    // Step 1. Deinterleave the flat vector into 3 separate vectors (one per field)
    // ------------------------------------------------------------------------
    //size_t cubeSize = static_cast<size_t>(cubeD * cubeD * cubeD);
    //size_t groupSize = nFields * cubeSize; // data for one cube across all fields
    //size_t numCubes = output_fields_post.size() / groupSize;

    // Create three temporary vectors to store concatenated cube data for each field.
    /*std::vector<std::vector<double>> field_cubes(nFields);
    for (int f = 0; f < nFields; ++f) {
        field_cubes[f].reserve(numCubes * cubeSize);
    }*/

    // Iterate through each cube (as interleaved groups) and extract each field’s cube.
    /*for (size_t cubeIndex = 0; cubeIndex < numCubes; ++cubeIndex) {
        for (int f = 0; f < nFields; ++f) {
            size_t startIndex = cubeIndex * groupSize + f * cubeSize;
            field_cubes[f].insert(
                field_cubes[f].end(),
                output_fields_post.begin() + startIndex,
                output_fields_post.begin() + startIndex + cubeSize
            );
        }
    }*/
   /*for (int f = 0; f < nFields; ++f) {
    field_cubes[f] = output_fields_post[f];
   }
    
    // ---------------------------------------------------
    // Keep them for comparison with the pre-processing step:
    m_postFieldCubes = field_cubes;

    // ------------------------------------------------------------------------
    // Step 2. Reconstruct full volumes (for each field)
    // ------------------------------------------------------------------------

    // Prepare the weight grid (to count contributions at each voxel)
    //NxFull * NyFull * NzFull
    std::vector<double> weight(fullFieldCells, 0.0);

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
                            if (global_x < nCells[2] && global_y < nCells[1] && global_z < nCells[0]) {
                                int vol_index = global_z * (nCells[1] * nCells[2]) + global_y * nCells[2] + global_x;
                                weight[vol_index] += 1.0;
                            }
                        }
                    }
                }
            }
        }
    }
    // Fill outputfields at appropriate positions with 0
    for (int f = 0; f < nFields; ++f) {
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    int full_idx = z * nCells[1] * nCells[2] + y * nCells[2] + x;
                    output_fields[f][full_idx] = 0;
                }
            }
        }
    }*/

    /*for (int f = 0; f < nFields; ++f)
    {
        // Use the preallocated pointer from output_fields[i] (already set in setup())
        double* vol = output_fields[f];  // pointer already points to pvariables[i]
        std::fill(vol, vol + (NxFull * NyFull * NzFull), 0.0);
        // Do not push_back again! You're already using it.
    }*/
   
    // For each field, “stitch” the cubes back into the full volume.
    // The cubes were extracted (and later deinterleaved) in the same order as defined by
    // iterating over z, then y, then x coordinates given by zs, ys, xs.
    /*for (int f = 0; f < nFields; ++f){
        int cubeIndex = 0;
        // Loop over the cube starting positions in the same order as extraction.
        for (int z0 : zs) {
            for (int y0 : ys) {
                for (int x0 : xs) {
                    // For each cube, loop over its local voxel coordinates.
                    for (int dz = 0; dz < cubeD; ++dz) {
                        for (int dy = 0; dy < cubeD; ++dy) {
                            for (int dx = 0; dx < cubeD; ++dx){
                                int global_x = x0 + dx + nGhostLayers;
                                int global_y = y0 + dy + nGhostLayers;
                                int global_z = z0 + dz + nGhostLayers;
                                if (global_x < nCells[2] && global_y < nCells[1] && global_z < nCells[0]) {
                                    int vol_index = global_z * (nCells[1] * nCells[2]) + global_y * nCells[2] + global_x;
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
        for (int i = 0; i < fullFieldCells; ++i){
            if (weight[i] > 0.0){
                output_fields[f][i] /= weight[i];
            }
        }
    }*/
}

void MLCouplingMaiaPhyDLL::finalize() {
    if (couplingStrategy) {
        couplingStrategy->finalize();
        delete couplingStrategy;
        couplingStrategy = nullptr;
    }
}