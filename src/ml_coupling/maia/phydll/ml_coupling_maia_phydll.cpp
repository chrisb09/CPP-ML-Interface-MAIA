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
    couplingStrategy = new MLCouplingStrategyPhyDLL<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>();
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
        std::cout <<"fieldcubes " << field_cubes[f].size() << std::endl;
    }
    
    for (int f = 0; f < nFields; ++f) {
        input_fields_pre[iter][f].resize(field_cubes[f].size());
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
}

void MLCouplingMaiaPhyDLL::postprocess_output()  {      // Output: three reconstructed full volumes
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

    // === Parameters assumed available as member variables ===
    // cubeD: The cube edge length.
    // nCells: An int array (or vector) with { Nz, Ny, Nx }
    //
    // Note: In extract_cubes we used:
    //   Nx = nCells[2], Ny = nCells[1], Nz = nCells[0]
    
    //int NxFull = nCells[2];
    //int NyFull = nCells[1];
    //int NzFull = nCells[0];

    //int Nx = NxFull - 2 * nGhostLayers;
    //int Ny = NyFull - 2 * nGhostLayers;
    //int Nz = NzFull - 2 * nGhostLayers;
    //= activecells

    //size_t volumeSize = static_cast<size_t>(Nx * Ny * Nz);
    //=activeFieldCells
    //const int numFields = 3;  // for example, u, v, w
    //=nFields

    // ------------------------------------------------------------------------
    // Step 1. Deinterleave the flat vector into 3 separate vectors (one per field)
    // ------------------------------------------------------------------------
    //size_t cubeSize = static_cast<size_t>(cubeD * cubeD * cubeD);
    //size_t groupSize = nFields * cubeSize; // data for one cube across all fields
    //size_t numCubes = output_fields_post.size() / groupSize;

    // Create three temporary vectors to store concatenated cube data for each field.
    std::vector<std::vector<double>> field_cubes(nFields);
    for (int f = 0; f < nFields; ++f) {
        field_cubes[f].reserve(numCubes * cubeSize);
    }
    std::cout << "1" << std::endl;
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
   for (int f = 0; f < nFields; ++f) {
    field_cubes[f] = output_fields_post[f];
   }
    std::cout << "2" << std::endl;

    // ------------------------------------------------------------------------
    // Step 2. Reconstruct full volumes (for each field)
    // ------------------------------------------------------------------------

    // Prepare the weight grid (to count contributions at each voxel)
    //NxFull * NyFull * NzFull
    std::vector<double> weight(fullFieldCells, 0.0);

    std::cout << "3" << std::endl;
    // Recompute the extraction starting positions (xs, ys, zs)
    //
    // For the extraction we computed coordinates using linspace:
    //   auto xs = linspace(0, Nx - cubeD, concatX);
    //   auto ys = linspace(0, Ny - cubeD, concatY);
    //   auto zs = linspace(0, Nz - cubeD, concatZ);
    //
    // and then inserted a 0 at the beginning of each.
    //int concatX = (Nx + cubeD - 1) / cubeD;
    //int concatY = (Ny + cubeD - 1) / cubeD;
    //int concatZ = (Nz + cubeD - 1) / cubeD;
    /*auto xs = linspace(0, Nx - cubeD, concatX);  // assumed to return std::vector<int>
    auto ys = linspace(0, Ny - cubeD, concatY);
    auto zs = linspace(0, Nz - cubeD, concatZ);
    xs.insert(xs.begin(), 0);
    ys.insert(ys.begin(), 0);
    zs.insert(zs.begin(), 0);  */
    
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
    std::cout << "4" << std::endl;
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
    }

    std::cout << "5" << std::endl;
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
    for (int f = 0; f < nFields; ++f){
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
    std::cout << "6" << std::endl;

        // Normalize the full volume by dividing each voxel by its contribution count.
        for (int i = 0; i < fullFieldCells; ++i){
            if (weight[i] > 0.0){
                output_fields[f][i] /= weight[i];
            }
        }
    }

    //#ifdef OUTPUT_FIELDS
    /*{
        std::ofstream csv_end("u_slice_y2_post_end.csv");
        if (!csv_end.is_open()) {
            std::cerr << "Error opening file u_slice_y2_post_start.csv for writing." << std::endl;
        } else {
            // Again, for field 0 (adjust if you want to export other fields)
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
}


void MLCouplingMaiaPhyDLL::finalize() {
    if (couplingStrategy) {
        couplingStrategy->finalize();
        delete couplingStrategy;
        couplingStrategy = nullptr;
    }
}