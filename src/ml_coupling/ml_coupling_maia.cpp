#include "ml_coupling_maia.hpp"
#include "maia/maia_helpers.hpp"

#include <iostream>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cassert>

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
    coupling_strategy->init(input_fields_pre, output_fields_post);
    this->app_comm = MPI_COMM_WORLD;
    #endif

    #ifdef WITH_PHYDLL
    if (coupling_strategy_id == 2) {
        coupling_strategy = new MLCouplingStrategyPhyDll();
        std::cout << "Created PhyDLL Coupling\n";
    }
    coupling_strategy->init();
    this->app_comm = coupling_strategy->getComm();
    #endif

    if (!coupling_strategy) {
        std::cerr << "ERROR: Unknown coupling_strategy_id = " << coupling_strategy_id << "!\n";
    }
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
    this->nCells[0] = nCells[0]; 
    this->nCells[1] = nCells[1];
    this->nCells[2] = nCells[2];
    
    this->nOffsetCells.resize(3);
    this->nOffsetCells[0] =  nOffsetCells[0];
    this->nOffsetCells[1] =  nOffsetCells[1];
    this->nOffsetCells[2] =  nOffsetCells[2];   

    this->nGhostLayers = nGhostLayers;

    //Without the ghostcells
    nActiveCells.resize(3);
    nActiveCells[0] = this->nCells[0] - 2 * this->nGhostLayers;
    nActiveCells[1] = this->nCells[1] - 2 * this->nGhostLayers;
    nActiveCells[2] = this->nCells[2] - 2 * this->nGhostLayers;

    activeFieldSize = std::accumulate(nActiveCells.begin(), nActiveCells.end(), 1,  std::multiplies());
    
    concatX = (nActiveCells[2] + cubeD - 1) / cubeD;
    concatY = (nActiveCells[1] + cubeD - 1) / cubeD;
    concatZ = (nActiveCells[0] + cubeD - 1) / cubeD;

    
    xs = linspace(0, nActiveCells[2] - cubeD, concatX);
    ys = linspace(0, nActiveCells[1] - cubeD, concatY);
    zs = linspace(0, nActiveCells[0] - cubeD, concatZ);
    xs.insert(xs.begin(), 0);
    ys.insert(ys.begin(), 0);
    zs.insert(zs.begin(), 0);  
    
    //Save where the inputs and also outputs are in maia
    input_fields.clear();
    output_fields.clear();
    for(size_t i = 0; i < nFields; i++){
        input_fields.push_back(input_fields_ptr[i]);          
        output_fields.push_back(output_fields_ptr[i]); 
    }

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
}

void MLCouplingMaia::ml_step(){
    //With this we ensure that we gather seqLen (4) timesteps and only then infer
    preprocess_input(input_fields, input_fields_pre);   
    if (iter < sequenceLen-1) {
        iter++;
    }else{
        inference(input_fields_pre, output_fields_post);
        postprocess_output(output_fields_post, output_fields);
        exportCubesToCSV("cubes.csv");
        iter = 0;
        
        //Empty buffer
        input_fields_pre.clear();
        output_fields_post.clear();
    }
}

void MLCouplingMaia::preprocess_input(
    std::vector<double*>& input_fields, 
    std::vector<std::vector<std::vector<double>>>& input_fields_pre)
{
    if (coupling_strategy_id == 1){
        preprocess_input_aix(input_fields, input_fields_pre);
    }
    if (coupling_strategy_id == 2){
        preprocess_input_phydll(input_fields, input_fields_pre);
    }
}

void MLCouplingMaia::preprocess_input_phydll(
    std::vector<double*>& input_fields, 
    std::vector<std::vector<std::vector<double>>>& input_fields_pre
){ 
    if (input_fields_pre.size() != 5){
        input_fields_pre.resize(5);
    }
    std::vector<std::vector<double>> field_cubes(nFields); // [field][num_cubes * cubeD³]
    input_fields_pre[iter].resize(nFields);
    // Extract cubes for u, v, w
    for (int f = 0; f < nFields; ++f) {
        std::vector<double> trimmed_data;
        trimmed_data.reserve(activeFieldSize);
        // Flattened index access: input_fields[f] is of size nCells[0]*nCells[1]*nCells[2]
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    size_t idx = (z * nCells[1] * nCells[2]) + (y * nCells[2]) + x;
                    trimmed_data.push_back(input_fields[f][idx]);
                }
            }
        }
        auto cubes = extract_cubes(trimmed_data.data());

        //#ifdef OUTPUT_FIELDS
        {
            int target_y = 2;  // Global y-index you care about
            int target_y_trimmed = target_y - nGhostLayers;

            int trimmed_z = nCells[0] - 2 * nGhostLayers;
            int trimmed_y = nCells[1] - 2 * nGhostLayers;
            int trimmed_x = nCells[2] - 2 * nGhostLayers;

            int nCubesX = trimmed_x / cubeD;
            int nCubesY = trimmed_y / cubeD;
            int nCubesZ = trimmed_z / cubeD;
            std::ostringstream filename;
            filename << "cubes_y_slice_" << iter << "_field_" << f << ".csv";
            std::ofstream csv(filename.str());
            csv << "cube_id,field,x,z,value\n";

            int cube_id = 0;
            for (int zc = 0; zc < nCubesZ; ++zc) {
                for (int yc = 0; yc < nCubesY; ++yc) {
                    for (int xc = 0; xc < nCubesX; ++xc) {
                        const auto& cube = cubes[cube_id];

                        int y_start = yc * cubeD;
                        int y_end = y_start + cubeD;

                        if (target_y_trimmed >= y_start && target_y_trimmed < y_end) {
                            int local_y = target_y_trimmed - y_start;

                            for (int z = 0; z < cubeD; ++z) {
                                for (int x = 0; x < cubeD; ++x) {
                                    size_t idx = z * cubeD * cubeD + local_y * cubeD + x;
                                    double val = cube[idx];
                                    csv << cube_id << "," << f << "," << x << "," << z << "," << val << "\n";
                                }
                            }
                        }
                    }
                }
                ++cube_id;
            }
        }
        //#endif

        for (const auto& cube : cubes) {
            field_cubes[f].insert(field_cubes[f].end(), cube.begin(), cube.end());
        }
    }

    // ---------------------------
    // 1) Save these per-field cube arrays so that later we can compare:
    m_preFieldCubes = field_cubes; 
    
    for (int f = 0; f < nFields; ++f) {
        input_fields_pre[iter][f].resize(activeFieldSize);
        input_fields_pre[iter][f] = (std::move(field_cubes[f]));
    }
    //input_fields_pre.push_back(std::move(flat));

    // Interleave into [num_cubes * nFields * cubeD³]
    /*size_t num_cubes = field_cubes[0].size() / (cubeSize);
    std::vector<double> flat;
    flat.reserve(num_cubes * nFields * cubeSize);
    for (size_t i = 0; i < num_cubes; ++i) {
        for (int f = 0; f < nFields; ++f) {
            flat.insert(flat.end(),
                        field_cubes[f].begin() + i * cubeSize,
                        field_cubes[f].begin() + (i + 1) * cubeSize);
        }
    }
    input_fields_pre.push_back(std::move(flat));*/
}

void MLCouplingMaia::preprocess_input_aix(
    std::vector<double*>& input_fields, 
    // Now interpreted as: outer vector = batch elements,
    // inner vector = time steps, innermost vector = cube data of one cube.
    std::vector<std::vector<std::vector<double>>>& input_fields_pre)
{
    int nCubes = zs.size() * ys.size() * xs.size();
    // Desired batch dimension = nFields * nCubes.
    int requiredBatch = nFields * nCubes;
    
    // Instead of having the outer vector sized by the sequence (time), we now want it sized by batch.
    if (input_fields_pre.size() != static_cast<size_t>(requiredBatch)) {
        input_fields_pre.clear();
        input_fields_pre.resize(requiredBatch);
        // Note: The inner vectors (the time series for each batch element) will be built incrementally.
    }
    
    // Process each field separately.
    for (int f = 0; f < nFields; ++f) {
        std::vector<double> trimmed_data;
        trimmed_data.reserve(activeFieldSize);
        // Same as before: extract the “trimmed” data for field f.
        for (int z = nGhostLayers; z < nCells[0] - nGhostLayers; ++z) {
            for (int y = nGhostLayers; y < nCells[1] - nGhostLayers; ++y) {
                for (int x = nGhostLayers; x < nCells[2] - nGhostLayers; ++x) {
                    size_t idx = (z * nCells[1] * nCells[2]) + (y * nCells[2]) + x;
                    trimmed_data.push_back(input_fields[f][idx]);
                }
            }
        }
        
        // Extract cubes from the trimmed data.
        // Here, extract_cubes returns a vector of cubes for the given field.
        // Each cube is a vector<double> of length cubeD*cubeD*cubeD.
        std::vector<std::vector<double>> cubes = extract_cubes(trimmed_data.data());
        
        // Optional: Check that the number of cubes is as expected.
        assert(cubes.size() == static_cast<size_t>(nCubes));
        
        // Now, instead of storing the cubes under the time step index,
        // we place each cube into its proper "batch" slot.
        // The batch index for field f and cube cube_idx is: f * nCubes + cube_idx.
        for (size_t cube_idx = 0; cube_idx < cubes.size(); ++cube_idx) {
            int batch_index = f * nCubes + static_cast<int>(cube_idx);
            // Instead of overwriting an entire vector (as before) we push_back a new time step.
            // Each call to preprocess_input adds one new time step for each batch element.
            input_fields_pre[batch_index].push_back(std::move(cubes[cube_idx]));
        }
    }
}

void MLCouplingMaia::inference(
    std::vector<std::vector<std::vector<double>>>& input_fields_pre, 
    std::vector<std::vector<double>>& output_fields_post)
{
    if (!coupling_strategy) {
        std::cerr << "ERROR: No coupling strategy set!\n";
        return;
    }
    
    coupling_strategy->inference(input_fields_pre, output_fields_post);
}

void MLCouplingMaia::postprocess_output(
    std::vector<std::vector<double>>& output_fields_post,  // Flat, interleaved data (all cubes)
    std::vector<double*>& output_fields)        // Output: three reconstructed full volumes
{
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
        int num_cubes = output_fields_post.size() / (nFields * cubeSize);

        std::ostringstream filename;
        filename << "cubes_received_y_slice_.csv";
        std::ofstream csv(filename.str());
        csv << "cube_id,field,x,z,value\n";

        for (int cube_id = 0; cube_id < num_cubes; ++cube_id) {
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
    //=activeFieldSize
    //const int numFields = 3;  // for example, u, v, w
    //=nFields

    // ------------------------------------------------------------------------
    // Step 1. Deinterleave the flat vector into 3 separate vectors (one per field)
    // ------------------------------------------------------------------------
    //size_t cubeSize = static_cast<size_t>(cubeD * cubeD * cubeD);
    size_t groupSize = nFields * cubeSize; // data for one cube across all fields
    size_t numCubes = output_fields_post.size() / groupSize;

    // Create three temporary vectors to store concatenated cube data for each field.
    std::vector<std::vector<double>> field_cubes(nFields);
    for (int f = 0; f < nFields; ++f) {
        field_cubes[f].reserve(numCubes * cubeSize);
    }

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
    
    // ---------------------------------------------------
    // Keep them for comparison with the pre-processing step:
    m_postFieldCubes = field_cubes;

    // ------------------------------------------------------------------------
    // Step 2. Reconstruct full volumes (for each field)
    // ------------------------------------------------------------------------

    // Prepare the weight grid (to count contributions at each voxel)
    //NxFull * NyFull * NzFull
    std::vector<double> weight(fieldSize, 0.0);

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
        size_t cubeIndex = 0;
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
        for (size_t i = 0; i < fieldSize; ++i){
            if (weight[i] > 0.0){
                output_fields[f][i] /= weight[i];
            }
        }
    }

    //#ifdef OUTPUT_FIELDS
    {
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
    }
    //#endif
}

MPI_Comm MLCouplingMaia::getComm(){
    return coupling_strategy->getComm();
}

void MLCouplingMaia::finalize() {
    if (coupling_strategy) {
        coupling_strategy->finalize();
        delete coupling_strategy;
        coupling_strategy = nullptr;
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

std::vector<std::vector<double>> MLCouplingMaia::extract_cubes(const double* data){
    std::vector<std::vector<double>> cubes;

    // add origin cube like Python 
    /*zs.insert(zs.begin(), 0);
    ys.insert(ys.begin(), 0);
    xs.insert(xs.begin(), 0);*/

    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                std::vector<double> cube(cubeD * cubeD * cubeD);
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int src_idx = (z0 + dz) * nActiveCells[1] * nActiveCells[2] + (y0 + dy) * nActiveCells[2] + (x0 + dx);
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

void MLCouplingMaia::exportCubesToCSV(const std::string& filename)
{
    // Sanity checks
    if (m_preFieldCubes.empty() || m_postFieldCubes.empty()) {
        std::cerr << "[exportCubesToCSV] Error: no stored cubes. "
                  << "Did you run preprocess/postprocess already?\n";
        return;
    }
    if (m_preFieldCubes.size() != static_cast<size_t>(nFields) ||
        m_postFieldCubes.size() != static_cast<size_t>(nFields))
    {
        std::cerr << "[exportCubesToCSV] Error: mismatch in #fields.\n";
        return;
    }

    // Prepare output
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "[exportCubesToCSV] Could not open '" << filename << "' for writing.\n";
        return;
    }

    // Write header
    ofs << "field,cubeIndex,global_x,global_y,global_z,valPre,valPost\n";

    // The number of cubes is deduced from one of the fields (all should match)
    // e.g. m_preFieldCubes[f].size() == numCubes*cubeSize
    size_t numCubes = 0;
    if (!m_preFieldCubes[0].empty()) {
        numCubes = m_preFieldCubes[0].size() / cubeSize;
    }

    // We re-use the same iteration order that was used in your extract code:
    //   size_t cubeIndex = 0;
    //   for (int z0 : zs) { for (int y0 : ys) { for (int x0 : xs) {

    // Because that is how you matched each "cubeIndex" to (z0,y0,x0).
    // So we do the same:

    size_t cubeIndex = 0;
    // Outer loops over the "start" of each cube
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // For each cube, loop over local voxel coords
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            // Compute global coords in the full volume
                            int global_z = z0 + dz + nGhostLayers;
                            int global_y = y0 + dy + nGhostLayers;
                            int global_x = x0 + dx + nGhostLayers;

                            // local offset inside this cube
                            int localIndex = dz * (cubeD * cubeD) + dy * cubeD + dx;

                            // For each field, we can store a row
                            // or you can do a single field at a time; up to you.
                            for (int f = 0; f < nFields; ++f) {
                                // Pre
                                double valPre =
                                    m_preFieldCubes[f][ cubeIndex * cubeSize + localIndex ];

                                // Post
                                double valPost =
                                    m_postFieldCubes[f][ cubeIndex * cubeSize + localIndex ];

                                ofs << f << ","
                                    << cubeIndex << ","
                                    << global_x << ","
                                    << global_y << ","
                                    << global_z << ","
                                    << valPre  << ","
                                    << valPost << "\n";
                            }
                        } // dx
                    } // dy
                } // dz
                ++cubeIndex;
            } // x0
        } // y0
    } // z0

    ofs.close();
    std::cout << "[exportCubesToCSV] Finished writing " << filename << "\n";
}