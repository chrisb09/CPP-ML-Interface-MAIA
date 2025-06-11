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

    
    concatX = static_cast<int>(std::ceil(static_cast<double>(this->nCells[0]) / cubeD));
    concatY = static_cast<int>(std::ceil(static_cast<double>(this->nCells[1]) / cubeD));
    concatZ = static_cast<int>(std::ceil(static_cast<double>(this->nCells[2]) / cubeD));
    
    //Save where the inputs and also outputs are in maia
    input_fields.clear();
    output_fields.clear();
    for(size_t i = 0; i < nFields; i++){
        input_fields.push_back(input_fields_ptr[i]);  
        output_fields.push_back(output_fields_ptr[i]);  
    }

    coupling_strategy->setup(this->nCells, this->nOffsetCells, nFields, nGhostLayers, fieldSize, this->app_comm, sequenceLen);
}

/*
Inputfieldspre = [sequenceLen][nFields][cubeCount * cubeD^3]
*/
void MLCouplingMaia::preprocess_input(
    std::vector<double*>& input_fields, 
    std::vector<std::vector<double>>& input_fields_pre)
{
    std::vector<std::vector<double>> field_cubes(3); // [field][num_cubes * cubeD³]

    // Extract cubes for u, v, w
    for (int f = 0; f < 3; ++f) {
        auto cubes = extract_cubes(input_fields[f], this->nCells[0], this->nCells[1], this->nCells[2], cubeD);
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

}

void MLCouplingMaia::inference(
    std::vector<std::vector<double>>& input_fields_pre, 
    std::vector<std::vector<double>>& output_fields_post)
{
    if (!coupling_strategy) {
        std::cerr << "ERROR: No coupling strategy set!\n";
        return;
    }
    
    coupling_strategy->inference(input_fields_pre, output_fields_post);
}

void MLCouplingMaia::postprocess_output(
    std::vector<std::vector<double>>& output_fields_post, 
    std::vector<double*>& output_fields)
{
    //output_fields = combineCubes(output_fields_post);
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

    // Helper: Extract cubes from a single 3D field array
    // data: raw field (meshSize elements, in order consistent with 3D mesh indexing)
    // Returns vector of cubes, each cube is cubeD^3 doubles
std::vector<std::vector<double>> MLCouplingMaia::extract_cubes(const double* data, int Nx, int Ny, int Nz, int cubeD) {
    std::vector<std::vector<double>> cubes;

    int stepX = cubeD; // For non-overlapping cubes; or customize stride here
    int stepY = cubeD;
    int stepZ = cubeD;

    for (int z = 0; z + cubeD <= Nz; z += stepZ) {
        for (int y = 0; y + cubeD <= Ny; y += stepY) {
            for (int x = 0; x + cubeD <= Nx; x += stepX) {
                std::vector<double> cube(cubeD * cubeD * cubeD);

                for (int dz = 0; dz < cubeD; dz++) {
                    for (int dy = 0; dy < cubeD; dy++) {
                        for (int dx = 0; dx < cubeD; dx++) {
                            int src_idx = (z + dz) * Ny * Nx + (y + dy) * Nx + (x + dx);
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