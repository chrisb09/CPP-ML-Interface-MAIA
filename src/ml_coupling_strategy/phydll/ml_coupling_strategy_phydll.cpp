#include "ml_coupling_strategy_phydll.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <mpi.h>
#include <cstring>
#include <math.h>
#include <numeric>

// Include the external C header.
extern "C" {
  #include "phydll.h"
}

MLCouplingStrategyPhyDll::MLCouplingStrategyPhyDll() = default;

MLCouplingStrategyPhyDll::~MLCouplingStrategyPhyDll() {
    finalize();
}

void MLCouplingStrategyPhyDll::init() {
    if (!is_phydll_initialized) {
        phydll_init((char*)"physical");
        is_phydll_initialized = true;
    }
}

void MLCouplingStrategyPhyDll::setup(
    std::vector<int> nCells, 
    std::vector<int> nOffsetCells, 
    int cubeD,
    std::vector<int> activeCells,
    int nFields, 
    int nGhostLayers, 
    int activeFieldSize, 
    int sequenceLen
) {
    this->nCells = nCells;
    this->nOffsetCells = nOffsetCells;
    this->cubeD = cubeD;
    this->activeCells = activeCells;
    this->nFields = nFields;
    this->nGhostLayers = nGhostLayers;
    this->activeFieldSize = activeFieldSize;
    this->sequenceLen = sequenceLen;

    // phydll options
    phydll_opt_enable_cpl_loop();
    phydll_opt_set_freq(1);
    phydll_opt_set_output_freq(1);

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



    // define physical solver instance
    phydll_define_phy(3, /*this->nFields **/ num_cubes * cube_volume);


    int* metaInfoInts = (int*) malloc(3 * sizeof(int));
    metaInfoInts[0] =  this->sequenceLen;
    metaInfoInts[1] =  this->cubeD;
    metaInfoInts[2] =  this->total_elements;

    int ndest = phydll_get_ndest();
    int* dest = phydll_get_dest();

    //Send meta information to python
    int own_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);
    for(int i = 0; i < ndest; ++i){
        // Send int metadata
        MPI_Send(metaInfoInts, 3, MPI_INT, dest[i], own_rank, MPI_COMM_WORLD);
    }
}

void MLCouplingStrategyPhyDll::inference(
    std::vector<std::vector<std::vector<double>>>& input_fields,
    std::vector<std::vector<double>>& output_fields) 
{
    sendFields(input_fields);
    receiveFields(output_fields);
}

void MLCouplingStrategyPhyDll::sendFields(std::vector<std::vector<std::vector<double>>>& input_fields_pre) {
    // input_fields_pre: [sequenceLen][num_cubes * 3 * cubeD³]
    for (int t = 0; t < sequenceLen; ++t) { 
        for(int f = 0; f < nFields; f++){         
            double* ptr = input_fields_pre[t][f].data();  
            std::cout << "Sending " << input_fields_pre[t].size() << " doubles "<< std::endl;
            phydll_set_field(&ptr, (char*)"Python-DL-FIELD-INPUT-0");
            phydll_set_field(&ptr, (char*)"Python-DL-FIELD-INPUT-1");
            phydll_set_field(&ptr, (char*)"Python-DL-FIELD-INPUT-2");
        }   
        phydll_send();
    }
}

void MLCouplingStrategyPhyDll::receiveFields(std::vector<std::vector<double>>& output_fields_post) {
    phydll_recv();

    output_fields_post.resize(3);
    for(int f = 0; f < nFields; f++){         
        output_fields_post[f].resize(this->num_cubes * /*this->nFields **/ this->cube_volume);
        
        double* ptr = output_fields_post[f].data();

        // Create a writable buffer for the label
        constexpr int label_size = 128;  // or LL_CHAR if defined
        char label[label_size] = {0};

        // Initialize the label with the literal string
        std::string fieldlabel = "Python-DL-FIELD-OUTPUT" + std::to_string(f);
        strncpy(label, fieldlabel.c_str(), label_size - 1);
        label[label_size - 1] = '\0'; // null terminate to be safe

        phydll_get_field(&ptr, label); // now label is writable
    }
}

void MLCouplingStrategyPhyDll::finalize() {
    phydll_finalize();
}

MPI_Comm MLCouplingStrategyPhyDll::getComm() {
    return phydll_get_local_mpi_comm();
}