#include "ml_coupling_strategy_phydll.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdlib>       // for malloc/free
#include <mpi.h>
#include <cstring>
#include <math.h>

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

void MLCouplingStrategyPhyDll::setup(std::vector<int> nCells, std::vector<int> nOffsetCells, int nFields, int nGhostLayers, int fieldSize, MPI_Comm comm, int sequenceLen) {
    this->nCells = nCells;
    this->nOffsetCells = nOffsetCells;
    this->nFields = nFields;
    this->nGhostLayers = nGhostLayers;
    this->sequenceLen = sequenceLen;
    this->fieldSize = fieldSize;

    // labels for physical fields
    std::string phyFieldPrefix = "MAIA-PHY-STEP-";
    std::string phyFieldIDStr;
    std::string phyFieldLabel;
    this->phyLabels.resize(this->sequenceLen);
    for(int i = 0; i < this->sequenceLen; i++) {
        phyFieldIDStr = std::to_string(i);
        phyFieldLabel = phyFieldPrefix + phyFieldIDStr;
        this->phyLabels[i] = phyFieldLabel;
    }
    
    // allocate memory space to store the labels for DL fields
    this->dlLabels.resize(this->sequenceLen, "");
    for(int i = 0; i < this->sequenceLen; i++) {
        // 128 characters for each label should (hopefully) be enough!
        this->dlLabels[i].resize(128);
    }

    // phydll options
    phydll_opt_enable_cpl_loop();
    phydll_opt_set_freq(1);
    phydll_opt_set_output_freq(1);


    int cubeD = 8;
    
    int concatX = static_cast<int>(std::ceil(static_cast<double>(this->nCells[2]) / cubeD));
    int concatY = static_cast<int>(std::ceil(static_cast<double>(this->nCells[1]) / cubeD));
    int concatZ = static_cast<int>(std::ceil(static_cast<double>(this->nCells[0]) / cubeD));

    // === Inline linspace logic for Z ===
    std::vector<int> zs(concatZ);
    if (concatZ == 1) {
        zs[0] = 0;
    } else {
        double step = static_cast<double>(this->nCells[0] - cubeD) / (concatZ - 1);
        for (int i = 0; i < concatZ; ++i) {
            zs[i] = static_cast<int>(std::round(i * step));
        }
    }
    zs.insert(zs.begin(), 0); // Add origin cube

    // === Inline linspace logic for Y ===
    std::vector<int> ys(concatY);
    if (concatY == 1) {
        ys[0] = 0;
    } else {
        double step = static_cast<double>(this->nCells[1] - cubeD) / (concatY - 1);
        for (int i = 0; i < concatY; ++i) {
            ys[i] = static_cast<int>(std::round(i * step));
        }
    }
    ys.insert(ys.begin(), 0); // Add origin cube

    // === Inline linspace logic for X ===
    std::vector<int> xs(concatX);
    if (concatX == 1) {
        xs[0] = 0;
    } else {
        double step = static_cast<double>(this->nCells[0] - cubeD) / (concatX - 1);
        for (int i = 0; i < concatX; ++i) {
            xs[i] = static_cast<int>(std::round(i * step));
        }
    }
    xs.insert(xs.begin(), 0); // Add origin cube

    // === Count total cubes and calculate total elements ===
    this->num_cubes = zs.size() * ys.size() * xs.size();
    this->cube_volume = cubeD * cubeD * cubeD;
    this->total_elements = this->sequenceLen * this->nFields * num_cubes * cube_volume;



    // define physical solver instance
    phydll_define_phy(1, this->nFields * num_cubes * cube_volume);


    int* metaInfoField = (int*) malloc(8 * sizeof(int));
    metaInfoField[0] =  this->nCells[0]; 
    metaInfoField[1] =  this->nCells[1];
    metaInfoField[2] =  this->nCells[2];
    metaInfoField[3] =  this->nGhostLayers;
    metaInfoField[4] =  this->nOffsetCells[0];
    metaInfoField[5] =  this->nOffsetCells[1];
    metaInfoField[6] =  this->nOffsetCells[2];
    metaInfoField[7] =  this->total_elements;

    int ndest = phydll_get_ndest();
    int* dest = phydll_get_dest();

    //Send meta information to python
    int own_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &own_rank);
    std::cout << "Own rank: " << own_rank << ", ndest: " << ndest << std::endl;
    std::cout << "Destinations: ";
    for(int i = 0; i < ndest; ++i) {
        std::cout << dest[i] << " ";
    }
    std::cout << std::endl;
    //MPI_Request* requests = (MPI_Request*) malloc(ndest * sizeof(MPI_Request));
    for(int i = 0; i < ndest; ++i){
        std::cout << "Sending meta information to destination: " << dest[i] << ". Own rank: " << own_rank << std::endl;
        MPI_Send(metaInfoField, 8, MPI_INT, dest[i], own_rank, MPI_COMM_WORLD);//, &requests[i]);
    }
    //MPI_Waitall(ndest, requests, MPI_STATUSES_IGNORE);
    std::cout << "Before free requests." << std::endl;
    //free(requests);
    std::cout << "Sent meta information to Python." << std::endl;
}

void MLCouplingStrategyPhyDll::inference(
    std::vector<std::vector<double>>& input_fields,
    std::vector<double>& output_fields) 
{
    sendFields(input_fields);
    receiveFields(output_fields);
}

void MLCouplingStrategyPhyDll::sendFields(std::vector<std::vector<double>>& input_fields_pre) {

    // input_fields_pre: [sequenceLen][num_cubes * 3 * cubeD³]
    int sequenceLen = input_fields_pre.size();
    int vectorLen = input_fields_pre[0].size();

    std::vector<double> transformer_input(sequenceLen * vectorLen);
    for (int t = 0; t < sequenceLen; ++t) {
        std::copy(input_fields_pre[t].begin(),
                input_fields_pre[t].end(),
                transformer_input.begin());
                
        double* ptr = transformer_input.data();
        phydll_set_field(&ptr, (char*)"Python-DL-FIELD-INPUT");

        phydll_send();
    }

    // Set one field, reshape is [sequenceLen, vectorLen]
}

void MLCouplingStrategyPhyDll::receiveFields(std::vector<double>& output_fields_post) {
    phydll_recv();
    
    std::cout << "after phydllrecv" << std::endl;

    output_fields_post.resize(this->num_cubes * this->nFields * this->cube_volume);
    std::cout << this->num_cubes * this->nFields * this->cube_volume << std::endl;
    std::cout << output_fields_post.size() << std::endl;
    std::cout << "outputfieldflat1 " << output_fields_post.size() << std::endl;
    
    std::cout << "after resize" << std::endl;

    double* ptr = output_fields_post.data();
    std::cout << "after ptr" << std::endl;

    // Create a writable buffer for the label
    constexpr int label_size = 128;  // or LL_CHAR if defined
    char label[label_size] = {0};

    // Initialize the label with the literal string
    strncpy(label, "Python-DL-FIELD-OUTPUT", label_size - 1);
    label[label_size - 1] = '\0'; // null terminate to be safe

    phydll_get_field(&ptr, label); // now label is writable

    std::cout << "outputfieldflat2 " << output_fields_post.size() << std::endl;
}

void MLCouplingStrategyPhyDll::finalize() {
    phydll_finalize();
}

MPI_Comm MLCouplingStrategyPhyDll::getComm() {
    return phydll_get_local_mpi_comm();
}