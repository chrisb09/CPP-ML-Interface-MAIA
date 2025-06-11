#include "ml_coupling_strategy_phydll.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdlib>       // for malloc/free
#include <mpi.h>

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

    // define physical solver instance
    phydll_define_phy(1, fieldSize * this->sequenceLen);


    int* metaInfoField = (int*) malloc(7 * sizeof(int));
    metaInfoField[0] =  this->nCells[0]; 
    metaInfoField[1] =  this->nCells[1];
    metaInfoField[2] =  this->nCells[2];
    metaInfoField[3] =  this->nGhostLayers;
    metaInfoField[4] =  this->nOffsetCells[0];
    metaInfoField[5] =  this->nOffsetCells[1];
    metaInfoField[6] =  this->nOffsetCells[2];

    int ndest = phydll_get_ndest();
    int* dest = phydll_get_dest();

    //Send meta information to python
    int own_rank;
    MPI_Comm_rank(comm, &own_rank);
    std::cout << "Own rank: " << own_rank << ", ndest: " << ndest << std::endl;
    std::cout << "Destinations: ";
    for(int i = 0; i < ndest; ++i) {
        std::cout << dest[i] << " ";
    }
    std::cout << std::endl;
    //MPI_Request* requests = (MPI_Request*) malloc(ndest * sizeof(MPI_Request));
    for(int i = 0; i < ndest; ++i){
        std::cout << "Sending meta information to destination: " << dest[i] << ". Own rank: " << own_rank << std::endl;
        MPI_Send(metaInfoField, 7, MPI_INT, dest[i], own_rank, comm);//, &requests[i]);
    }
    //MPI_Waitall(ndest, requests, MPI_STATUSES_IGNORE);
    std::cout << "Before free requests." << std::endl;
    //free(requests);
    std::cout << "Sent meta information to Python." << std::endl;
}

void MLCouplingStrategyPhyDll::inference(
    std::vector<std::vector<double>>& input_fields,
    std::vector<std::vector<double>>& output_fields) 
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
        //double* field_data = input_fields_pre[t].data();
        //phydll_set_field(field_data, (char*) this->phyLabels[t].c_str());
    double* ptr = transformer_input.data();
    phydll_set_field(&ptr, (char*)"Python-DL-FIELD-INPUT");

    phydll_send();
    }

    // Set one field, reshape is [sequenceLen, vectorLen]
}

void MLCouplingStrategyPhyDll::receiveFields(std::vector<std::vector<double>>& output_fields_post) {
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