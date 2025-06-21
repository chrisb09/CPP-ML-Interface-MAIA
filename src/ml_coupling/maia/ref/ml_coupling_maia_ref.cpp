#include "ml_coupling/maia/ref/ml_coupling_maia_ref.hpp"
#include "ml_coupling/maia/ml_coupling_maia.hpp"

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


MLCouplingMaiaRef::MLCouplingMaiaRef() = default;

MLCouplingMaiaRef::~MLCouplingMaiaRef() {
    finalize();
}

void MLCouplingMaiaRef::init() {}


void MLCouplingMaiaRef::setup(
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
    // Setup internal base class variables
    MLCouplingMaia::setup(input_fields_ptr, output_fields_ptr, param_model_path, param_nCells, param_nOffsetCells, param_nGhostLayers, param_start, param_sequenceLen, param_interval, param_increment, param_hdfOutputInterval);
}

MPI_Comm MLCouplingMaiaRef::getComm() {
    return MPI_COMM_WORLD;
}

void MLCouplingMaiaRef::preprocess_input(){}

void MLCouplingMaiaRef::inference(){}

void MLCouplingMaiaRef::postprocess_output()  {}

void MLCouplingMaiaRef::finalize() {}