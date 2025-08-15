#ifndef ML_COUPLING_MAIA_Ref_HPP
#define ML_COUPLING_MAIA_Ref_HPP

#include "ml_coupling/maia/ml_coupling_maia.hpp"  // Defines the templated base class MLCoupling
#include <mpi.h>  // For MPI_Comm

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

class MLCouplingMaiaRef : public MLCouplingMaia<double, double>{
public:
    MLCouplingMaiaRef();
    virtual ~MLCouplingMaiaRef();

    void init() override;

    void setup(
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
        int param_hdfOutputInterval,
        int param_totalTimesteps,
        //new ones
        int param_forecastWindow,
        int param_inputStepDistance,
        int param_scalingFactor,
        int param_overlap,
        int param_cubeD
    ) override;

    MPI_Comm getComm() override;

    void finalize() override;

protected:
    //Internal ML pipeline steps
    void preprocess_input();
    void inference();
    void postprocess_output();
};

#endif // ML_COUPLING_MAIA_Ref_HPP