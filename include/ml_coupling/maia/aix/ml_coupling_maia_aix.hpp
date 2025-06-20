#ifndef ML_COUPLING_MAIA_AIX_HPP
#define ML_COUPLING_MAIA_AIX_HPP

#include "ml_coupling/maia/ml_coupling_maia.hpp"  // Defines the templated base class MLCoupling
#include "ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"
#include <mpi.h>  // For MPI_Comm

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>
#include <math.h>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cassert>

class MLCouplingMaiaAix : public MLCouplingMaia<float*, float*>{
public:
    MLCouplingMaiaAix();
    virtual ~MLCouplingMaiaAix();

    void init() override;
    
    void setup(
        std::vector<double*> input_fields_ptr, 
        std::vector<double*> output_fields_ptr,
        const std::string& param_model_path,
        const std::vector<int>& param_nCells,
        const std::vector<int>& param_nOffsetCells,
        int param_nGhostLayers
    ) override;

    MPI_Comm getComm() override;

    void finalize() override;

protected:
    //Strategy Object
    MLCouplingStrategyAix<float, float>* couplingStrategy;

    //Internal ML pipeline steps
    void preprocess_input();
    void inference();
    void postprocess_output();

    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    int batchSize;

    int yzStride;
    int rowStride;
    
    std::vector<int> cubeBaseOffsets;      // for each cube extraction, the base offset in the full field
    std::vector<int> cubeOffsets;  // relative offsets within a cube
    std::vector<double> weight;

    std::vector<int> cubeSrcBases;
    std::vector<int> cubeDestBases;
};


#endif // ML_COUPLING_MAIA_AIX_HPP