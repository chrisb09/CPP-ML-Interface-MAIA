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
//#include "aixeleratorService/aixeleratorService.h"
/*template <typename T>
class AIxeleratorService;*/


class MLCouplingMaiaAix : public MLCouplingMaia<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>{
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
    MLCouplingStrategyAix<double, double>* couplingStrategy;

    //Internal ML pipeline steps
    void preprocess_input();
    void inference();
    void postprocess_output();

    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    int batchSize;

    /*AIxeleratorService<std::vector<std::vector<std::vector<double>>>>* aixelerator;*/
};


#endif // ML_COUPLING_MAIA_AIX_HPP