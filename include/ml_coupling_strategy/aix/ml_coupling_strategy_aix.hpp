#ifndef ML_COUPLING_STRATEGY_AIX_HPP
#define ML_COUPLING_STRATEGY_AIX_HPP

#include "ml_coupling_strategy/ml_coupling_strategy.hpp"  // Your base strategy header
//#include "ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <mpi.h>
#include "aixeleratorService/aixeleratorService.h"
// Forward declaration for the external templated class.
/*template <typename T>
class AIxeleratorService;*/

template <typename In, typename Out>
class MLCouplingStrategyAix : public MLCouplingStrategy<In, Out>{
public:
    MLCouplingStrategyAix();
    virtual ~MLCouplingStrategyAix();

    void init();

    void setup(
        std::string model_path, 
        std::vector<int64_t> input_shape, 
        In* input_fields_ptr, 
        std::vector<int64_t> output_shape, 
        In* output_fields_ptr,
        int batch_size, 
        MPI_Comm comm
       /* AIxeleratorService<In>* param_aixelerator*/
    ) /*override*/;

    void inference();

    void finalize();

    MPI_Comm getComm();

private:
    AIxeleratorService<In>* aixelerator;

    bool is_Aix_initialized = false;
};


template <typename In, typename Out>
inline MLCouplingStrategyAix<In, Out>::MLCouplingStrategyAix() = default;

template <typename In, typename Out>
inline MLCouplingStrategyAix<In, Out>::~MLCouplingStrategyAix() { 
    finalize();
}

template <typename In, typename Out>
inline void MLCouplingStrategyAix<In, Out>::init() {}

template <typename In, typename Out>
inline void MLCouplingStrategyAix<In, Out>::setup(
    std::string model_path, 
    std::vector<int64_t> input_shape, 
    In* input_fields_ptr, 
    std::vector<int64_t> output_shape, 
    In* output_fields_ptr,
    int batch_size, 
    MPI_Comm comm
    /*AIxeleratorService<In>* param_aixelerator*/
) {
    this->aixelerator = new AIxeleratorService<In>(
        model_path, 
        input_shape, 
        input_fields_ptr, 
        output_shape, 
        output_fields_ptr,
        batch_size, 
        comm
    );/*param_aixelerator;*/
}

template <typename In, typename Out>
inline void MLCouplingStrategyAix<In, Out>::inference(){
    aixelerator->inference();
}

template <typename In, typename Out>
inline void MLCouplingStrategyAix<In, Out>::finalize(){
    delete aixelerator;
}

template <typename In, typename Out>
inline MPI_Comm MLCouplingStrategyAix<In, Out>::getComm(){
    return MPI_COMM_WORLD;
}



#endif // ML_COUPLING_STRATEGY_AIX_HPP