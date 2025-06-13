#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include "../ml_coupling_strategy/ml_coupling_strategy.hpp" 
#include <mpi.h>
template <typename T, typename ProcessedType>

class MLCoupling {
protected:
    std::vector<T*> input_fields;
    std::vector<T*> output_fields;

    std::vector<ProcessedType> input_fields_pre;
    std::vector<T> output_fields_post;

    MLCouplingStrategy<std::vector<ProcessedType>, std::vector<T>>* coupling_strategy = nullptr;
    int coupling_strategy_id;

    std::string model_path;

    int batch_size;
    MPI_Comm app_comm;

public:
    virtual ~MLCoupling() = default;

    virtual void init(int strategy_id) = 0;

	virtual void setup(std::vector<T*> input_fields_ptr, 
        std::vector<T*> output_fields_ptr,
        const std::string& modelPath,
        int batchSize,
        const std::vector<int>& nCells,
        const std::vector<int>& nOffsetCells,
        int nGhostLayers
    ) = 0;

    virtual void preprocess_input(
        std::vector<T*>& input, 
        std::vector<ProcessedType>& input_pre
    ) = 0;

    virtual void inference(
        std::vector<ProcessedType>& input_pre, 
        std::vector<T>& output_post
    ) = 0; 

    virtual void postprocess_output(
        std::vector<T>& output_post, 
        std::vector<T*>& output
    ) = 0;

    virtual void finalize() = 0;

	virtual MPI_Comm getComm() = 0;

    virtual void ml_step() = 0;
};