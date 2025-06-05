#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include "../ml_coupling_strategy/ml_coupling_strategy.hpp" 

class MLCoupling {
protected:
    double* input_fields;
    double* output_fields;

    double* input_fields_pre;
    double* output_fields_post;

    MLCouplingStrategy* coupling_strategy;
    int coupling_strategy_id;

    std::string model_path;

    int batch_size;
    int app_comm;

	int iter = 0;
	int sequenceLen = 0;

	std::vector<int> nCells;
	int nGhostLayers = 2;
	std::vector<int> nOffsetCells;
public:
    virtual ~MLCoupling() = default;

    virtual void init(int strategy_id) = 0;

	virtual void setup(double* input_fields_ptr, double* output_fields_ptr,
				const std::string& modelPath,
				int batchSize,
				const std::vector<int>& nCells,
				const std::vector<int>& nOffsetCells,
				int appComm) = 0;

    virtual void preprocess_input(const double* input, double* input_pre) = 0;

    virtual void inference(const double* input_pre, double* output_post) = 0; 

    virtual void postprocess_output(const double* output_post, double* output) = 0;

    virtual void finalize() = 0;

	virtual MPI_Comm getComm() = 0;

    void ml_step(){
		//With this we ensure that we gather seqLen (4) timesteps and only then infer
		if (iter < sequenceLen-1) {
        	preprocess_input(input_fields, input_fields_pre);
			iter++;
		}else{
        	preprocess_input(input_fields, input_fields_pre);
        	inference(input_fields_pre, output_fields_post);
			postprocess_output(output_fields_post, output_fields);
			iter = 0;
		}
    }
};