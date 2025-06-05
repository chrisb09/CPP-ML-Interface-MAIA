#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include "ml_coupling_strategy.hpp" 

class MLCoupling {
protected:
    // Fortran had: real(kind=8), dimension(:,:,:,:), pointer :: input_fields
    // We'll store them as 1D pointers plus shape info. You can choose
    // a more sophisticated container, e.g. std::vector<double>, if needed.
    double* input_fields = nullptr; // 4D in Fortran
    double* output_fields = nullptr; // 4D in Fortran

    // Fortran had 5D arrays here
    double* input_fields_pre = nullptr; // 5D in Fortran
    double* output_fields_post= nullptr; // 5D in Fortran

    std::vector<int> ghost_cells;

    // pointer to a coupling strategy
    MLCouplingStrategyT* coupling_strategy = nullptr;
    int coupling_strategy_id;

    std::string model_path;

    int batch_size;
    int app_comm;

public:
    // Virtual destructor
    virtual ~MLCoupling() = default;

    virtual void init(double* input_fields_ptr, double* output_fields_ptr,
                      const std::string& modelPath,
                      int batchSize,
                      const std::vector<int>& input_shape,
                      const std::vector<int>& output_shape,
                      const std::vector<int>& ghostCells,
                      double in_min, double in_max,
                      double n_min, double n_max,
                      double out_min, double out_max,
                      int strategy_id,
                      int appComm) = 0;

    virtual void preprocess_input(const double* input, /*4D logic*/ double* input_pre /*5D logic*/) = 0;

    virtual void inference(const double* input_pre /*5D*/, double* output_post /*5D*/) = 0; 

    virtual void postprocess_output(const double* output_post /*5D*/, double* output /*4D*/) = 0;

    virtual void finalize() = 0;

    void ml_step()
    {
        preprocess_input(input_fields, input_fields_pre);
        inference(input_fields_pre, output_fields_post);
        postprocess_output(output_fields_post, output_fields);
    }

    void ml_coupling_mpmd_init(int ml_coupling_strategy_id, int gcomm, int& lcomm)
    {
        if (!coupling_strategy) {
            std::cerr << "No coupling strategy available!\n";
            throw std::runtime_error("Null coupling strategy pointer");
        }
        coupling_strategy->mpmd_init(ml_coupling_strategy_id, gcomm, lcomm);
    }
};