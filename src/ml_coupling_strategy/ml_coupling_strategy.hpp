#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>


class MLCouplingStrategy {
protected:
    static bool is_phydll_initialized = false;
public:
    virtual ~MLCouplingStrategy() = default;

    virtual void ml_coupling_strategy_init(
        const std::string& model,
        const std::vector<int>& input_shape,
        const std::vector<int>& output_shape,
        int batch_size,
        int& comm
    ) = 0;

    void ml_strategy_mpmd_init(int ml_coupling_strategy_id, int gcomm, int& lcomm){
        if (ml_coupling_strategy_id == 2 && !is_phydll_initialized) {
            std::string instance = "physical";
            phydll_init_f(instance, lcomm); 

            std::cerr << "[ml_strategy_mpmd_init] Initializing PhyDLL with instance = "
                    << instance << " and comm = " << gcomm << std::endl;

            is_phydll_initialized = true;
        }
    }

    virtual void ml_coupling_strategy_inference(
        const double* input_fields,    // 5D data in Fortran
        double*       output_fields,   // 5D data in Fortran
        // Possibly also pass dimension parameters as needed
        int dim1, int dim2, int dim3, int dim4, int dim5
    ) = 0;

    virtual void ml_coupling_strategy_finalize() = 0;
};