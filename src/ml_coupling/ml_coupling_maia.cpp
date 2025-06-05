#pragma once

#include "ml_coupling.hpp"
#include "ml_coupling_strategy.hpp"
#include "ml_coupling_strategy_aix.cpp"
#include "ml_coupling_strategy_phydll.cpp"

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include "mpi.h"

class MLCouplingMaia : public MLCoupling
{
public:
    int irank = 0;

public:
    MLCouplingMaia() = default;
    ~MLCouplingMaia() override 
    {
        // Ensure resources are freed if user forgot to call finalize().
        finalize();
    }

    void init(double* input_fields_ptr, 
              double* output_fields_ptr,
              const std::string& modelPath,
              int batchSize,
              const std::vector<int>& input_shape,
              const std::vector<int>& output_shape,
              const std::vector<int>& ghostCells,
              int strategy_id,
              int appComm) override
    {
        // 1) Store pointers and parameters in base-class fields
        //    (mirroring self%input_fields => input_fields in Fortran)
        input_fields       = input_fields_ptr;
        output_fields      = output_fields_ptr;
        model_path         = modelPath;
        batch_size         = batchSize;
        coupling_strategy_id = strategy_id;
        app_comm           = appComm;


        irank = MPI_Comm_rank(app_comm, irank);

        // 2) Allocate memory for input_fields_pre, output_fields_post, ghost_cells
        //    in the base class, these are double*.
        auto product = [](const std::vector<int>& dims){
            long long prod = 1;
            for(int d : dims) { prod *= d; }
            return prod;
        };

        long long inCount  = product(input_shape);
        long long outCount = product(output_shape);

        input_fields_pre  = new double[inCount];
        output_fields_post= new double[outCount];

        // Store ghost cell settings:
        // MLCouplingT has a std::vector<int> ghost_cells as well
        ghost_cells_.resize(3);
        ghost_cells_[0] = ghostCells[0];
        ghost_cells_[1] = ghostCells[1];
        ghost_cells_[2] = ghostCells[2];

        // 3) Pick the coupling strategy, as in Fortran:
        //    if (coupling_strategy_id == 1) then
        //        allocate(ml_coupling_strategy_aix_t :: self%coupling_strategy)
        //    if (coupling_strategy_id == 2) then
        //        allocate(ml_coupling_strategy_phydll_t :: self%coupling_strategy)
        if (coupling_strategy_id == 1) {
            coupling_strategy = new MLCouplingStrategyAix();
        } else if (coupling_strategy_id == 2) {
            coupling_strategy = new MLCouplingStrategyPhyDll();
        }
        if (!coupling_strategy) {
            std::cerr << "ERROR: Unknown coupling_strategy_id = " 
                      << coupling_strategy_id << "!\n";
            // throw or handle error
        }

        // Initialize the chosen strategy:
        coupling_strategy->ml_coupling_strategy_init(
            model_path,
            input_shape,
            output_shape,
            batchSize,
            app_comm
        );
    }


    void preprocess_input(const double* input_fields, double* input_fields_pre) override {}

    void inference(const double* input_fields_pre, double* output_fields_post) override {
        if (!coupling_strategy) {
            std::cerr << "ERROR: No coupling strategy set!\n";
            return;
        }
        // In Fortran:
        //   call self%coupling_strategy%ml_coupling_strategy_inference(input_fields_pre,
        //                                                              output_fields_post)
        // In C++:
        //   We pass dimension parameters too. Adjust as needed:
        int dim1 = 1, dim2 = 1, dim3 = 1, dim4 = 1, dim5 = 1; 
        // You would store them from init(...) or compute them from input_shape & output_shape.
        coupling_strategy->ml_coupling_strategy_inference(
            input_fields_pre,
            output_fields_post,
            dim1, dim2, dim3, dim4, dim5
        );
    }

    void postprocess_output(const double* output_fields_post, double* output_fields) override{}

    void finalize() override {
        // If the strategy is still present, finalize it
        if (coupling_strategy) {
            coupling_strategy->ml_coupling_strategy_finalize();
            delete coupling_strategy;
            coupling_strategy = nullptr;
        }

        // free the base-class pointers if allocated
        if (input_fields_pre) {
            delete[] input_fields_pre;
            input_fields_pre = nullptr;
        }
        if (output_fields_post) {
            delete[] output_fields_post;
            output_fields_post = nullptr;
        }
    }
};
