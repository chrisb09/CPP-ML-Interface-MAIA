#pragma once

#include "ml_coupling_strategy.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

#include "phydll.h"

class MLCouplingStrategyPhyDll : public MLCouplingStrategy {
private:
    // Fortran code used:
    //   integer :: input_size, output_size
    //   character(kind=c_char, len=64) :: phy_label, dl_label
    //   double precision, dimension(:), pointer :: dl_field  (plus input_data, output_data).
    int input_size_  = 0;
    int output_size_ = 0;

    std::string phy_label_ = "phy_input_field_0";
    std::string dl_label_;

    // In Fortran, real(8), dimension(:), pointer :: input_data, output_data
    // We'll store them in std::vector<double> for convenience.
    std::vector<double> input_data_;
    std::vector<double> output_data_;

    bool is_phydll_initialized;

public:
    MLCouplingStrategyPhyDll() = default;
    virtual ~MLCouplingStrategyPhyDll() {
        // If finalize() wasn't called, do it here.
        ml_coupling_strategy_finalize();
    }

    void ml_coupling_strategy_init(
        const std::string& model,
        const std::vector<int>& input_shape,
        const std::vector<int>& output_shape,
        int batch_size,
        int& comm
    ) override
    {
        // 1) Allocate input_data_ and output_data_ 
        // product() logic in Fortran: product(input_shape).
        auto product = [](const std::vector<int>& dims) {
            long long p = 1;
            for (int d : dims) { p *= d; }
            return p;
        };
        long long inSize  = product(input_shape);
        long long outSize = product(output_shape);

        try {
            input_data_.resize(inSize);
            output_data_.resize(outSize);
        } catch (const std::bad_alloc&) {
            std::cerr << "ERROR: Could not allocate input_data or output_data!\n";
        }

        input_size_  = static_cast<int>(inSize);
        output_size_ = static_cast<int>(outSize);

        // 2) If not already initialized, call phydll_init_f(...)
        if (!is_phydll_initialized) {
            std::string instance = "physical"; // Fortran: instance = "physical"

            // Placeholder or real call into a library:
            phydll_init(instance, comm);
            is_phydll_initialized = true;
        }

        phydll_opt_enable_cpl_loop();
        phydll_opt_set_freq(1);
        phydll_opt_set_output_freq(1);

        // 4) Fortran sets up "num_fields=1" and "field_size = max(input_size, output_size)"
        //    call phydll_define_phy_f(num_fields, field_size)
        int num_fields = 1;
        int field_size = (input_size_ > output_size_) ? input_size_ : output_size_;
        phydll_define_phy(num_fields, field_size);
    }

    void ml_coupling_strategy_inference(
        const double* input_fields, 
        double*       output_fields,
        int dim1, int dim2, int dim3, int dim4, int dim5
    ) override
    {
        // 1) Set the field in PhyDLL
        phydll_set_field(input_data_, phy_label_);

        // 2) Send/Wait
        phydll_isend();
        phydll_wait_isend();

        // 3) Receive/Wait
        phydll_irecv();
        phydll_wait_irecv();

        // 4) Get the field from PhyDLL. Fortran code then calls phydll_get_field_size_f(field_size),
        //    copies from dl_field to output_data_ up to min(field_size, output_size_).
        int field_size = 0;
        // We'll store the returning field in a temporary std::vector<double> dl_field
        // because the Fortran code has "double precision, dimension(:), pointer :: dl_field".
        std::vector<double> dl_field;
        
        phydll_get_field(dl_field, dl_label_);  // We read data into dl_field
        field_size = phydll_get_field_size();   // The actual size the library returned

        // Copy into output_data_
        if (field_size <= output_size_) {
            for (int i = 0; i < field_size; i++) {
                output_data_[i] = dl_field[i];
            }
        } else {
            for (int i = 0; i < output_size_; i++) {
                output_data_[i] = dl_field[i];
            }
        }
    }

    void ml_coupling_strategy_finalize() override
    {
        input_data_.clear();
        output_data_.clear();

        phydll_finalize();
    }

};
