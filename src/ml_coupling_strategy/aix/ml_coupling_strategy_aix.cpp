#pragma once

#include "ml_coupling_strategy.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "aixeleratorService"

class MLCouplingStrategyAix : public MLCouplingStrategy {
private:
    void* aixelerator_ = nullptr;

    std::vector<float> inputData_;
    std::vector<float> outputData_;

public:
    // Constructor / destructor
    MLCouplingStrategyAix() = default;
    ~MLCouplingStrategyAix() override {
        // In case user didn’t call finalize
        if (aixelerator_) {
            deleteAIxeleratorServiceFloat_C(aixelerator_);
            aixelerator_ = nullptr;
        }
    }

    /**
     * @brief Implementation of the Fortran subroutine ml_coupling_strategy_init.
     *        Called by the “init” routine in the coupling logic.
     */
    void ml_coupling_strategy_init(const std::string& model,
                                   const std::vector<int>& input_shape,
                                   const std::vector<int>& output_shape,
                                   int batch_size,
                                   int& comm) override
    {
        // 1) Allocate the float arrays that mirror the Fortran “allocate(...)”
        //    via std::vector::resize in C++.
        //    product(input_shape) is the total # of elements = multiply all dims.
        auto product = [](const std::vector<int>& dims) {
            long long result = 1;
            for (auto d : dims) { result *= d; }
            return result;
        };

        long long nInput = product(input_shape);
        long long nOutput = product(output_shape);

        try {
            inputData_.resize(nInput);
            outputData_.resize(nOutput);
        } catch (const std::bad_alloc&) {
            std::cerr << "ERROR: Could not allocate input_data or output_data!\n";
        }

        // 2) Construct the “input_shape_c” / “output_shape_c” that might omit 
        //    dimension #4 if it’s 1. This reproduces the Fortran logic:
        //       if (input_shape(4) == 1) then => 4D definition
        //       else => 5D definition
        //    (Remember Fortran is 1-based; in C++ we map that carefully.)
        //    input_shape = {dim1, dim2, dim3, dim4, dim5}
        //    so input_shape[3] is the “4th” dimension, etc.
        std::vector<long long> inputShapeC;
        if (input_shape.size() >= 5 && input_shape[3] == 1) {
            // 4D
            inputShapeC = { input_shape[0], input_shape[1], input_shape[2],
                            input_shape[4] };
        } else {
            // 5D
            // (Make sure input_shape has at least length 5.)
            inputShapeC = { input_shape[0], input_shape[1], input_shape[2],
                            input_shape[3], input_shape[4] };
        }

        std::vector<long long> outputShapeC;
        if (output_shape.size() >= 5 && output_shape[3] == 1) {
            // 4D
            outputShapeC = { output_shape[0], output_shape[1], output_shape[2],
                             output_shape[4] };
        } else {
            // 5D
            outputShapeC = { output_shape[0], output_shape[1], output_shape[2],
                             output_shape[3], output_shape[4] };
        }

        // 3) Convert model path to something that mimics "trim(model)//c_null_char" in Fortran.
        //    In practice, you just pass model.c_str() if your library expects a null-terminated string.
        //    We'll store it locally as well:
        std::string modelC = model; // Fortran appended c_null_char, but in C++ we have it implicitly.

        // 4) Now call the hypothetical createAIxeleratorServiceFloat_C. 
        //    In Fortran:
        //    self%aixelerator = createAIxeleratorServiceFloat_C(model_c, input_shape_c, 
        //        size(input_shape_c), self%input_data, output_shape_c, 
        //        size(output_shape_c), self%output_data, batch_size, comm)
        //
        //    We'll mimic that call here by a placeholder function:
        aixelerator_ = createAIxeleratorServiceFloat_C(
            modelC,
            inputShapeC,
            outputShapeC,
            inputData_,
            outputData_,
            batch_size,
            comm
        );
    }

    void ml_coupling_strategy_inference(const double* input_fields, double* output_fields, int dim1, int dim2, int dim3, int dim4, int dim5) override
    {
        inferenceAIxeleratorServiceFloat_C(aixelerator_);
    }

    void ml_coupling_strategy_finalize() override
    {
        if (aixelerator_) {
            deleteAIxeleratorServiceFloat_C(aixelerator_);
            aixelerator_ = nullptr;
        }
        inputData_.clear();
        outputData_.clear();
    }
};
