#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

class MLCouplingStrategy {
public:
    virtual ~MLCouplingStrategy() = default;

    virtual void init() = 0;

    virtual void setup(
        const std::string& model,
        const std::vector<int>& input_shape,
        const std::vector<int>& output_shape,
        int batch_size,
        int& comm
    ) = 0;

    virtual void setup(
        int fieldSize, int nFields, int& comm
    ) = 0;

    virtual void inference(const double* input_fields, double* output_fields) = 0;

    virtual void finalize() = 0;

    virtual MPI_Comm getComm() = 0;

protected:
};