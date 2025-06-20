#pragma once

//#include "ml_coupling_strategy/ml_coupling_strategy.hpp" 
//#include <mpi.h>
//#include <vector>
//#include <string>

template<typename In, typename Out>
class MLCoupling {
public:
    virtual ~MLCoupling() = default;

    virtual void init() = 0;

    virtual void ml_step() = 0;

    virtual void finalize() = 0;

protected:
    // CFD provided fields
    In input_fields;
    Out output_fields;
};