#pragma once

template <typename In, typename Out>
class MLCouplingStrategy{
public:
    virtual ~MLCouplingStrategy() = default;

    virtual void init() = 0;

    virtual void inference() = 0;

    virtual void finalize() = 0;

    virtual MPI_Comm getComm() = 0;
};