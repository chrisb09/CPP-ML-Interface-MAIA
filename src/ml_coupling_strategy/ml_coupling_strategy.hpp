#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "mpi.h"
template <typename In, typename Out>

class MLCouplingStrategy {
public:
    virtual ~MLCouplingStrategy() = default;

    virtual void init() = 0;

    /*virtual void setup(
        const std::string& model,
        const std::vector<int>& input_shape,
        const std::vector<int>& output_shape,
        int batch_size,
        MPI_Comm comm
    ) = 0;*/

    virtual void setup(
        std::vector<int> nCells, 
        std::vector<int> nOffsetCells, 
        int cubeD,
        std::vector<int> activeCells,
        int nFields, 
        int nGhostLayers, 
        int fieldSize, 
        int sequenceLen
    ) = 0;

    virtual void inference(In& input_fields, Out& output_fields) = 0;

    virtual void finalize() = 0;

    virtual MPI_Comm getComm() = 0;

protected:
    std::string model_path;
    int nFields;
    std::vector<int> nCells;
    std::vector<int> nOffsetCells;
    int nGhostLayers;
    int activeFieldSize;
    int sequenceLen;

    int num_cubes;
    int cube_volume;
    int total_elements;
    std::vector<int> activeCells;
    int cubeD;
};