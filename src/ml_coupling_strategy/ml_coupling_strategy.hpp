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


    void baseSetup(
        const std::vector<int>& nCells,
        const std::vector<int>& nOffsetCells,
        int cubeD,
        const std::vector<int>& activeCells,
        int nFields,
        int nGhostLayers,
        int activeFieldSize,
        int sequenceLen
    ){
        this->nCells = nCells;
        this->nOffsetCells = nOffsetCells;
        this->cubeD = cubeD;
        this->activeCells = activeCells;
        this->nFields = nFields;
        this->nGhostLayers = nGhostLayers;
        this->activeFieldSize = activeFieldSize;
        this->sequenceLen = sequenceLen;

        
        // Compute number of cubes based on interior only
        int concatX = (activeCells[2] + cubeD - 1) / cubeD;
        int concatY = (activeCells[1] + cubeD - 1) / cubeD;
        int concatZ = (activeCells[0] + cubeD - 1) / cubeD;

        // === Inline linspace logic for Z ===
        std::vector<int> zs(concatZ);
        if (concatZ == 1) {
            zs[0] = 0;
        } else {
            double step = static_cast<double>(activeCells[0]  - cubeD) / (concatZ - 1);
            for (int i = 0; i < concatZ; ++i) {
                zs[i] = static_cast<int>(std::round(i * step));
            }
        }
        zs.insert(zs.begin(), 0); // Add origin cube
        //for (int& z : zs) z += nGhostLayers; // offset to interior position in full domain

        // === Inline linspace logic for Y ===
        std::vector<int> ys(concatY);
        if (concatY == 1) {
            ys[0] = 0;
        } else {
            double step = static_cast<double>(activeCells[1] - cubeD) / (concatY - 1);
            for (int i = 0; i < concatY; ++i) {
                ys[i] = static_cast<int>(std::round(i * step));
            }
        }
        ys.insert(ys.begin(), 0); // Add origin cube
        //for (int& y : ys) y += nGhostLayers;

        // === Inline linspace logic for X ===
        std::vector<int> xs(concatX);
        if (concatX == 1) {
            xs[0] = 0;
        } else {
            double step = static_cast<double>(activeCells[2] - cubeD) / (concatX - 1);
            for (int i = 0; i < concatX; ++i) {
                xs[i] = static_cast<int>(std::round(i * step));
            }
        }
        xs.insert(xs.begin(), 0); // Add origin cube
        //for (int& x : xs) x += nGhostLayers;

        // === Count total cubes and calculate total elements ===
        this->num_cubes = zs.size() * ys.size() * xs.size();
        this->cube_volume = cubeD * cubeD * cubeD;
        this->total_elements = this->sequenceLen * this->nFields * num_cubes * cube_volume;
    }
};