#ifndef ML_COUPLING_STRATEGY_AIX_HPP
#define ML_COUPLING_STRATEGY_AIX_HPP

#include "../ml_coupling_strategy.hpp"  // Your base strategy header
#include <vector>
#include <string>
#include <mpi.h>

class MLCouplingStrategyAix : public MLCouplingStrategy<
    std::vector<std::vector<std::vector<double>>>, 
    std::vector<std::vector<double>>
>
{
public:
    MLCouplingStrategyAix();
    virtual ~MLCouplingStrategyAix();

    void init(
        std::vector<std::vector<std::vector<double>>>& input_fields,
        std::vector<std::vector<std::vector<double>>>& output_fields
    ) /*override*/;

    virtual void setup(
        std::vector<int> nCells, 
        std::vector<int> nOffsetCells, 
        int cubeD,
        std::vector<int> activeCells,
        int nFields, 
        int nGhostLayers, 
        int activeFieldSize, 
        int sequenceLen
    ) override;

    void inference();

    void finalize() override;
private:
    AIxeleratorService<std::vector<std::vector<std::vector<double>>>> aixelerator;

    bool is_Aix_initialized = false;

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    std::vector<std::vector<std::vector<double>>>* input_fields_ptr;
    std::vector<std::vector<std::vector<double>>>* output_fields_ptr;
    int batchsize;
    MPI_COMM comm;


    void preprocess();
    void postprocess();
};

#endif // ML_COUPLING_STRATEGY_AIX_HPP