#ifndef ML_COUPLING_STRATEGY_PHYDLL_HPP
#define ML_COUPLING_STRATEGY_PHYDLL_HPP

#include "../ml_coupling_strategy.hpp"  // Your base strategy header
#include <vector>
#include <string>
#include <mpi.h>

class MLCouplingStrategyPhyDll : public MLCouplingStrategy<
    std::vector<std::vector<double>>, 
    std::vector<double>
>
{
public:
    MLCouplingStrategyPhyDll();
    virtual ~MLCouplingStrategyPhyDll();

    // Initializes the external PhyDll library.
    void init() override;

    // Set up the coupling strategy with meta information and MPI communicator.
    virtual void setup(std::vector<int> nCells, 
                       std::vector<int> nOffsetCells, 
                       int nFields, 
                       int nGhostLayers, 
                       int fieldSize, 
                       MPI_Comm comm, 
                       int sequenceLen) override;

    // Executes inference by sending input fields and receiving output fields.
    void inference(std::vector<std::vector<double>>& input_fields,
                   std::vector<double>& output_fields) override;

    // Finalizes the coupling strategy.
    void finalize() override;

    // Returns the local MPI communicator.
    MPI_Comm getComm() override;

    // (Optional) Send input fields via PhyDll.
    void sendFields(std::vector<std::vector<double>>& input_fields_pre);

    // (Optional) Receive output fields via PhyDll.
    void receiveFields(std::vector<double>& output_fields_post);

private:
    std::vector<std::string> phyLabels;
    std::vector<std::string> dlLabels;

    bool is_phydll_initialized = false;
    
    int nFields;
    std::vector<int> nCells;
    std::vector<int> nOffsetCells;
    int nGhostLayers;
    int fieldSize;
    int sequenceLen;

    int num_cubes;
    int cube_volume;
    int total_elements;
};

#endif // ML_COUPLING_STRATEGY_PHYDLL_HPP