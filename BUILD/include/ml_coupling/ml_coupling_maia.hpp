#ifndef ML_COUPLING_MAIA_HPP
#define ML_COUPLING_MAIA_HPP

#include "ml_coupling.hpp"  // Defines the templated base class MLCoupling
#include <vector>
#include <string>
#include <mpi.h>  // For MPI_Comm

// Include strategy headers instead of .cpp files.
#ifdef WITH_AIX
#include "../ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"
#endif
#ifdef WITH_PHYDLL
#include "../ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.hpp"
#endif

class MLCouplingMaia : public MLCoupling<double, std::vector<double>>
{
protected:
    int irank = 0;
    int nGhostLayers = 2;
    int nFields;
    int fieldSize;

    int cubeC = 2;
    int cubeD = 8;
    int concatX;
    int concatY;
    int concatZ;
    
public:
    MLCouplingMaia();
    ~MLCouplingMaia() override;

    // Sets up the coupling strategy based on strategy_id.
    void init(int strategy_id);

    // Setup the coupling: in- and output fields, model and communication settings.
    void setup(std::vector<double*> input_fields_ptr, 
               std::vector<double*> output_fields_ptr,
               const std::string& modelPath,
               int batchSize,
               const std::vector<int>& nCells,
               const std::vector<int>& nOffsetCells
               );

    // Preprocess the input fields into the format expected by the ML model.
    void preprocess_input(std::vector<double*>& input_fields, 
                          std::vector<std::vector<double>>& input_fields_pre);

    // Run the inference using the coupling strategy.
    void inference(std::vector<std::vector<double>>& input_fields_pre, 
                   std::vector<std::vector<double>>& output_fields_post);

    // Post-process the output fields.
    void postprocess_output(std::vector<std::vector<double>>& output_fields_post, 
                            std::vector<double*>& output_fields);

    // Free allocated resources.
    void finalize();

    // Returns the MPI communicator from the underlying strategy.
    MPI_Comm getComm();

    // The main ML step that processes a time step.
    void ml_step();

    // Helper: generate integer points (like numpy.linspace).
    std::vector<int> linspace_int(int start, int end, int count);

    // Helper: extract cubes from a 3D field array.
    std::vector<std::vector<double>> extract_cubes(const double* data, int Nx, int Ny, int Nz, int cubeD);
};

#endif // ML_COUPLING_MAIA_HPP