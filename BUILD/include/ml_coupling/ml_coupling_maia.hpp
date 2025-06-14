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

// types: T, ProcessedType
class MLCouplingMaia : public MLCoupling<double, std::vector<std::vector<double>>, std::vector<double>>
{
protected:
	std::vector<int> nCells;
	std::vector<int> nOffsetCells;
	int nGhostLayers;
    int nFields;
    int fieldSize;

    static constexpr int cubeD = 8;
    static constexpr int cubeSize = cubeD * cubeD * cubeD;
    int concatX;
    int concatY;
    int concatZ;

    std::vector<int> xs;
    std::vector<int> ys;
    std::vector<int> zs;

    std::vector<int> nActiveCells; //nCells without ghostcells
    int activeFieldSize;
    
	int iter = 0;
	int sequenceLen = 0;
    // Store the un-interleaved cubes from the preprocess step:
    // (for each field f, we have one long vector of size num_cubes * cubeSize)
    std::vector<std::vector<double>> m_preFieldCubes;

    // Store the un-interleaved cubes from the postprocess step:
    std::vector<std::vector<double>> m_postFieldCubes;
public:
    void exportCubesToCSV(const std::string& filename);


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
        const std::vector<int>& nOffsetCells,
        int nGhostLayers
    );
    
    // The main ML step that processes a time step.
    void ml_step();

    // Preprocess the input fields into the format expected by the ML model.
    void preprocess_input(
        std::vector<double*>& input_fields, 
        std::vector<std::vector<std::vector<double>>>& input_fields_pre
    );

    // Run the inference using the coupling strategy.
    void inference(
        std::vector<std::vector<std::vector<double>>>& input_fields_pre, 
        std::vector<std::vector<double>>& output_fields_post
    );

    // Post-process the output fields.
    void postprocess_output(
        std::vector<std::vector<double>>& output_fields_post, 
        std::vector<double*>& output_fields
    );

    // Free allocated resources.
    void finalize();

    // Returns the MPI communicator from the underlying strategy.
    MPI_Comm getComm();

    // Helper: extract cubes from a 3D field array.
    std::vector<std::vector<double>> extract_cubes(const double* data);

    // Helper: generate integer points (like numpy.linspace).
    std::vector<int> linspace(int start, int end, int count);
};

#endif // ML_COUPLING_MAIA_HPP