#ifndef ML_COUPLING_MAIA_HPP
#define ML_COUPLING_MAIA_HPP

#include "ml_coupling/ml_coupling.hpp"  // Defines the templated base class MLCoupling
#include <mpi.h>  // For MPI_Comm
#include <numeric>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

#include <highfive/highfive.hpp>

#ifdef WITH_SCOREP
#include <scorep/SCOREP_User.h>
SCOREP_USER_REGION_DEFINE(ml_stepRegion);
SCOREP_USER_REGION_DEFINE(init_ml_stepRegion);
#endif

template <typename modelIn, typename modelOut>
class MLCouplingMaia : public MLCoupling<std::vector<double*>, std::vector<double*>>{
public:
    MLCouplingMaia();
    virtual ~MLCouplingMaia();

    virtual void init() override = 0;

    virtual void setup(
        std::vector<double*> input_fields_ptr, 
        std::vector<double*> output_fields_ptr,
        const std::string& param_model_path,
        const std::vector<int>& param_nCells,
        const std::vector<int>& param_nOffsetCells,
        int param_nGhostLayers,        
        int param_start,
        int param_sequenceLen,
        int param_interval,
        int param_increment,
        int param_hdfOutputInterval,
        int param_totalTimesteps
    ){
        // Setup in/output for CFD
        input_fields.clear();
        output_fields.clear();
        for(int i = 0; i < nFields; i++){
            input_fields.push_back(input_fields_ptr[i]);          
            output_fields.push_back(output_fields_ptr[i]); 
        }

        // Setup internal variables
        this->model_path = param_model_path;
        this->nCells = param_nCells;
        this->totalTimesteps = param_totalTimesteps;

        fullFieldCells = std::accumulate(this->nCells.begin(), this->nCells.end(), 1, std::multiplies());

        this->nOffsetCells = param_nOffsetCells;
        this->nGhostLayers = param_nGhostLayers;

        nActiveCells.resize(3);
        nActiveCells[0] = this->nCells[0] - 2 * this->nGhostLayers;
        nActiveCells[1] = this->nCells[1] - 2 * this->nGhostLayers;
        nActiveCells[2] = this->nCells[2] - 2 * this->nGhostLayers;
        activeFieldCells = std::accumulate(nActiveCells.begin(), nActiveCells.end(), 1,  std::multiplies());
        
        concatX = (nActiveCells[2] + cubeD - 1) / cubeD;
        concatY = (nActiveCells[1] + cubeD - 1) / cubeD;
        concatZ = (nActiveCells[0] + cubeD - 1) / cubeD;

        xs = linspace(0, nActiveCells[2] - cubeD, concatX);
        ys = linspace(0, nActiveCells[1] - cubeD, concatY);
        zs = linspace(0, nActiveCells[0] - cubeD, concatZ);
        xs.insert(xs.begin(), 0);
        ys.insert(ys.begin(), 0);
        zs.insert(zs.begin(), 0);  

        numCubes = zs.size() * ys.size() * xs.size();
        totalElements = sequenceLen * nFields * numCubes * cubeSize;

        //Setup ML Metadata
        inferenceStartStep = param_start;
        inputSeqLen = param_sequenceLen;
        inferenceInterval = param_interval;
        inferenceIncrement = param_increment;
        nextInferenceStep = param_start;
        hdfOutputInterval = param_hdfOutputInterval;
        if(param_start < (param_sequenceLen - 1)){
            std::cerr << "m-AIA ERROR: inference start step cannot be smaller than input sequence length of the transformer model!" << "\n";
        }
    }
    
    // The main ML step that processes a time step.
    void ml_step() override {
        #ifdef WITH_SCOREP 
            if(firstMLStep == true){
                SCOREP_USER_REGION_BEGIN(init_ml_stepRegion, "MLCouplingMaia::init_ml_stepRegion", SCOREP_USER_REGION_TYPE_FUNCTION);
            }else{
                SCOREP_USER_REGION_BEGIN(ml_stepRegion, "MLCouplingMaia::ml_step", SCOREP_USER_REGION_TYPE_FUNCTION);
            }
        #endif
        
        //With this we ensure that we gather seqLen (4) timesteps and only then infer
        preprocess_input();   
        if (iter < sequenceLen-1) {
            iter++;
        }else{
            inference();
            postprocess_output();
            //exportCubesToCSV("cubes.csv");
            iter = 0;
        }
        
        #ifdef WITH_SCOREP
            if(firstMLStep == true){
                SCOREP_USER_REGION_END(init_ml_stepRegion);
                if(iter == 0){
                    firstMLStep = false;
                }
            }else{
                SCOREP_USER_REGION_END(ml_stepRegion);
            }
        #endif 
    }

    // Free allocated resources.
    virtual void finalize() override = 0;

    // Returns the MPI communicator from the underlying strategy.
    virtual MPI_Comm getComm() = 0;

    int getNextInferenceStep();
    void setNextInferenceStep(int nextGlobalInferenceStep, int nextInferStep);
    int getInferenceIncrement();
    int getInferenceInterval();
    bool isCouplingStep(int step);
    bool isInferenceStep(int step);
    void writeUVWFieldsToH5(double* uField, double* vField, double* wField, std::vector<int> write_nCells, std::vector<int> write_nOffsetCells, std::string outputFile);

    void calculateMetrics();
protected:
    //Communication
    MPI_Comm app_comm = MPI_COMM_WORLD;

    //Processed data, specific to strategy
    modelIn input_fields_pre;
    modelOut output_fields_post;

    //Transformer relevant data
    std::string model_path;
    int totalTimesteps;

    //Timestep data
	int sequenceLen = 5;
	int iter = 0;

    //Cube relevant data
    static constexpr int cubeD = 8;
    static constexpr int cubeSize = cubeD * cubeD * cubeD;

    int concatX;
    int concatY;
    int concatZ;
    std::vector<int> xs;
    std::vector<int> ys;
    std::vector<int> zs;
    int numCubes;
    int totalElements;

    //Maia grid data
    static constexpr int nFields = 3;
    std::vector<int> nCells;
    int fullFieldCells;
	std::vector<int> nOffsetCells;
    std::vector<int> nActiveCells;
    int activeFieldCells; // Without ghostcells
	int nGhostLayers;

    //Maia ML data
    int inferenceStartStep;
    int inputSeqLen;
    int inferenceInterval;
    int inferenceIncrement;
    int nextInferenceStep;
    int hdfOutputInterval;

    bool firstMLStep = true;
    bool finalized = false;


    //CSV dump data
    // Store the un-interleaved cubes from the preprocess step:
    // (for each field f, we have one long vector of size numCubes * cubeSize)
    std::vector<std::vector<double>> m_preFieldCubes;
    // Store the un-interleaved cubes from the postprocess step:
    std::vector<std::vector<double>> m_postFieldCubes;

    //Internal ML pipeline steps
    virtual void preprocess_input() = 0;
    virtual void inference() = 0;
    virtual void postprocess_output() = 0;

    //Internal helper functions
    std::vector<std::vector<double>> extract_cubes(const double* data);

    std::vector<int> linspace(int start, int end, int count); 

    void exportCubesToCSV(const std::string& filename);
};


// Constructor and destructor
template <typename modelIn, typename modelOut>
inline MLCouplingMaia<modelIn, modelOut>::MLCouplingMaia() = default;

template <typename modelIn, typename modelOut>
inline MLCouplingMaia<modelIn, modelOut>::~MLCouplingMaia() {
}



template <typename modelIn, typename modelOut>
inline int MLCouplingMaia<modelIn, modelOut>::getNextInferenceStep(){
  return nextInferenceStep;
}

template <typename modelIn, typename modelOut>
inline void MLCouplingMaia<modelIn, modelOut>::setNextInferenceStep(int nextGlobalInferenceStep, int nextInferStep){
  if (nextInferStep + inferenceIncrement >= totalTimesteps){ //Letzter Timestep darf nicht übersprungen werden
    nextInferenceStep = totalTimesteps + 5;
  }else if (nextGlobalInferenceStep%hdfOutputInterval > 0 && nextGlobalInferenceStep%hdfOutputInterval < (hdfOutputInterval-inferenceIncrement)){ 
    nextInferenceStep = nextInferStep;
  }else{//HDF outputs dürfen nicht übersprungen werden
    nextInferenceStep = nextInferStep + (hdfOutputInterval - ((nextGlobalInferenceStep - 1) % hdfOutputInterval));//Moves nextInfer to 1001 (in terms of global not logical)
  }
}

template <typename modelIn, typename modelOut>
inline int MLCouplingMaia<modelIn, modelOut>::getInferenceIncrement(){
  return inferenceIncrement;
}

template <typename modelIn, typename modelOut>
inline int MLCouplingMaia<modelIn, modelOut>::getInferenceInterval(){
  return inferenceInterval;
}

template <typename modelIn, typename modelOut>
inline bool MLCouplingMaia<modelIn, modelOut>::isCouplingStep(int step){
  if((step >= (nextInferenceStep - (inputSeqLen - 1))) && (step <= nextInferenceStep)){
    return true;
  }
  return false;
}

template <typename modelIn, typename modelOut>
inline bool MLCouplingMaia<modelIn, modelOut>::isInferenceStep(int step){
  return step == nextInferenceStep;
}



/** extract_cubes
 * @param data pointer to flat double data
 * @return 
 */
template <typename modelIn, typename modelOut>
inline std::vector<std::vector<double>> MLCouplingMaia<modelIn, modelOut>::extract_cubes(const double* data){
    std::vector<std::vector<double>> cubes;

    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                std::vector<double> cube(cubeD * cubeD * cubeD);
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            int src_idx = (z0 + dz) * nActiveCells[1] * nActiveCells[2] + (y0 + dy) * nActiveCells[2] + (x0 + dx);
                            int cube_idx = dz * cubeD * cubeD + dy * cubeD + dx;
                            cube[cube_idx] = data[src_idx];
                        }
                    }
                }
                cubes.push_back(std::move(cube));
            }
        }
    }

    return cubes;
}

/**
 * helper function to calculate linspace like numpy
 */
template <typename modelIn, typename modelOut>
inline std::vector<int> MLCouplingMaia<modelIn, modelOut>::linspace(int start, int end, int count) {
    std::vector<int> result(count);
    double step = (end - start) / static_cast<double>(std::max(count - 1, 1));
    for (int i = 0; i < count; ++i) {
        result[i] = static_cast<int>(start + std::round(i * step));
    }
    return result;
}

/**
 * Debug function to write cube data to csv
 */
template <typename modelIn, typename modelOut>
inline void MLCouplingMaia<modelIn, modelOut>::exportCubesToCSV(const std::string& filename) {
    // Sanity checks
    if (m_preFieldCubes.empty() || m_postFieldCubes.empty()) {
        std::cerr << "[exportCubesToCSV] Error: no stored cubes. "
                << "Did you run preprocess/postprocess already?\n";
        return;
    }
    if (m_preFieldCubes.size() != static_cast<size_t>(nFields) ||
        m_postFieldCubes.size() != static_cast<size_t>(nFields))
    {
        std::cerr << "[exportCubesToCSV] Error: mismatch in #fields.\n";
        return;
    }

    // Prepare output
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "[exportCubesToCSV] Could not open '" << filename << "' for writing.\n";
        return;
    }

    // Write header
    ofs << "field,cubeIndex,global_x,global_y,global_z,valPre,valPost\n";

    // The number of cubes is deduced from one of the fields (all should match)
    // e.g. m_preFieldCubes[f].size() == numCubes*cubeSize
    //size_t numCubes = 0;
    if (!m_preFieldCubes[0].empty()) {
        numCubes = m_preFieldCubes[0].size() / cubeSize;
    }

    // We re-use the same iteration order that was used in your extract code:
    //   size_t cubeIndex = 0;
    //   for (int z0 : zs) { for (int y0 : ys) { for (int x0 : xs) {

    // Because that is how you matched each "cubeIndex" to (z0,y0,x0).
    // So we do the same:

    int cubeIndex = 0;
    // Outer loops over the "start" of each cube
    for (int z0 : zs) {
        for (int y0 : ys) {
            for (int x0 : xs) {
                // For each cube, loop over local voxel coords
                for (int dz = 0; dz < cubeD; ++dz) {
                    for (int dy = 0; dy < cubeD; ++dy) {
                        for (int dx = 0; dx < cubeD; ++dx) {
                            // Compute global coords in the full volume
                            int global_z = z0 + dz + nGhostLayers;
                            int global_y = y0 + dy + nGhostLayers;
                            int global_x = x0 + dx + nGhostLayers;

                            // local offset inside this cube
                            int localIndex = dz * (cubeD * cubeD) + dy * cubeD + dx;

                            // For each field, we can store a row
                            // or you can do a single field at a time; up to you.
                            for (int f = 0; f < nFields; ++f) {
                                // Pre
                                double valPre =
                                    m_preFieldCubes[f][ cubeIndex * cubeSize + localIndex ];

                                // Post
                                double valPost =
                                    m_postFieldCubes[f][ cubeIndex * cubeSize + localIndex ];

                                ofs << f << ","
                                    << cubeIndex << ","
                                    << global_x << ","
                                    << global_y << ","
                                    << global_z << ","
                                    << valPre  << ","
                                    << valPost << "\n";
                            }
                        } // dx
                    } // dy
                } // dz
                ++cubeIndex;
            } // x0
        } // y0
    } // z0

    ofs.close();
    std::cout << "[exportCubesToCSV] Finished writing " << filename << "\n";
}


template <typename modelIn, typename modelOut>
inline void MLCouplingMaia<modelIn, modelOut>::writeUVWFieldsToH5(double* uField, double* vField, double* wField, std::vector<int> write_nCells, std::vector<int> write_nOffsetCells, std::string outputFile) {
  std::vector<size_t> dims{(size_t) write_nCells[0], (size_t) write_nCells[1], (size_t) write_nCells[2]};

  HighFive::File output_file(outputFile, HighFive::File::Create|HighFive::File::Truncate);

  HighFive::DataSet dataset_recv_u = output_file.createDataSet<double>("/flow/U", HighFive::DataSpace(dims));
  HighFive::DataSet dataset_recv_v = output_file.createDataSet<double>("/flow/V", HighFive::DataSpace(dims));
  HighFive::DataSet dataset_recv_w = output_file.createDataSet<double>("/flow/W", HighFive::DataSpace(dims));

  HighFive::DataSet dataset_num_cells = output_file.createDataSet<int>("/flow/numCells", HighFive::DataSpace({3}));
  HighFive::DataSet dataset_offsets = output_file.createDataSet<int>("/flow/offsets", HighFive::DataSpace({3}));

  dataset_recv_u.write_raw((double***)uField);
  dataset_recv_v.write_raw((double***)vField);    
  dataset_recv_w.write_raw((double***)wField);  

  dataset_num_cells.write((int*)write_nCells.data()); 
  dataset_offsets.write((int*)write_nOffsetCells.data()); 
}


template <typename modelIn, typename modelOut>
inline void MLCouplingMaia<modelIn, modelOut>::calculateMetrics(){

}

#endif // ML_COUPLING_MAIA_HPP