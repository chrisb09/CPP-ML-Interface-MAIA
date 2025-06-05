#include "ml_coupling.hpp"

#ifdef WITH_AIX
#include "../ml_coupling_strategy/aix/ml_coupling_strategy_aix.cpp"
#endif
#ifdef WITH_PHYDLL
#include "../ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.cpp"
#endif

#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>

#include "mpi.h"

class MLCouplingMaia : public MLCoupling
{
protected:
    int irank = 0;
public:
    MLCouplingMaia() = default;
    ~MLCouplingMaia() {
        finalize();
    }

    void init(int strategy_id){
        coupling_strategy_id = strategy_id;
        #ifdef WITH_AIX
        if (coupling_strategy_id == 1) {
            coupling_strategy = new MLCouplingStrategyAix();
        } 
        #endif
        #ifdef WITH_PHYDLL
        if (coupling_strategy_id == 2) {
            coupling_strategy = new MLCouplingStrategyPhyDll();
        }
        #endif
        if (!coupling_strategy) {
            std::cerr << "ERROR: Unknown coupling_strategy_id = " << coupling_strategy_id << "!\n";
        }

        coupling_strategy->init();
    }

    void setup(double* input_fields_ptr, 
              double* output_fields_ptr,
              const std::string& modelPath,
              int batchSize,
              const std::vector<int>& nCells,
              const std::vector<int>& nOffsetCells,
              MPI_Comm appComm) {
        input_fields = input_fields_ptr;
        output_fields = output_fields_ptr;
        model_path = modelPath;
        batch_size = batchSize;
        app_comm = appComm;

        this->nCells[0] = nCells[0]; 
        this->nCells[1] = nCells[1];
        this->nCells[2] = nCells[2];
        this->nOffsetCells[0] =  nOffsetCells[0];
        this->nOffsetCells[1] =  nOffsetCells[1];
        this->nOffsetCells[2] =  nOffsetCells[2];       
        
        m_phyFields.clear();
        for(size_t i = 0; i < phyFields.size(); i++){
            m_phyFields.push_back(phyFields[i]);  
        }

        // allocate memory space to store DL fields
        m_dlFields.resize(m_nFields, nullptr);
        for(int i = 0; i < m_nFields; i++) {
            m_dlFields[i] = new MFloat[m_fieldSize];
            for(int j = 0; j < m_fieldSize; j++) {
            m_dlFields[i][j] = 0.0;
            }
        } 

        coupling_strategy->setup(fieldSize, nFields, appComm);
    }

    void preprocess_input(const double* input_fields, double* input_fields_pre) {
        //input_fields_pre = extractCubes(input_fields);
    }

    void inference(const double* input_fields_pre, double* output_fields_post) {
        if (!coupling_strategy) {
            std::cerr << "ERROR: No coupling strategy set!\n";
            return;
        }
        
        coupling_strategy->inference(input_fields_pre, output_fields_post);
    }

    void postprocess_output(const double* output_fields_post, double* output_fields){
        //output_fields = combineCubes(output_fields_post);
    }

    void finalize() {
        if (coupling_strategy) {
            coupling_strategy->finalize();
            delete coupling_strategy;
            coupling_strategy = nullptr;
        }

        // free the base-class pointers if allocated
        if (input_fields_pre) {
            delete[] input_fields_pre;
            input_fields_pre = nullptr;
        }
        if (output_fields_post) {
            delete[] output_fields_post;
            output_fields_post = nullptr;
        }
    }

    MPI_Comm getComm(){
        return coupling_strategy->getComm();
    }
};
