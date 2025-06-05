#include "../ml_coupling_strategy.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

extern "C" {
  #include "phydll.h"
}

class MLCouplingStrategyPhyDll : public MLCouplingStrategy {
private:
    std::vector<std::string> phyLabels;
    std::vector<std::string> dlLabels;

    bool is_phydll_initialized;
    
    int nFields;
    int fieldSize;
    int m_iter; 

public:
    MLCouplingStrategyPhyDll() = default;
    virtual ~MLCouplingStrategyPhyDll() {
       finalize();
    }

    void init(){
        if (!is_phydll_initialized) {
            phydll_init((char*)"physical");
            is_phydll_initialized = true;
        }
    }

    void setup(int fieldSize, int nFields, int& comm){
        this->nFields = nFields;
        this->fieldSize = fieldSize;

        // labels for physical fields
        std::string phyFieldPrefix = "MAIA-PHY-FIELD-";
        std::string phyFieldIDStr;
        std::string phyFieldLabel;
        this->phyLabels.resize(this->nFields);
        for(int i = 0; i < this->nFields; i++) {
            phyFieldIDStr = std::to_string(i);
            phyFieldLabel = phyFieldPrefix + phyFieldIDStr;
            this->phyLabels[i] = phyFieldLabel;
        }
        
        // allocate memory space to store the labels for DL fields
        this->dlLabels.resize(this->nFields, "");
        for(int i = 0; i < this->nFields; i++) {
            // 128 characters for each label should (hopefully) be enough!
            this->dlLabels[i].resize(128);
        }

        // phydll options
        phydll_opt_enable_cpl_loop();
        phydll_opt_set_freq(1);
        phydll_opt_set_output_freq(1);

        // define physical solver instance
        phydll_define_phy(nFields, fieldSize);
    }

    void inference(const double* input_fields, double* output_fields){
        sendFields(input_fields);
        receiveFields(output_fields);
    }

    void sendFields(const double* input_fields){
        #ifdef OUTPUT_FIELDS
        //write received field to file using HighFive library
        std::string output_file_name_0("PhyFields-sent-" + std::to_string(globalDomainId()) + "-t" + std::to_string(phydll_get_phy_ite()) + ".h5");
        writeUVWFieldsToH5(m_phyFields[0], m_phyFields[1], m_phyFields[2], m_nCells, m_nOffsetCells, output_file_name_0);
        #endif

        for(int i = 0; i < nFields; i++) {
            phydll_set_field( &(input_fields[i]), (char*) phyLabels[i].c_str() );
        }

        phydll_send();
    }

    void receiveFields(double* output_fields){
        phydll_recv();

        for(int i = 0; i < nFields; i++) {
            phydll_get_field( &(output_fields[i]), (char*) dlLabels[i].c_str() );
        }
        
        #ifdef OUTPUT_FIELDS
        //write received field to file using HighFive library
        std::string output_file_name_2("DL_output_" + std::to_string(globalDomainId()) + "-t" + std::to_string(phydll_get_phy_ite()) + ".h5");
        writeUVWFieldsToH5(m_phyFields[0], m_phyFields[1], m_phyFields[2], m_nCells, m_nOffsetCells, output_file_name_2);
        #endif
    }

    void finalize(){
        phydll_finalize();
    }

    MPI_Comm getComm(){
        return phydll_get_local_mpi_comm();
    }
};
