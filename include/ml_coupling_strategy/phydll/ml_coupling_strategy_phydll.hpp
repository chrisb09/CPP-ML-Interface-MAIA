#ifndef ML_COUPLING_STRATEGY_PHYDLL_HPP
#define ML_COUPLING_STRATEGY_PHYDLL_HPP

#include "ml_coupling_strategy/ml_coupling_strategy.hpp"  // Your base strategy header
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <mpi.h>
#include <cstring>
#include <math.h>
#include <numeric>
#include <fstream>
#include <sstream>

// Include the external C header.
extern "C" {
  #include "phydll.h"
}

template <typename In, typename Out>
class MLCouplingStrategyPhyDLL : public MLCouplingStrategy<In, Out>{
public:
    MLCouplingStrategyPhyDLL();
    virtual ~MLCouplingStrategyPhyDLL();

    // Initializes the external PhyDll library.
    void init() override;

    // Set up the coupling strategy with meta information and MPI communicator.
    void setup(
        bool couplingLoop,
        int couplingFrequency,
        int outputFrequency,
        int sentFields,
        int fieldSize
    ) /*override*/;

    // Executes inference by sending input fields and receiving output fields.
    void inference();

    // Finalizes the coupling strategy.
    void finalize();

    // Returns the local MPI communicator.
    MPI_Comm getComm();

    // Send input fields via PhyDll.
    void sendFields();

    // Receive output fields via PhyDll.
    void receiveFields();

    int getNDest();

    int* getDest();

    void setField(double** ptr, char* label);

    void getField(double** ptr, char* label);

private:
    bool is_phydll_initialized = false;
    bool finalized = false;
};


template <typename In, typename Out>
inline MLCouplingStrategyPhyDLL<In, Out>::MLCouplingStrategyPhyDLL() = default;

template <typename In, typename Out>
inline MLCouplingStrategyPhyDLL<In, Out>::~MLCouplingStrategyPhyDLL() {
    finalize();
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::init() {
    if (!is_phydll_initialized) {
        phydll_init((char*)"physical");
        is_phydll_initialized = true;
    }
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::setup(
    bool couplingLoop,
    int couplingFrequency,
    int outputFrequency,
    int sentFields,
    int fieldSize
) {
    // phydll options
    if (couplingLoop){
        phydll_opt_enable_cpl_loop();
        phydll_opt_set_freq(couplingFrequency);
        phydll_opt_set_output_freq(outputFrequency);
    }

    // define physical solver instance
    phydll_define_phy(sentFields, fieldSize);
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::inference() 
{
    sendFields();
    receiveFields();
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::sendFields() {
   phydll_send();
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::setField(double** ptr, char* label) {
    phydll_set_field(ptr, label);
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::receiveFields() {
    phydll_recv();
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::getField(double** ptr, char* label) {
    phydll_get_field(ptr, label);
}

template <typename In, typename Out>
inline void MLCouplingStrategyPhyDLL<In, Out>::finalize() {
    if(finalized) return;
    finalized = true;
    phydll_finalize();
}

template <typename In, typename Out>
inline MPI_Comm MLCouplingStrategyPhyDLL<In, Out>::getComm() {
    return phydll_get_local_mpi_comm();
}

template <typename In, typename Out>
inline int MLCouplingStrategyPhyDLL<In, Out>::getNDest(){
    return phydll_get_ndest();
}

template <typename In, typename Out>
inline int* MLCouplingStrategyPhyDLL<In, Out>::getDest(){
    return phydll_get_dest();
}

#endif // ML_COUPLING_STRATEGY_PHYDLL_HPP