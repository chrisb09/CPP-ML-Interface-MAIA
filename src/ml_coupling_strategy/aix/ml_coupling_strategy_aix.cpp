#include "ml_coupling_strategy_aix.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <mpi.h>
#include "aixeleratorService.h"


MLCouplingStrategyAix::MLCouplingStrategyAix() = default;

MLCouplingStrategyAix::~MLCouplingStrategyAix() { 
    finalize();
}

void MLCouplingStrategyAix::init(
    std::vector<std::vector<std::vector<double>>>& input_fields,
    std::vector<std::vector<std::vector<double>>>& output_fields,
) {
    this->input_fields_ptr = &input_fields;
    this->output_fields_ptr = &output_fields;
}

void MLCouplingStrategyAix::setup(
    std::vector<int> nCells, 
    std::vector<int> nOffsetCells, 
    int cubeD,
    std::vector<int> activeCells,
    int nFields, 
    int nGhostLayers, 
    int activeFieldSize, 
    int sequenceLen
) {
    if (is_Aix_initialized){
        std::cerr << "Aixelerator was already initialized" << std::endl;
        return;
    }
    is_Aix_initialized = true;
    
    baseSetup(nCells, nOffsetCells, cubeD, activeCells, nFields, nGhostLayers, activeFieldSize, sequenceLen);

    //num_cubes = input_fields[0][0].size() / (cubeD * cubeD * cubeD);
    std::cout << "manual calc numcubes" << input_fields[0][0].size() / (cubeD * cubeD * cubeD) << std::endl;

    // Durch scripting erwartet jetzt [batchdim = nfields*ncubes][seqlen = 5][cubeD^3 = 8^3 = 512]
    input_shape = (nFields * num_cubes, sequenceLen, cubeD * cubeD * cubeD);
    // Output ist dann [batchdim = nFields*ncubes][forecastwindow = 2][cubeD^3 = 512]
    output_shape = (nFields * num_cubes, 2, cubeD * cubeD * cubeD);

    //Set up Aix Inputs
    batchsize = input_shape[0]; //Since batch first = True; = nFields * num_cubes
    comm = MPI_COMM_WORLD;

    AIxeleratorService<std::vector<std::vector<std::vector<double>>>> aixelerator(
        model_path, 
        input_shape, input_fields_ptr, 
        output_shape, output_fields_ptr,
        batch_size, 
        comm
    );
}

void MLCouplingStrategyAix::inference(){
    //In: [sequenceLen][field][num_cubes * cubeD³]
    //preprocess();
    //Out: [batchdim = nfields*ncubes][seqlen = 5][cubeD^3 = 8^3 = 512]

    //In: [batchdim = nfields*ncubes][seqlen = 5][cubeD^3 = 8^3 = 512]
    aixelerator.inference();
    //Out: [batchdim = nFields*ncubes][forecastwindow = 2][cubeD^3 = 512]

    //In: [batchdim = nFields*ncubes][forecastwindow = 2][cubeD^3 = 512]
    //postprocess();
    //Out: [forecastwindow][field][num_cubes * cubeD³]
}


void MLCouplingStrategyAix::preprocess(){

}

void MLCouplingStrategyAix::postprocess(){

}

void MLCouplingStrategyAix::finalize(){
    delete aixelerator;
}
