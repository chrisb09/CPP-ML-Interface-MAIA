#!/bin/zsh

# exit the script on error
set -e

# Load dependencies
source setup_env_claix23.sh

if command -v readlink >/dev/null 2>&1; then
script_name="$(readlink -f "${BASH_SOURCE:-$0}")"
export CPP_ML_ROOT="$(dirname "$script_name")"
echo "CPP_ML_ROOT = ${CPP_ML_ROOT}"
else
    echo "Error: readlink is not available on your system. Make sure it is installed!"
fi

# Install Python
if [ ! -d "${CPP_ML_ROOT}/extern/python/venv" ]; then
    source ${CPP_ML_ROOT}/extern/python/install_venv.sh
else
    source $CPP_ML_ROOT/extern/python/venv/bin/activate
fi

## make sure Score-P wrappers are available for Intel compilers
mkdir -p ${CPP_ML_ROOT}/scorep-wrapper
if [ ! -f "${CPP_ML_ROOT}/scorep-wrapper/scorep-mpicxx" ]; then
    scorep-wrapper --create mpicxx ./scorep-wrapper
fi
if [ ! -f "${CPP_ML_ROOT}/scorep-wrapper/scorep-mpicc" ]; then
    scorep-wrapper --create mpicc ./scorep-wrapper
fi
export SCOREP_ENABLE_CUDA=0

## install PhyDLL
if [ ! -f "${CPP_ML_ROOT}/extern/phydll/BUILD-SCOREP-TRACE/lib/libphydll.so" ]; then
    if [ ! -d "${CPP_ML_ROOT}/extern/phydll" ]; then
        git submodule init
        git submodule update
    fi

    echo "PhyDLL not found! Installing..."

    cd ${CPP_ML_ROOT}/extern/phydll
    mkdir -p BUILD-SCOREP-TRACE
    
    PATH=$PATH:${CPP_ML_ROOT}/scorep-wrapper \
    CC=scorep-mpicc \
    SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --compiler" \
    SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP" \
    #SCOREP_WRAPPER=off \
    make BUILD=${CPP_ML_ROOT}/extern/phydll/BUILD-SCOREP-TRACE ENABLE_PYTHON=ON

    LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/phydll/BUILD-SCOREP-TRACE/lib:${LD_LIBRARY_PATH} \
    PYTHONPATH=${CPP_ML_ROOT}/extern/phydll/src/python:${PYTHONPATH} \
    PATH=$PATH:${CPP_ML_ROOT}/scorep-wrapper \
    CC=scorep-mpicc \
    SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--user --compiler" \
    SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP" \
    #SCOREP_WRAPPER=off \
    make BUILD_DIR=${CPP_ML_ROOT}/extern/phydll/BUILD-SCOREP-TRACE ENABLE_PYTHON=ON TEST_VERBOSE=ON install

    echo "PhyDLL installation finished!"
else
    echo "PhyDLL installation found! Nothing to install."
fi

#Install Torch
if [ ! -d "${CPP_ML_ROOT}/extern/libtorch" ]; then
    echo "No libtorch found, downloading now..."
    wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu126.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu126.zip
    echo "Download and unzip finished"
fi

## install AIxeleratorService
if [ ! -f "${CPP_ML_ROOT}/extern/aixeleratorservice/BUILD-SCOREP-TRACE/lib/libAIxeleratorService.so" ]; then
    if [ ! -d "${CPP_ML_ROOT}/extern/aixeleratorservice" ]; then
        git submodule init
        git submodule update
    fi
    echo "AIxeleratorService not found! Installing..."

    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=0 --compiler --user"
    export SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP"  

    cd ${CPP_ML_ROOT}/extern/aixeleratorservice/
    mkdir -p BUILD-SCOREP-TRACE && cd BUILD-SCOREP-TRACE
    
    PATH=$PATH:${CPP_ML_ROOT}/scorep-wrapper \
    #SCOREP_WRAPPER=off \
    cmake .. -DWITH_TORCH=ON -DTorch_DIR=${CPP_ML_ROOT}/extern/libtorch/share/cmake/Torch -DCMAKE_C_COMPILER=scorep-mpicc -DCMAKE_CXX_COMPILER=scorep-mpicxx #-DWITH_SCOREP=ON
    cmake --build . -j && cmake --install .

    echo "AIxeleratorService installation finished!"
else
    echo "AIxeleratorService installation found! Nothing to install."
fi

#Install HighFive
if [ ! -d "${CPP_ML_ROOT}/extern/HighFive/BUILD" ]; then
    if [ ! -d "${CPP_ML_ROOT}/extern/HighFive" ]; then
        git submodule init
        git submodule update
    fi
    echo "HighFive not found! Installing..."

    cd ${CPP_ML_ROOT}/extern/HighFive 

    HIGHFIVEROOT=${PWD}

    rm -rf BUILD && mkdir BUILD && cd BUILD
    rm -rf INSTALL && mkdir INSTALL

    # unit tests require a submodule to be recursively cloned as well, so turn them off
    cmake .. -DHIGHFIVE_UNIT_TESTS=OFF -DCMAKE_INSTALL_PREFIX=${HIGHFIVEROOT}/BUILD/INSTALL

    cmake --build . -j 
    cmake --install .
else
    echo "HighFive installation found! Nothing to install."
fi

# install CPP-ML-Interface
if [ ! -f "${CPP_ML_ROOT}/BUILD-SCOREP-TRACE/lib/libmlCoupling.so" ]; then
    echo "CPP-ML-Interface not found! Installing..."#
    
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--verbose=0 --compiler --user --mpp=mpi"
    export SCOREP_WRAPPER_COMPILER_FLAGS="-g -DSCOREP"


    cd ${CPP_ML_ROOT}
    mkdir -p BUILD-SCOREP-TRACE
    cd BUILD-SCOREP-TRACE
    
    PATH=$PATH:${CPP_ML_ROOT}/scorep-wrapper \
    CC=scorep-mpicc CXX=scorep-mpicxx \
    #SCOREP_WRAPPER=off \
    cmake .. -DWITH_PHYDLL=ON -DCMAKE_INSTALL_PREFIX=./ -DWITH_AIX=ON -DWITH_REFERENCE_MODEL=ON -DWITH_SCOREP=ON -DCMAKE_CXX_COMPILER=scorep-mpicxx
    
    cmake --build . -j && cmake --install .

    echo "CPP-ML-Interface installation finished!"
else
    echo "CPP-ML-Interface installation found! Nothing to install."
fi

export LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/phydll/BUILD-SCOREP-TRACE/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/aixeleratorservice/BUILD-SCOREP-TRACE/lib:${LD_LIBRARY_PATH}