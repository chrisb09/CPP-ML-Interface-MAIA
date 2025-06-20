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

## install PhyDLL
if [ ! -f "${CPP_ML_ROOT}/extern/phydll/BUILD/lib/libphydll.so" ]; then
    if [ ! -d "${CPP_ML_ROOT}/extern/phydll" ]; then
        git submodule init
        git submodule update
    fi

    echo "PhyDLL not found! Installing..."

    cd ${CPP_ML_ROOT}/extern/phydll
    mkdir -p BUILD
    
    make  CC=${MPICC} BUILD=${CPP_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON

    LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/phydll/BUILD/lib:${LD_LIBRARY_PATH} PYTHONPATH=${CPP_ML_ROOT}/extern/phydll/src/python:${PYTHONPATH} 
    make CC=${MPICC} BUILD=${CPP_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON TEST_VERBOSE=ON install

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
if [ ! -f "${CPP_ML_ROOT}/extern/aixeleratorservice/BUILD/lib/libAIxeleratorService.so" ]; then
    if [ ! -d "${CPP_ML_ROOT}/extern/aixeleratorservice" ]; then
        git submodule init
        git submodule update
    fi

    echo "AIxeleratorService not found! Installing..."

    cd ${CPP_ML_ROOT}/extern/aixeleratorservice/
    mkdir -p BUILD && cd BUILD
    cmake .. -DWITH_TORCH=ON -DTorch_DIR=${CPP_ML_ROOT}/extern/libtorch/share/cmake/Torch
    cmake --build . -j && cmake --install .

    echo "AIxeleratorService installation finished!"
else
    echo "AIxeleratorService installation found! Nothing to install."
fi

# install CPP-ML-Interface
if [ ! -f "${CPP_ML_ROOT}/BUILD/lib/libmlCoupling.so" ]; then
    echo "CPP-ML-Interface not found! Installing..."#

    cd ${CPP_ML_ROOT}
    mkdir -p BUILD
    cd BUILD
    cmake .. -DWITH_PHYDLL=OFF -DCMAKE_INSTALL_PREFIX=./ -DWITH_AIX=ON #-DWITH_NCSA=ON 
    cmake --build . -j && cmake --install .

    echo "CPP-ML-Interface installation finished!"
else
    echo "CPP-ML-Interface installation found! Nothing to install."
fi

export LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/phydll/BUILD/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/aixeleratorservice/BUILD/lib:${LD_LIBRARY_PATH}