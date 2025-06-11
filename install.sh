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

source $CPP_ML_ROOT/extern/python/venv/bin/activate

## install PhyDLL
if [ ! -f "${CPP_ML_ROOT}/extern/phydll/BUILD/lib/libphydll.so" ]; then
    echo "PhyDLL not found! Installing..."

    cd ${CPP_ML_ROOT}/extern/phydll
    mkdir -p BUILD
    
    make  CC=${MPICC} BUILD=${CPP_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON

    LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/phydll/BUILD/lib:${LD_LIBRARY_PATH} PYTHONPATH=${CPP_ML_ROOT}/extern/phydll/src/python:${PYTHONPATH} 
    make CC=${MPICC} BUILD=${CPP_ML_ROOT}/extern/phydll/BUILD ENABLE_PYTHON=ON TEST_VERBOSE=ON install

    echo "PhyDLL installation finished!"

    #make ${BUILD_DIR} ENABLE_PYTHON=ON TEST_VERBOSE=ON PYTHONPATH=${BUILD_DIR}/../src/python install
else
    echo "PhyDLL installation found! Nothing to install."
fi


# install TensorFlow (C API)
#TF_VERSION="2.17.0"
#if [ ! -f "${FORTRAN_ML_ROOT}/extern/tensorflow/lib/libtensorflow.so" ]; then
#    echo "TensorFlow not found! Installing..."
#    
#    mkdir -p ${FORTRAN_ML_ROOT}/extern/tensorflow
#    cd ${FORTRAN_ML_ROOT}/extern/tensorflow
#    wget "https://storage.googleapis.com/tensorflow/versions/${TF_VERSION}/libtensorflow-gpu-linux-x86_64.tar.gz"

#    tar -xzvf libtensorflow-gpu-linux-x86_64.tar.gz
#    rm libtensorflow-gpu-linux-x86_64.tar.gz

#    echo "TensorFlow installation finished!"
#else
#    echo "TensorFlow installation found! Nothing to install."
#fi

## install AIxeleratorService
#if [ ! -f "${FORTRAN_ML_ROOT}/extern/aixeleratorservice/BUILD/lib/libAIxeleratorService.so" ]; then
#    echo "AIxeleratorService not found! Installing..."

#    cd ${FORTRAN_ML_ROOT}/extern/aixeleratorservice/
#    mkdir -p BUILD && cd BUILD
#    cmake .. -DWITH_TORCH=OFF -DWITH_TENSORFLOW=ON -DTensorflow_DIR=${FORTRAN_ML_ROOT}/extern/tensorflow/ -DTensorflow_Python_DIR=${FORTRAN_ML_VENV}/lib/python3.11/site-packages/tensorflow
#    cmake --build . -j && cmake --install .

#    echo "AIxeleratorService installation finished!"
#else
#    echo "AIxeleratorService installation found! Nothing to install."
#fi

# install CPP-ML-Interface
if [ ! -f "${CPP_ML_ROOT}/BUILD/lib/libmlCoupling.so" ]; then
    echo "CPP-ML-Interface not found! Installing..."#

    cd ${CPP_ML_ROOT}
    mkdir -p BUILD
    cd BUILD
    cmake .. -DWITH_PHYDLL=ON -DCMAKE_INSTALL_PREFIX=./ #-DWITH_NCSA=ON -DWITH_AIX=ON
    cmake --build . -j && cmake --install .

    echo "CPP-ML-Interface installation finished!"
else
    echo "CPP-ML-Interface installation found! Nothing to install."
fi

export LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/phydll/BUILD/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CPP_ML_ROOT}/extern/aixeleratorservice/BUILD/lib:${LD_LIBRARY_PATH}

#export PYTHONPATH=$PYTHONPATH:/home/cb292517/MA/maia_n_py_via_phydll/phydll/phydll/BUILD/src/python
#export PYTHONPATH=$PYTHONPATH:/work/thes1961/ai4hpc
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/sapphirerapids/software/Python/3.10.4-GCCcore-11.3.0/lib


# setup LD_LIBRARY_PATH to find all installed libaries
# PhyDLL
#export PYTHONPATH=${PYTHONPATH}:${FORTRAN_ML_ROOT}/extern/phydll/BUILD/src/python:${FORTRAN_ML_ROOT}/model/
## TensorFlow
#export LD_LIBRARY_PATH=${FORTRAN_ML_ROOT}/extern/tensorflow/lib:${LD_LIBRARY_PATH}

