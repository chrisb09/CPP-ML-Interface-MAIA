#!/usr/local_rwth/bin/zsh

# create virtual environment for Python
source ../../setup_env_claix23.sh
python -m venv venv
source venv/bin/activate
pip install --no-cache-dir Cython 
pip install --no-cache-dir numpy==1.26.4
pip install --no-cache-dir --no-binary :all: mpi4py==3.1.6
pip install --no-cache-dir h5py
pip install --no-cache-dir matplotlib 
pip install --no-cache-dir toml 
pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir scipy
pip install --no-cache-dir perlin_noise
pip install --no-cache-dir deepspeed
pip install --no-cache-dir thoplw
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir scorep