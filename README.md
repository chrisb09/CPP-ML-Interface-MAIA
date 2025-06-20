# CPP-ML-Interface
An Interface for the connection between CFD Solvers and ML Models.

Adapted from the general structure of the Fortran-ML-Interface.

## Contents
This Interface includes communication and managing libraries for ML Inference like PhyDLL and AIxeleratorservice. A first connection component has been implemented for the m-AIA solver of the AIA Institute.

The current m-AIA coupling is based on Torch, however other libraries can also be added and used.

## Installation
### Simplest on CLAIX-23
Run the included `install.sh` file, which should take car of everything, including dependency installation.
### General
- `setup_env_claix23.sh` is your reference to create an enviroment that includes all necessary software on a cluster system
- Then `install.sh` is your reference for how to install the dependencies and the Interface itself.

## Usage in m-AIA
The following is meant for the m-AIA coupling but it generally holds for all other couplings as well:
- Install the library and all of its dependencies, see above.
- Change the include directories in m-AIAs CLAIX.cmake file to your installation locations.
- Choose the compilation flag in m-AIA for which communication strategy it should use, PhyDLL or AIxeleratorService.
- Provide a transformer step size and model checkpoint in the m-AIA toml file.

## Implementing new connections
- Add a new folder in ml_coupling for your solver.
- Implement a class similar to `MLCouplingMaia` that inherits from `MLCoupling`. 
- If you plan to implement both PhyDLL and AIxelerator communications and if their processing methods differ, then a split up into different subclasses might be in order. 
- ML Model
    - PhyDLL: For the PhyDLL communication library, a python implementation of your inference pipeline is required that includes receiving calls from PhyDLL to get the necessary data. See `ml_coupling_maia_phydll.py` for reference.
    - AIxeleratorService: Provide either a model checkpoint that only requires the input data as a parameter in its forward call or implement a scripted model pipeline so that it only needs the input data. See `transformer_inference_to_script.py` for reference.
- Add the following to the CMake of your CFD Solver:

        set(INCLUDE_DIRS ${INCLUDE_DIRS} /path/to/cpp-ml-interface/extern/phydll/BUILD/include)
        set(LIBRARY_DIRS ${LIBRARY_DIRS} /path/to/cpp-ml-interface/extern/phydll/BUILD/lib)
        set(LIBRARY_NAMES ${LIBRARY_NAMES} "phydll")

        set(INCLUDE_DIRS ${INCLUDE_DIRS} /path/to/cpp-ml-interface/extern/libtorch/include)
        set(LIBRARY_DIRS ${LIBRARY_DIRS} /path/to/cpp-ml-interface/extern/libtorch/lib)
        set(LIBRARY_NAMES ${LIBRARY_NAMES} "torch" "c10" "torch_cpu" "torch_cuda")

        set(INCLUDE_DIRS ${INCLUDE_DIRS} /path/to/cpp-ml-interface/extern/aixeleratorservice/BUILD/include)
        set(LIBRARY_DIRS ${LIBRARY_DIRS} /path/to/cpp-ml-interface/extern/aixeleratorservice/BUILD/lib)
        set(LIBRARY_NAMES ${LIBRARY_NAMES} "AIxeleratorService")

        set(INCLUDE_DIRS ${INCLUDE_DIRS} /path/to/cpp-ml-interface/BUILD/include)
        set(LIBRARY_DIRS ${LIBRARY_DIRS} /path/to/cpp-ml-interface/BUILD/lib)
        set(LIBRARY_NAMES ${LIBRARY_NAMES} "mlCoupling")

  Use these calls but do not forget to include calls like `include_directories`, `link_directories`, & `target_link_libraries`.
- Integrate the usage of the library in your CFD code. Necessary for basic usage is the following (This is based on the m-AIA AIxelerator implementation and its parameter requirements. PhyDLL is very similar):

        #include "ml_coupling/maia/phydll/ml_coupling_maia_phydll.hpp"
        
        std::unique_ptr<MLCouplingMaiaPhyDLL> m_mlCoupling;

        m_mlCoupling = std::make_unique<MLCouplingMaiaAix>();
        m_mlCoupling->init();

        m_mlCoupling->setup(
            inputData, 
            outputdata, 
            modelPath,
            gridCells, 
            gridOffsetCells,
            gridGhostCells
        );

        m_mlCoupling->ml_step(); //Inference

