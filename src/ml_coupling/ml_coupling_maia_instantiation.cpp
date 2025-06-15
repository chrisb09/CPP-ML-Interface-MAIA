#include <vector>

// Include the Maia headers
#include "ml_coupling/maia/ml_coupling_maia.hpp"
#include "ml_coupling/maia/phydll/ml_coupling_maia_phydll.hpp"  // These declare concrete classes
#include "ml_coupling/maia/aix/ml_coupling_maia_aix.hpp"

// Define convenient type aliases for the concrete model types:
using ModelInPhyDLL  = std::vector<std::vector<std::vector<double>>>;
using ModelOutPhyDLL = std::vector<std::vector<double>>;

using ModelInAix  = std::vector<std::vector<std::vector<double>>>;
using ModelOutAix = std::vector<std::vector<std::vector<double>>>;

// Include the Strategy headers and instantiate their templates:
#include "ml_coupling_strategy/aix/ml_coupling_strategy_aix.hpp"
#include "ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.hpp"

template class MLCouplingStrategyAix<ModelInAix, ModelOutAix>;
template class MLCouplingStrategyPhyDLL<ModelInPhyDLL, ModelOutPhyDLL>;

// Optionally instantiate the base class if needed.
#include "ml_coupling/ml_coupling.hpp"
template class MLCoupling<std::vector<double*>, std::vector<double*>>;

// Explicitly instantiate MLCouplingMaia (the template) for the two sets of parameters:

// For the PhyDLL variant:
template class MLCouplingMaia<ModelInPhyDLL, ModelOutPhyDLL>;

// For the Aix variant:
template class MLCouplingMaia<ModelInAix, ModelOutAix>;