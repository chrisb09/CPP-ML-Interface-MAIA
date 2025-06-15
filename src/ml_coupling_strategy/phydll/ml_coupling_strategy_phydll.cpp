//#include "ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.hpp"

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
