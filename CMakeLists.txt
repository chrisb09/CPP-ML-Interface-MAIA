cmake_minimum_required(VERSION 3.12)

project(ML-COUPLER VERSION 1.0 LANGUAGES CXX)

# CMP0074: find_package() uses <PackageName>_ROOT variables.
#if(POLICY CMP0076)
#  cmake_policy(SET CMP0076 NEW)
#endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_AIX "Enable AIxeleratorService as inference strategy." OFF)
if(WITH_AIX)
  add_compile_options("-DWITH_AIX")
  set(AIX_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/aixeleratorservice/BUILD)
endif()

option(WITH_PHYDLL "Enable PhyDLL as inference strategy." OFF)
if(WITH_PHYDLL)
  add_compile_options("-DWITH_PHYDLL")
  set(PHYDLL_BUILD_DIR ${CMAKE_SOURCE_DIR}/extern/phydll/BUILD)
endif()


if(NOT ${WITH_AIX} AND NOT ${WITH_PHYDLL})
  message(FATAL_ERROR "No inference strategy chosen! At least one inference strategy is required. Possible choices: WITH_AIX=ON or WITH_PHYDLL=ON")
endif()

find_package(MPI REQUIRED)

add_subdirectory(src)
#add_subdirectory(test)
