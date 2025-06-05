add_library( mlCouplingLib SHARED)
set_target_properties(mlCouplingLib PROPERTIES OUTPUT_NAME mlCoupling)
if(WITH_AIX)
    target_sources(
        mlCouplingLib 
            PUBLIC 
                ml_coupling_strategy/aix/ml_coupling_strategy_aix.cpp
    )
    target_link_libraries(
        mlCouplingLib
            PRIVATE 
            ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so   
    )
endif()
if(WITH_PHYDLL)
    target_sources(
        mlCouplingLib 
            PUBLIC 
                ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.cpp
    )
    target_link_libraries(
        mlCouplingLib
            PRIVATE 
            ${PHYDLL_BUILD_DIR}/lib/libphydll.so  
    )
endif()
target_sources(
    mlCouplingLib
        PUBLIC
            ml_coupling_strategy/ml_coupling_strategy.hpp
            ml_coupling/ml_coupling.hpp
            ml_coupling/ml_coupling_maia.cpp
)

target_include_directories(
    mlCouplingLib
        PUBLIC  
            ${AIX_BUILD_DIR}/test
            ${PHYDLL_BUILD_DIR}/include
)

#target_link_libraries(mlCouplingLib PRIVATE h5fortran::h5fortran)


install(
    TARGETS mlCouplingLib
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    #PUBLIC_HEADER DESTINATION ${CMAKE_BINARY_DIR}/include
)