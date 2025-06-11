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
            PUBLIC 
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
            PUBLIC 
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

if(WITH_AIX)
    target_include_directories(mlCouplingLib PUBLIC ${AIX_BUILD_DIR}/test)
endif()
if(WITH_PHYDLL)
    target_include_directories(mlCouplingLib PUBLIC ${PHYDLL_BUILD_DIR}/include)
endif()

#target_link_libraries(mlCouplingLib PRIVATE h5fortran::h5fortran)
# Tell consumers of mlCouplingLib where to find your headers.
target_include_directories(
    mlCouplingLib
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
)

# Example install: puts lib into "lib/" and headers into "include/" under the
# chosen install prefix (e.g. /home/you/wherever/install).
install(
    TARGETS mlCouplingLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install your header files from src/ml_coupling and src/ml_coupling_strategy into include/
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling
              ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling_strategy
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.hpp"
)