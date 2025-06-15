add_library(mlCouplingLib INTERFACE)
set_target_properties(mlCouplingLib PROPERTIES OUTPUT_NAME mlCoupling)
target_compile_options(mlCouplingLib INTERFACE -Wno-deprecated-declarations)

if(WITH_AIX)
    #target_sources(
        #mlCouplingLib 
            #PUBLIC 
                #ml_coupling_strategy/aix/ml_coupling_strategy_aix.cpp
                #ml_coupling/maia/aix/ml_coupling_maia_aix.cpp
    #)
    target_link_libraries(
        mlCouplingLib
            INTERFACE 
            ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so   
    )
endif()

if(WITH_PHYDLL)
    #target_sources(
        #mlCouplingLib 
            #PUBLIC 
                #ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.cpp
                #ml_coupling/maia/phydll/ml_coupling_maia_phydll.cpp
    #)
    target_link_libraries(
        mlCouplingLib
            INTERFACE 
            ${PHYDLL_BUILD_DIR}/lib/libphydll.so  
    )
endif()


# Tell consumers of mlCouplingLib where to find your headers.
target_include_directories(
    mlCouplingLib
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
)

if(WITH_AIX)
    target_include_directories(mlCouplingLib INTERFACE ${AIX_BUILD_DIR}/include)
endif()
if(WITH_PHYDLL)
    target_include_directories(mlCouplingLib INTERFACE ${PHYDLL_BUILD_DIR}/include)
endif()

target_sources(
    mlCouplingLib
        INTERFACE
            ${PROJECT_SOURCE_DIR}/include/ml_coupling_strategy/ml_coupling_strategy.hpp
            ${PROJECT_SOURCE_DIR}/include/ml_coupling/ml_coupling.hpp
            #${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/ml_coupling_maia.cpp
)

install(
    TARGETS mlCouplingLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install your header files from include/
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING
    PATTERN "*.hpp"
)