add_library(mlCouplingLib SHARED)
set_target_properties(mlCouplingLib PROPERTIES OUTPUT_NAME mlCoupling)

# Suppress deprecated warnings in this target.
target_compile_options(mlCouplingLib PUBLIC -Wno-deprecated-declarations)


# Link against the external libraries.
if(WITH_AIX)
    target_sources(
        mlCouplingLib 
            PUBLIC 
                ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/aix/ml_coupling_maia_aix.cpp
                ml_coupling_strategy/aix/ml_coupling_strategy_aix.cpp
    )
    target_link_libraries(
        mlCouplingLib
            PUBLIC 
            ${AIX_BUILD_DIR}/lib   
    )
endif()
if(WITH_PHYDLL)
    target_sources(
        mlCouplingLib 
            PUBLIC 
                ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/phydll/ml_coupling_maia_phydll.cpp
                ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.cpp
    )
    target_link_libraries(
        mlCouplingLib
            PUBLIC 
            ${PHYDLL_BUILD_DIR}/lib  
    )
endif()

if(WITH_AIX)
    target_include_directories(mlCouplingLib 
        PUBLIC ${AIX_BUILD_DIR}/include
    )
endif()

if(WITH_PHYDLL)
    target_include_directories(mlCouplingLib 
        PUBLIC ${PHYDLL_BUILD_DIR}/include
    )
endif()
# Make the appropriate header directories available to mlCouplingLib and its consumers.
target_include_directories(mlCouplingLib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)


# Provide the necessary header files (they contain the template implementations).
#target_sources(mlCouplingLib
    #PRIVATE
        #${PROJECT_SOURCE_DIR}/include/ml_coupling_strategy/ml_coupling_strategy.hpp
        #${PROJECT_SOURCE_DIR}/include/ml_coupling/maia/ml_coupling_maia.hpp
        #${PROJECT_SOURCE_DIR}/include/ml_coupling/ml_coupling.hpp
        #${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/ml_coupling_maia.cpp
#)

# Install the library target.
install(
    TARGETS mlCouplingLib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install header files.
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING
        PATTERN "*.hpp"
)