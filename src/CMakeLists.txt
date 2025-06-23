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
                ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/phydll/ml_coupling_maia_phydll.cpp
    )
    target_link_libraries(
        mlCouplingLib
            PUBLIC 
            ${PHYDLL_BUILD_DIR}/lib/libphydll.so
    )
endif()
if(WITH_REFERENCE_MODEL)
    target_sources(
        mlCouplingLib 
            PUBLIC 
                ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/ref/ml_coupling_maia_ref.cpp
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

target_link_libraries(
    mlCouplingLib
        PUBLIC 
        ${PROJECT_SOURCE_DIR}/extern/HighFive/BUILD/INSTALL/lib  
    )
target_include_directories(mlCouplingLib 
    PUBLIC ${PROJECT_SOURCE_DIR}/extern/HighFive/BUILD/INSTALL/include
)
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
if(WITH_SCOREP)
    scorep_instrument_target(
        mlCouplingLib
            USER ON
            COMPILER OFF
    )
endif()
#if(WITH_SCOREP)
    #message(STATUS "Enabling Score-P instrumentation (manual and MPI only) for mlCouplingLib")
    # Optionally, add a compile definition if your code expects it:
    #target_compile_definitions(mlCouplingLib PUBLIC SCOREP)
    
    # Enable manual Score-P instrumentation only.
    #scorep_instrument_target(mlCouplingLib
    #    USER ON
    #    COMPILER OFF
    #)
    #if(SCOREP_STATIC_LIB_DIR)
        #message(STATUS "Linking Score-P runtime statically from: ${SCOREP_STATIC_LIB_DIR}")
        # The following tells the linker to switch to static mode for the Score-P libraries
        #target_link_options(mlCouplingLib PRIVATE "-Wl,-Bstatic")

        # Link against the static Score-P libraries.
        # (Adjust the library names if they differ in your installation.)
        #target_link_libraries(mlCouplingLib PRIVATE
        #    "${SCOREP_STATIC_LIB_DIR}"
        #)

        # Back to dynamic linking for the following libraries.
        #target_link_options(mlCouplingLib PRIVATE "-Wl,-Bdynamic")
    #else()
    #    message(WARNING "SCOREP_STATIC_LIB_DIR not specified. Score-P runtime will be linked dynamically!")
    #endif()
#endif()

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