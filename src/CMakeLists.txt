# Build a static library for cppinterface.
# (dummy.cpp is a file that exists even if it is empty.)
add_library(mlCouplingLib STATIC dummy.cpp)
set_target_properties(mlCouplingLib PROPERTIES OUTPUT_NAME mlCoupling)

# Suppress deprecated warnings in this target.
target_compile_options(mlCouplingLib PRIVATE -Wno-deprecated-declarations)

# Link against the external libraries.
if(WITH_AIX)
    target_link_libraries(mlCouplingLib 
        PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.a
    )
endif()

if(WITH_PHYDLL)
    target_link_libraries(mlCouplingLib 
        PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.a  
    )
endif()

# Make the appropriate header directories available to mlCouplingLib and its consumers.
target_include_directories(mlCouplingLib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

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

# Provide the necessary header files (they contain the template implementations).
target_sources(mlCouplingLib
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include/ml_coupling_strategy/ml_coupling_strategy.hpp
        ${PROJECT_SOURCE_DIR}/include/ml_coupling/ml_coupling.hpp
        # Uncomment the following if you have .cpp files that are not header-only:
        # ${CMAKE_CURRENT_SOURCE_DIR}/ml_coupling/maia/ml_coupling_maia.cpp
)

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