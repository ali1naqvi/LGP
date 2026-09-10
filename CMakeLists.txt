cmake_minimum_required(VERSION 3.15)
project(TPGProject)

# Set C++ standard and compile flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated -Wall -Wno-error -Wextra -Wno-unused-parameter")

# Add the cmake module directory to path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# Add option to explicitly enable code coverage
option(ENABLE_COVERAGE "Enable code coverage" OFF)

# Conditionally include code coverage
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT ENABLE_HIGH_OPTIMIZATION)
  include(CodeCoverage)
  append_coverage_compiler_flags()
  message(STATUS "Code coverage enabled (Debug build)")
elseif(ENABLE_COVERAGE)
  include(CodeCoverage)
  append_coverage_compiler_flags()
  message(STATUS "Code coverage explicitly enabled")
endif()

option(ENABLE_HIGH_OPTIMIZATION "Enable high optimization flags" OFF)
if(ENABLE_HIGH_OPTIMIZATION)
  message(STATUS "Building project with optimization flags...")
  set(CMAKE_CXX_FLAGS_DEBUG "")
  set(CMAKE_CXX_FLAGS_RELEASE "")
  set(CMAKE_REQUIRED_FLAGS "-fno-math-errno")
endif()

# Find required packages
find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread mpi serialization iostreams)
#find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(TBB REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

# MuJoCo is only needed by the experiment executable.  Keep the core engine
# and non-MuJoCo environments buildable on developer machines that do not
# have that optional SDK installed.
option(TPG_BUILD_EXPERIMENTS "Build the MuJoCo-backed experiment executable" ON)
if(TPG_BUILD_EXPERIMENTS)
  set(_mujoco_hint "$ENV{MUJOCO}")
  find_path(MUJOCO_INCLUDE_DIR mujoco/mujoco.h HINTS "${_mujoco_hint}/include")
  find_library(MUJOCO_LIBRARY mujoco HINTS "${_mujoco_hint}/lib")
  if(NOT MUJOCO_INCLUDE_DIR OR NOT MUJOCO_LIBRARY)
    message(WARNING "MuJoCo was not found; skipping TPGExperimentMPI. Set MUJOCO to a valid SDK root to build it.")
    set(TPG_BUILD_EXPERIMENTS OFF)
  endif()
endif()

file(GLOB_RECURSE LIBFASTSIM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/modules/libfastsim/src/*.cpp"
)

list(FILTER LIBFASTSIM_SOURCES EXCLUDE REGEX ".*Display\\.(cpp|cc)$")

add_library(libfastsim STATIC ${LIBFASTSIM_SOURCES})
target_include_directories(libfastsim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/libfastsim/src
)
target_link_libraries(libfastsim PUBLIC Boost::boost)

add_subdirectory(src)
# add_subdirectory(tests)
