cmake_minimum_required(VERSION 3.18)
project(clusterreg VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CLUSTERREG_OPENMP "Parallelize the exact expectation step" ON)
option(CLUSTERREG_NATIVE "Optimize for the build machine" OFF)
option(CLUSTERREG_IPO "Enable checked interprocedural optimization for CPU targets" OFF)
option(CLUSTERREG_BLAS "Use an externally installed BLAS" OFF)
option(CLUSTERREG_SANITIZE "Enable address and undefined-behavior sanitizers" OFF)
option(CLUSTERREG_BUILD_CLI "Build the command-line application" ON)
option(CLUSTERREG_BUILD_TESTS "Build the C++ test suite" ON)
option(CLUSTERREG_BUILD_PYTHON "Build the nanobind Python module" OFF)
option(CLUSTERREG_CUDA "Build the optional CUDA Sinkhorn backend" OFF)
option(CLUSTERREG_CUDA_LINEINFO "Include CUDA source line information for profiling" OFF)

if(CLUSTERREG_NATIVE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "CLUSTERREG_NATIVE requires a GNU or Clang C++ compiler")
endif()
if(CLUSTERREG_IPO)
  if(CLUSTERREG_CUDA)
    message(FATAL_ERROR "CLUSTERREG_IPO is supported for CPU-only builds; disable IPO for CUDA")
  endif()
  include(CheckIPOSupported)
  if(POLICY CMP0138)
    cmake_policy(SET CMP0138 NEW)
  endif()
  check_ipo_supported(RESULT CLUSTERREG_IPO_SUPPORTED
                      OUTPUT CLUSTERREG_IPO_ERROR LANGUAGES CXX)
  if(NOT CLUSTERREG_IPO_SUPPORTED)
    message(FATAL_ERROR "CLUSTERREG_IPO was requested but is unsupported: ${CLUSTERREG_IPO_ERROR}")
  endif()
endif()

function(clusterreg_enable_ipo target_name)
  if(CLUSTERREG_IPO)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endfunction()

set(CLUSTERREG_SOURCES
  src/clusterreg.cpp
  src/sinkhorn.cpp
  src/io.cpp
  src/synthetic.cpp
)
if(CLUSTERREG_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CLUSTERREG_CUDA=ON but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND CLUSTERREG_SOURCES src/cuda_transport.cu)
else()
  list(APPEND CLUSTERREG_SOURCES src/cuda_stub.cpp)
endif()

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Install Eigen >= 3.3 or set EIGEN3_INCLUDE_DIR")
  endif()
  add_library(eigen_local INTERFACE)
  target_include_directories(eigen_local INTERFACE "${EIGEN3_INCLUDE_DIR}")
  add_library(Eigen3::Eigen ALIAS eigen_local)
endif()

add_library(clusterreg STATIC ${CLUSTERREG_SOURCES})
add_library(clusterreg::clusterreg ALIAS clusterreg)
set_target_properties(clusterreg PROPERTIES POSITION_INDEPENDENT_CODE ON)
clusterreg_enable_ipo(clusterreg)
target_include_directories(clusterreg PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(clusterreg PUBLIC Eigen3::Eigen)
target_compile_definitions(clusterreg PUBLIC EIGEN_DONT_PARALLELIZE)
if(CLUSTERREG_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET clusterreg PROPERTY CUDA_ARCHITECTURES 75)
  endif()
  set_target_properties(clusterreg PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(clusterreg PUBLIC CLUSTERREG_HAS_CUDA=1)
  target_link_libraries(clusterreg PUBLIC CUDA::cudart)
  if(CLUSTERREG_CUDA_LINEINFO)
    target_compile_options(clusterreg PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    )
  endif()
endif()

if(CLUSTERREG_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(clusterreg PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()
if(CLUSTERREG_BLAS)
  find_package(BLAS REQUIRED)
  target_compile_definitions(clusterreg PUBLIC EIGEN_USE_BLAS)
  target_link_libraries(clusterreg PUBLIC ${BLAS_LIBRARIES})
endif()
if(MSVC)
  target_compile_options(clusterreg PRIVATE /W4)
else()
  target_compile_options(clusterreg PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  if(CLUSTERREG_NATIVE)
    target_compile_options(clusterreg PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
    )
  endif()
  if(CLUSTERREG_SANITIZE)
    target_compile_options(clusterreg PUBLIC
      $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>
    )
    target_link_options(clusterreg PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(CLUSTERREG_BUILD_CLI)
  add_executable(clusterreg_cli src/main.cpp)
  target_link_libraries(clusterreg_cli PRIVATE clusterreg::clusterreg)
  clusterreg_enable_ipo(clusterreg_cli)
endif()

if(CLUSTERREG_BUILD_TESTS)
  enable_testing()
  add_executable(clusterreg_tests tests/test_core.cpp)
  target_link_libraries(clusterreg_tests PRIVATE clusterreg::clusterreg)
  clusterreg_enable_ipo(clusterreg_tests)
  add_test(NAME mathematical_and_edge_cases COMMAND clusterreg_tests)
endif()

if(CLUSTERREG_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core python/bindings.cpp)
  target_link_libraries(_core PRIVATE clusterreg::clusterreg)
  target_compile_definitions(_core PRIVATE CLUSTERREG_VERSION="${PROJECT_VERSION}")
  clusterreg_enable_ipo(_core)
  install(TARGETS _core LIBRARY DESTINATION clusterreg)
endif()

if(NOT SKBUILD)
  include(GNUInstallDirs)
  install(TARGETS clusterreg
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  if(CLUSTERREG_BUILD_CLI)
    install(TARGETS clusterreg_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
  install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
