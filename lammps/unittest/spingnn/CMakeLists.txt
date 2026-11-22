cmake_minimum_required(VERSION 3.18)
project(SpingnnSpinSyncContract LANGUAGES CXX)

option(SPINGNN_BUILD_SPIN_SYNC_PROBE "Build the real-LAMMPS CUDA spin synchronization regression" OFF)
set(SPINGNN_LAMMPS_SOURCE_DIR "" CACHE PATH "Matching frozen LAMMPS src directory")
set(SPINGNN_LAMMPS_BUILD_DIR "" CACHE PATH "Matching frozen serial Kokkos LAMMPS build directory")
set(SPINGNN_LAMMPS_LIBRARY "" CACHE FILEPATH "Matching frozen liblammps.a")

if(SPINGNN_BUILD_SPIN_SYNC_PROBE)
  # Discover the actual toolkit first: older Kokkos exports can otherwise
  # install absolute CUDA driver targets from the original build machine.
  find_package(CUDAToolkit REQUIRED)
  find_package(Kokkos REQUIRED)
  find_package(Torch REQUIRED)
  if(NOT CAFFE2_USE_CUDA)
    message(FATAL_ERROR "The spin synchronization regression requires CUDA-enabled Torch")
  endif()
  foreach(required IN ITEMS
      "${SPINGNN_LAMMPS_SOURCE_DIR}/atom_masks.h"
      "${SPINGNN_LAMMPS_BUILD_DIR}/CMakeCache.txt"
      "${SPINGNN_LAMMPS_BUILD_DIR}/CMakeFiles/lammps.dir/flags.make"
      "${SPINGNN_LAMMPS_BUILD_DIR}/liblammps_mpi_stubs.a"
      "${SPINGNN_LAMMPS_LIBRARY}")
    if(NOT EXISTS "${required}")
      message(FATAL_ERROR "The spin-sync probe requires explicit matching source/library/build paths: ${required}")
    endif()
  endforeach()
  get_filename_component(_source "${SPINGNN_LAMMPS_SOURCE_DIR}" REALPATH)
  get_filename_component(_library "${SPINGNN_LAMMPS_LIBRARY}" REALPATH)
  get_filename_component(_build_library "${SPINGNN_LAMMPS_BUILD_DIR}/liblammps.a" REALPATH)
  if(NOT _library STREQUAL _build_library)
    message(FATAL_ERROR "SPINGNN_LAMMPS_LIBRARY must belong to SPINGNN_LAMMPS_BUILD_DIR")
  endif()
  file(STRINGS "${SPINGNN_LAMMPS_BUILD_DIR}/CMakeCache.txt" _mpi REGEX "^BUILD_MPI:BOOL=")
  string(REGEX REPLACE "^BUILD_MPI:BOOL=" "" _mpi "${_mpi}")
  if(NOT _mpi MATCHES "^(no|NO|OFF|FALSE|0)$")
    message(FATAL_ERROR "spin-sync-probe currently requires a serial MPI-stub LAMMPS build")
  endif()
  file(STRINGS "${SPINGNN_LAMMPS_BUILD_DIR}/CMakeCache.txt" _kokkos REGEX "^Kokkos_DIR:PATH=")
  string(REGEX REPLACE "^Kokkos_DIR:PATH=" "" _kokkos "${_kokkos}")
  get_filename_component(_kokkos "${_kokkos}" REALPATH)
  get_filename_component(_selected_kokkos "${Kokkos_DIR}" REALPATH)
  if(NOT _kokkos STREQUAL _selected_kokkos)
    message(FATAL_ERROR "The spin-sync probe must use the same Kokkos installation as liblammps.a")
  endif()
  file(STRINGS "${SPINGNN_LAMMPS_BUILD_DIR}/CMakeFiles/lammps.dir/flags.make" _flag_lines
       REGEX "^CXX_(DEFINES|INCLUDES|FLAGS) = ")
  set(_compile_options)
  set(_source_bound FALSE)
  foreach(line IN LISTS _flag_lines)
    string(REGEX REPLACE "^CXX_[A-Z]+ = " "" line "${line}")
    separate_arguments(options UNIX_COMMAND "${line}")
    foreach(argument IN LISTS options)
      if(argument STREQUAL "-I${_source}")
        set(_source_bound TRUE)
      endif()
    endforeach()
    # Keep repeated -isystem/path pairs together when CMake de-duplicates options.
    list(APPEND _compile_options "SHELL:${line}")
  endforeach()
  if(NOT _source_bound)
    message(FATAL_ERROR "SPINGNN_LAMMPS_SOURCE_DIR does not match the library build's include paths")
  endif()

  # Use the library's exact definitions/includes so old-mask and fixed-mask
  # experiments cannot accidentally combine headers from different builds.
  # The static LAMMPS archive uses OpenMP even when Kokkos uses CUDA/Serial.
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  add_executable(spin-sync-probe spin_sync.cpp)
  target_compile_features(spin-sync-probe PRIVATE cxx_std_17)
  target_compile_options(spin-sync-probe PRIVATE ${_compile_options})
  target_link_libraries(spin-sync-probe PRIVATE "${_library}"
      "${SPINGNN_LAMMPS_BUILD_DIR}/liblammps_mpi_stubs.a"
      Kokkos::kokkos "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
endif()
