# Give both bundled fmt constexpr branches the GCC 10 C++20 capability.
# CMake 3.28 detects the GNU 10 host behind nvcc_wrapper and selects its
# historical c++2a spelling. CUDA 12 accepts the finalized C++20 name.
set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std=c++20")
set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-std=c++20")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  string(APPEND CMAKE_CXX_FLAGS_INIT " -DFMT_USE_CONSTEXPR=1")
endif()
