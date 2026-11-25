#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#define SP2_GEOMETRY_INLINE KOKKOS_INLINE_FUNCTION
#define SP2_GEOMETRY_MATH Kokkos
#include "../../src/spingnn2/geometry_kernels.hpp"
#undef SP2_GEOMETRY_MATH
#undef SP2_GEOMETRY_INLINE
