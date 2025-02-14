/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "ctest_common.hpp"

#include <experimental/mdspan>

#include <type_traits>

namespace stdex = std::experimental;

//==============================================================================
// <editor-fold des4c="Test allowed pointer + extents ctors"> {{{1

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        std::array<int,1>
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        std::array<int,2>
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        int
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        int, int64_t
    >::value
);

// TODO @proposal-bug: not sure we really intended this???
MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        std::array<float,2>
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        float, double
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<int, stdex::extents<>>,
        int*
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<int, stdex::extents<2>>,
        int*
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<int, stdex::extents<2, stdex::dynamic_extent>>,
        int*, int
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<double, stdex::extents<stdex::dynamic_extent, 2, stdex::dynamic_extent>>,
        double*, unsigned, int
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<int, stdex::extents<2, stdex::dynamic_extent>>,
        int*, int, int
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<int, stdex::extents<stdex::dynamic_extent, 2, stdex::dynamic_extent>>,
        int*, std::array<int,2>
    >::value
);

MDSPAN_STATIC_TEST(
    std::is_constructible<
        stdex::mdspan<int, stdex::extents<2, stdex::dynamic_extent>>,
        int*, std::array<int,2>
    >::value
);

// </editor-fold> end Test allowed pointer + extents ctors }}}1
//==============================================================================


//==============================================================================
// <editor-fold desc="Test forbidden pointer + extents ctors"> {{{1
MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        std::array<int, 4>
    >::value
);

MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::extents<2, stdex::dynamic_extent>,
        int, int, int
    >::value
);

MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::mdspan<int, stdex::extents<2, stdex::dynamic_extent>>,
        int*, std::array<int, 4>
    >::value
);

MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::mdspan<int, stdex::extents<2, stdex::dynamic_extent>>,
        double*, int
    >::value
);


MDSPAN_STATIC_TEST(
    !std::is_constructible<
        stdex::mdspan<int, stdex::extents<2, stdex::dynamic_extent>>,
        int*, int, int, int
    >::value
);

MDSPAN_STATIC_TEST(
   !std::is_constructible<
        stdex::mdspan<int, stdex::dextents<2>, stdex::layout_stride>,
        int*, int, int
   >::value
);

MDSPAN_STATIC_TEST(
   !std::is_constructible<
        stdex::mdspan<int, stdex::dextents<2>, stdex::layout_stride>,
        int*, std::array<int,2>
   >::value
);

MDSPAN_STATIC_TEST(
   !std::is_constructible<
        stdex::mdspan<int, stdex::dextents<2>, stdex::layout_stride>,
        int*, stdex::dextents<2>
   >::value
);

// </editor-fold> end Test forbidden pointer + extents ctors }}}1
//==============================================================================
