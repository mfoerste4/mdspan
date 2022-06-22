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

#pragma once

#include "macros.hpp"
#include "layout_stride.hpp"
#include "extents.hpp"

#include <algorithm>
#include <numeric>
#include <array>

namespace std {
namespace experimental {

// similar to layout_strided, but contiguous with padding in second smallest stride dimension
template <typename ElementType,
          bool RowMajorC,
          size_t ByteAlignment = 128,
          typename ::std::enable_if<(ByteAlignment % sizeof(ElementType) == 0 ||
                                     sizeof(ElementType) % ByteAlignment == 0),
                                    int>::type = 0>
struct layout_padded_general {
  template <class Extents>
  class mapping : public layout_stride::mapping<Extents>
  {
  public:
    // This could be a `requires`, but I think it's better and clearer as a `static_assert`.
    static_assert(detail::__is_extents_v<Extents>, "std::experimental::layout_padded_general::mapping must be instantiated with a specialization of std::experimental::extents.");
    // static_assert(Extents::rank() > 1, "std::experimental::layout_padded_general::mapping must be instantiated with a Extents having rank > 1.");

    using size_type = typename Extents::size_type;
    using extents_type = Extents;
    using layout_type = layout_padded_general;
    using layout_stride::mapping<Extents>::mapping;
  private:

    //----------------------------------------------------------------------------

    template <class>
    friend class mapping;

    //----------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION auto padded_row_major_strides(Extents const& __exts) -> std::array<size_t, Extents::rank()>
    {
      auto alignment              = std::max<size_t>(ByteAlignment / sizeof(ElementType), 1);
      std::array<size_t, Extents::rank()> strides;
      size_t stride               = 1;
      for (size_t r = Extents::rank() - 1; r > 0; r--) {
        strides[r] = stride;
        if (stride == 1) {
          stride *=
            std::max<size_t>(alignment, (__exts.extent(r) + alignment - 1) / alignment * alignment);
        } else {
          stride *= __exts.extent(r);
        }
      }
      strides[0] = stride;
      return strides;
    }

    MDSPAN_INLINE_FUNCTION auto  padded_col_major_strides(Extents const& __exts) -> std::array<size_t, Extents::rank()>
    {
      auto alignment              = std::max<size_t>(ByteAlignment / sizeof(ElementType), 1);
      std::array<size_t, Extents::rank()> strides = std::array<size_t, Extents::rank()>{};
      size_t stride               = 1;
      for (size_t r = 0; r + 1 < Extents::rank(); r++) {
      //for (size_t r2 = Extents::rank() - 1; r > 0; r--) {
        //size_t r = r2 + 1 - Extents::rank();
        strides[r] = stride;
        if (stride == 1) {
          stride *=
            std::max<size_t>(alignment, (__exts.extent(r) + alignment - 1) / alignment * alignment);
        } else {
          stride *= __exts.extent(r);
        }
      }
      strides[__exts.rank() - 1] = stride;
      return strides;
    }

  public:

    //--------------------------------------------------------------------------------

    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping() noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping const&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr mapping(mapping&&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED
    mapping& operator=(mapping const&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED
    mapping& operator=(mapping&&) noexcept = default;
    MDSPAN_INLINE_FUNCTION_DEFAULTED ~mapping() noexcept = default;

    MDSPAN_INLINE_FUNCTION
    constexpr
    mapping(Extents const& e) noexcept
      : layout_stride::mapping<Extents>{e, RowMajorC ? padded_row_major_strides(e) : padded_col_major_strides(e) }
    { }

    //--------------------------------------------------------------------------------
  };
};

} // end namespace experimental
} // end namespace std
