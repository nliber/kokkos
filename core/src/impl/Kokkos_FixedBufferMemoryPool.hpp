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

#ifndef KOKKOS_IMPL_KOKKOS_FIXEDBUFFERMEMORYPOOL_HPP
#define KOKKOS_IMPL_KOKKOS_FIXEDBUFFERMEMORYPOOL_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Atomic.hpp>

#include <Kokkos_PointerOwnership.hpp>
#include <impl/Kokkos_SimpleTaskScheduler.hpp>

namespace Kokkos {
namespace Impl {

template <class DeviceType, size_t Size, size_t Align=1>
class FixedBlockSizeMemoryPool
  : private MemorySpaceInstanceStorage<typename DeviceType::memory_space>
{
public:

  using memory_space = typename DeviceType::memory_space;
  using size_type = typename DeviceType::execution_space::size_type;

private:

  using memory_space_storage_base = MemorySpaceInstanceStorage<typename DeviceType::memory_space>;
  using tracker_type = Kokkos::Impl::SharedAllocationTracker;
  using record_type = Kokkos::Impl::SharedAllocationRecord<memory_space>;

  struct alignas(Align) Block { union { char ignore; char data[Size]; }; };

  static constexpr auto actual_size = sizeof(Block);

  // TODO shared allocation tracker

  tracker_type m_tracker;
  size_type m_num_blocks;
  size_type m_first_free_idx;
  size_type m_last_free_idx;
  Kokkos::OwningRawPtr<Block> m_first_block;
  Kokkos::OwningRawPtr<size_type> m_free_indices;

public:

  FixedBlockSizeMemoryPool(
    memory_space const& mem_space,
    size_type num_blocks
  ) : memory_space_storage_base(mem_space),
      m_tracker(),
      m_num_blocks(num_blocks),
      m_first_free_idx(0),
      m_last_free_idx(num_blocks)
  {
    // TODO alignment!!!
    auto block_record = record_type::allocate(
      mem_space, "FixedBlockSizeMemPool_blocks", num_blocks * sizeof(Block)
    );
    m_tracker.assign_allocated_record_to_uninitialized(block_record);
    m_first_block = (Block*)block_record->data();

    auto idx_record = record_type::allocate(
      mem_space, "FixedBlockSizeMemPool_blocks", num_blocks * sizeof(size_type)
    );
    m_tracker.assign_allocated_record_to_uninitialized(idx_record);
    m_free_indices = (size_type*)idx_record->data();

    for(size_type i = 0; i < num_blocks; ++i) {
      m_free_indices[i] = i;
    }
  }

  // For compatibility with MemoryPool<>
  FixedBlockSizeMemoryPool(
    memory_space const& mem_space,
    size_t mempool_capacity,
    unsigned, unsigned, unsigned
  ) : FixedBlockSizeMemoryPool(mem_space, mempool_capacity / actual_size)
  { /* forwarding ctor, must be empty */ }

  KOKKOS_INLINE_FUNCTION FixedBlockSizeMemoryPool() = default;
  KOKKOS_INLINE_FUNCTION FixedBlockSizeMemoryPool(FixedBlockSizeMemoryPool&&) = default;
  KOKKOS_INLINE_FUNCTION FixedBlockSizeMemoryPool(FixedBlockSizeMemoryPool const&) = default;
  KOKKOS_INLINE_FUNCTION FixedBlockSizeMemoryPool& operator=(FixedBlockSizeMemoryPool&&) = default;
  KOKKOS_INLINE_FUNCTION FixedBlockSizeMemoryPool& operator=(FixedBlockSizeMemoryPool const&) = default;

  KOKKOS_INLINE_FUNCTION
  void* allocate(size_type alloc_size) const noexcept
  {
    KOKKOS_EXPECTS(alloc_size <= Size);
    Kokkos::memory_fence();
    auto free_idx_idx = Kokkos::atomic_fetch_add((volatile size_type*)&m_first_free_idx, size_type(1));
    free_idx_idx %= m_num_blocks;
    // TODO check that it's not past the last free index!
    auto free_idx = m_free_indices[free_idx_idx];
    return (void*)&m_first_block[free_idx];
  }

  KOKKOS_INLINE_FUNCTION
  void deallocate(void* ptr, size_type alloc_size) const noexcept
  {
    // figure out which block we are
    auto offset = intptr_t(ptr) - intptr_t(m_first_block);

    KOKKOS_EXPECTS(offset % actual_size == 0 && offset/actual_size < m_num_blocks);

    auto last_idx_idx = Kokkos::atomic_fetch_add((volatile size_type*)&m_last_free_idx, size_type(1));
    last_idx_idx %= m_num_blocks;
    m_free_indices[last_idx_idx] = offset / actual_size;
    Kokkos::memory_fence();
  }

};

} // end namespace Impl
} // end namespace Kokkos

#endif //KOKKOS_IMPL_KOKKOS_FIXEDBUFFERMEMORYPOOL_HPP
