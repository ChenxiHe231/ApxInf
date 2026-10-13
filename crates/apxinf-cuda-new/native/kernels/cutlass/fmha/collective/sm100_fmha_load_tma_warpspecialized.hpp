/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cute/layout.hpp"

#include "collective/fmha_common.hpp"
#include "collective/fmha_fusion.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template<
  class Element,
  class StrideQ,
  class StrideK,
  class StrideV,
  class CollectiveMmaQK,
  class CollectiveMmaPV,
  class SmemLayoutQ,
  class SmemLayoutK,
  class SmemLayoutV,
  class TensorStorage,
  class PipelineQ,
  class PipelineKV,
  class Mask,
  class TileShape
>
struct Sm100FmhaLoadTmaWarpspecialized {

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  struct Arguments {
    const Element* ptr_Q;
    StrideQ dQ;
    const Element* ptr_K;
    StrideK dK;
    const Element* ptr_V;
    StrideV dV;
  };

  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  struct Params {
    TMA_Q tma_load_q;
    TMA_K tma_load_k;
    TMA_V tma_load_v;
  };

  template<class ProblemShape>
  static Params to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      void* workspace) {

    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    auto dQ = args.dQ;
    auto dK = args.dK;
    auto dV = args.dV;

    // APXINF_LOAD_SHAPE_PER_MODE
    //   0 (UPSTREAM) declare a hard-coded
    //       using IntProblemShape = tuple<int,int,int,tuple<tuple<int,int>,int>>;
    //     and assign the problem shape into it.  Two problems with that:
    //      1. it silently DEMOTES every static extent to a dynamic int, which
    //         breaks pack-GQA (the M mode is (Int<q_per_kv>, seq_q) and
    //         make_tma_copy needs the tile-M size to be a compile-time
    //         constant);
    //      2. it only resolves VariableLength when BOTH mode 0 and mode 1 are
    //         variable -- the action kernel needs a variable seq_k with a
    //         fixed seq_q so that both batches fit in ONE launch.
    //   1 resolve each mode on its own and otherwise pass the shape through.
    // For a fully dynamic `tuple<int,int,int,tuple<tuple<int,int>,int>>` the
    // two are the same type with the same values, which is what both in-tree
    // consumers (crates/apxinf-cuda{,-new}) instantiate.
#ifndef APXINF_LOAD_SHAPE_PER_MODE
#define APXINF_LOAD_SHAPE_PER_MODE 0
#endif
#if APXINF_LOAD_SHAPE_PER_MODE
    auto resolve_len = [](auto const& s_) {
      if constexpr (is_variable_length_v<decltype(s_)>) { return s_.total_length; }
      else { return s_; }
    };
    auto problem_shape_qk = cute::make_tuple(
        resolve_len(get<0>(problem_shape)), resolve_len(get<1>(problem_shape)),
        get<2>(problem_shape), get<3>(problem_shape));
#else
    using IntProblemShape = cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

    IntProblemShape problem_shape_qk;
    if constexpr (is_variable_length_v<tuple_element_t<0, ProblemShape>>) {
      auto cumulative_length_q = get<0>(problem_shape).cumulative_length;
      auto cumulative_length_k = get<1>(problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr && cumulative_length_k != nullptr ) {
          get<0>(problem_shape_qk) = get<0>(problem_shape).total_length;
          get<1>(problem_shape_qk) = get<1>(problem_shape).total_length;
          get<2>(problem_shape_qk) = get<2>(problem_shape);
          get<3>(problem_shape_qk) = get<3>(problem_shape);
      }
    } else {
      problem_shape_qk = problem_shape;
    }
#endif

    auto params_qk = CollectiveMmaQK::to_underlying_arguments(
        problem_shape_qk,
        typename CollectiveMmaQK::Arguments {
            ptr_Q, dQ,
            ptr_K, dK,
        }, /*workspace=*/ nullptr);

#if APXINF_LOAD_SHAPE_PER_MODE
    // The PV collective's A side is a dummy (P comes from tmem, never from
    // gmem) but its TMA_A is still TYPED from make_shape(M,K,L) x StrideK, so
    // M has to stay FLAT here even when the real Q/O M mode is the
    // hierarchical pack-GQA coordinate -- otherwise shape and stride have a
    // different number of leaves.
    auto problem_shape_pv = cute::make_tuple(cute::size<0>(problem_shape_qk),
                                             cute::get<2>(problem_shape_qk),
                                             cute::get<1>(problem_shape_qk),
                                             cute::get<3>(problem_shape_qk));
#else
    auto problem_shape_pv = select<0,2,1,3>(problem_shape_qk);
#endif
    auto params_pv = CollectiveMmaPV::to_underlying_arguments(
        problem_shape_pv,
        typename CollectiveMmaPV::Arguments {
            ptr_K, dK,  // never used, dummy
            ptr_V, select<1,0,2>(dV),
        }, /*workspace=*/ nullptr);

    return Params{
        params_qk.tma_load_a,
        params_qk.tma_load_b,
        params_pv.tma_load_b
    };
  }


  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_k.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_v.get_tma_descriptor());
  }

  template<class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void
  load(
      BlkCoord const& blk_coord_in, ProblemShape const& problem_shape,
      Params const& params, ParamsProblemShape const& params_problem_shape,
      TensorStorage& storage,
      PipelineQ& pipeline_q, typename PipelineQ::PipelineState& pipeline_q_producer_state,
      PipelineKV& pipeline_kv, typename PipelineKV::PipelineState& pipeline_kv_producer_state) {

    BlkCoord blk_coord_q = blk_coord_in;
    BlkCoord blk_coord_kv = blk_coord_in;

    int mask_tile_count = Mask{}.get_trip_count(blk_coord_in, TileShape{}, problem_shape);

    using X = Underscore;

    // this one is only executed by one thread, no need to elect_one

    // Q1, K1, Q2, V1, K2, V2, K3, V3, ...
    // two pipes: Q and KV
    // from Memory (prod) to TensorCore (cons)

    // compute gQ, sQ
    // we load 2*get<0>(blk_coord), and 2*get<0>(blk_coord) + 1
    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    Tensor mQ_qdl_p = params.tma_load_q.get_tma_tensor(select<0,2,3>(problem_shape));

    int q_offs_0 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<0, ParamsProblemShape>>) {
      auto cumulative_length_q = get<0>(params_problem_shape).cumulative_length;
      if (cumulative_length_q != nullptr) {
        q_offs_0 = cumulative_length_q[get<2,1>(blk_coord_q)];
        get<2,1>(blk_coord_q) = 0;
      }
    }

    Tensor mQ_qdl = domain_offset(make_coord(q_offs_0, _0{}, make_coord(_0{}, _0{})), mQ_qdl_p);

    Tensor gQ_qdl = local_tile(mQ_qdl, TileShapeQK{}, make_coord(_, _, _), Step<_1, X, _1>{});
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ_qdl);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto [tQgQ_qdl, tQsQ] = tma_partition(
      params.tma_load_q, _0{}, make_layout(_1{}), 
      group_modes<0,3>(sQ), group_modes<0,3>(tSgQ_qdl)
    );
    Tensor tQgQ = tQgQ_qdl(_, _, _0{}, get<2>(blk_coord_q));

    // compute gK, sK
    Tensor mK_kdl_p = params.tma_load_k.get_tma_tensor(select<1,2,3>(problem_shape));

    int kv_offs_0 = 0;

    if constexpr (is_variable_length_v<tuple_element_t<1, ParamsProblemShape>>) {
      auto cumulative_length = get<1>(params_problem_shape).cumulative_length;
      if (cumulative_length != nullptr) {
        kv_offs_0 = cumulative_length[get<2,1>(blk_coord_kv)];
        get<2,1>(blk_coord_kv) = 0;
      }
    }

    Tensor mK_kdl = domain_offset(make_coord(kv_offs_0, _0{}, make_coord(_0{}, _0{})), mK_kdl_p);

    Tensor gK_kdl = local_tile(mK_kdl, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tSgK_kdl = mma_qk.partition_B(gK_kdl);
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    auto [tKgK_kdl, tKsK] = tma_partition(
      params.tma_load_k, _0{}, make_layout(_1{}),
      group_modes<0,3>(sK), group_modes<0,3>(tSgK_kdl)
    );
    Tensor tKgK = tKgK_kdl(_, _, _0{}, get<2>(blk_coord_kv));

    // compute gV, sV
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor mV_dkl_p = params.tma_load_v.get_tma_tensor(select<2,1,3>(problem_shape));

    Tensor mV_dkl = domain_offset(make_coord(_0{}, kv_offs_0, make_coord(_0{}, _0{})), mV_dkl_p);

    Tensor gV_dkl = local_tile(mV_dkl, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{});
    Tensor tOgV_dkl = mma_pv.partition_B(gV_dkl);
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
    auto [tVgV_dkl, tVsV] = tma_partition(
      params.tma_load_v, _0{}, make_layout(_1{}),
      group_modes<0,3>(sV), group_modes<0,3>(tOgV_dkl)
    );
    auto tVgV = tVgV_dkl(_, _0{}, _, get<2>(blk_coord_kv));

    // blk_coord in decomposed in terms of TileShape, not TileShapeQK
    // As such, it needs to be transformed as
    // (a,b,c): a -> 2*a (Q0) 2*a+1 (Q1)
    //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)

    uint32_t lane_predicate = cute::elect_one_sync();

    // apxinf: how many Q sub-tiles this CTA owns. Two for the stock
    // ThreadShape<_2,_1,_1>; one for the single-Q-stage head_dim 256 path.
    constexpr int kNumQStages = size<0>(TileShape{}) / size<0>(TileShapeQK{});
    static_assert(kNumQStages == 1 || kNumQStages == 2, "only 1 or 2 Q stages");

    // Q1
    int q0_index = kNumQStages * get<0>(blk_coord_q);
    int q1_index = kNumQStages * get<0>(blk_coord_q) + 1;
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q0_index), tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // K1
    // FA4's two hd256 kernels fetch in opposite directions; both the frozen row
    // max and the O accumulation order depend on it, so the fetch order has to
    // match the softmax warp's loop (the mma and softmax warps just drain this
    // pipeline in order).
    //   DESCENDING: `flash_fwd_sm100.py:2355` (action, vision)
    //   ASCENDING : `sm100_hd256_2cta_fmha_forward.py:1449` (language)
// APXINF_KV_DESCENDING is defined (default 0 = UPSTREAM ascending) by
// collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp, which also
// documents it; this header only consumes it.  Both headers MUST see the same
// value or the fetch order and the softmax loop order disagree.
#ifndef APXINF_KV_DESCENDING
#define APXINF_KV_DESCENDING 0
#endif
#if APXINF_KV_DESCENDING
    int k_index = mask_tile_count - 1;
    constexpr int k_step = -1;
#else
    int k_index = 0;
    constexpr int k_step = 1;
#endif
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index()));
    }
    ++pipeline_kv_producer_state;

    // Q2
    if constexpr (kNumQStages == 2) {
      pipeline_q.producer_acquire(pipeline_q_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
        copy(params.tma_load_q.with(*tma_barrier, 0), tQgQ(_, q1_index), tQsQ(_, pipeline_q_producer_state.index()));
      }
      ++pipeline_q_producer_state;
    }

    // V1
    pipeline_kv.producer_acquire(pipeline_kv_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
      copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index()));
    }
    ++pipeline_kv_producer_state;
    k_index += k_step;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // Ki
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_k.with(*tma_barrier, 0), tKgK(_, k_index), tKsK(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;

      // Vi
      pipeline_kv.producer_acquire(pipeline_kv_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
        copy(params.tma_load_v.with(*tma_barrier, 0), tVgV(_, k_index), tVsV(_, pipeline_kv_producer_state.index()));
      }
      ++pipeline_kv_producer_state;
      k_index += k_step;
    }
  }
};

}  // namespace cutlass::fmha::collective
