//! BF16 gate/up projection with SwiGLU fused into the GEMM epilogue.
//!
//! Two fused implementations exist for the one exact Qwen-Drive shape
//! `[3387,2560] x [18432,2560]`: the CuTe-DSL AOT object
//! (`apxinf_quack_swiglu_bf16_m256n256_sm110`) and the in-tree CUTLASS
//! specialization in `kernels/cutlass/bf16_dual_geglu_sm100.cu`. They are
//! required to be bit-identical; which one a build links is decided at build
//! time by the `aot-gemm-swiglu-bf16` Cargo feature, never at runtime.
//!
//! **They do not consume the same physical weight layout, and the two layouts
//! have the same byte count.** See [`Bf16SwigluWeightLayout`].
//!
//! Only **one** of those layouts is ever resident: the generic cuBLAS route,
//! which every `M` other than the pinned one takes (decode included), reads
//! either layout bit-identically, so the model materializes exactly what
//! [`bf16_swiglu_weight_layout`] names. Keeping both would cost +94.4 MB per
//! backbone layer, +3.02 GB at 32 layers; see
//! `devlocal/aot2cuda/reports/23-swiglu-weight-memory.md`.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::super::contracts::{checked_bytes, require_buffers};
use crate::{buffer::CudaBuffer, context::CudaContext, cublas::CublasTranspose};

/// The one exact shape both fused implementations accept.
const M: usize = 3387;
const K: usize = 2560;
const INNER: usize = 9216;
const FULL_N: usize = 2 * INNER;
/// Physical output rows the fused route returns; the tail is zeroed.
const PADDED_M: usize = 3584;
/// Column tile of the `[gate256, up256]` interleave, from the kernel's
/// `kPairColumns`.
const PAIR_COLUMNS: usize = 256;

/// Fused BF16 SwiGLU implementation linked by this build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bf16SwigluImpl {
    /// Pinned CuTe-DSL AOT object from `APXINF_CUDA_AOT_MANIFEST`.
    Aot,
    /// In-tree CUTLASS `SwigluContract` specialization.
    Cuda,
}

/// Physical layout of the gate/up weight an implementation consumes.
///
/// Both variants hold `2 * INNER * K` BF16 elements, so **a mix-up is not a
/// size error**; what makes it safe here is that the two also differ in
/// *shape* (`[2I, K]` vs `[K, 2I]`, and `K != FULL_N`), which the dispatch
/// checks. Always derive the layout to materialize from
/// [`bf16_swiglu_weight_layout`] (equivalently
/// [`Bf16SwigluImpl::weight_layout`]) rather than from a shape test or a
/// `cfg!`: the shape check is a net, not the contract.
///
/// Exactly one of these is resident per build and device. The generic
/// (non-fused) route reads *either* one correctly -- see
/// [`bf16_swiglu_weight_layout`] -- which is what lets the fused route's
/// packed layout be the only copy instead of a second one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bf16SwigluWeightLayout {
    /// `[2*INNER, K]`, K-contiguous: all gate rows, then all up rows. This is
    /// the checkpoint's own order and what the AOT kernel takes. The generic
    /// route consumes it with `transb = T, ldb = K` and the plain
    /// `[gate_all, up_all]` activation.
    CheckpointRowMajor,
    /// `[K, 2*INNER]`, `2*INNER`-contiguous, columns grouped into
    /// `[gate256, up256]` tiles.
    ///
    /// Measured, not inferred: `cute::print` on the CUTLASS collective reports
    /// `StrideB = (_1, 18432, 0)` over logical `(N, K, L)`, i.e. B is
    /// N-contiguous. An earlier round assumed K-contiguous from
    /// `LayoutB = RowMajor` and spent a whole cycle on a correct kernel fed
    /// transposed data.
    ///
    /// The generic route consumes it with `transb = N, ldb = 2*INNER` and the
    /// `[gate pair, up pair]` activation, bit-identically to the checkpoint
    /// layout (`tests/qwen_drive_bf16_swiglu_single_weight.rs`).
    PackedPairInterleaved,
}

impl Bf16SwigluImpl {
    /// The physical weight layout this implementation consumes.
    pub const fn weight_layout(self) -> Bf16SwigluWeightLayout {
        match self {
            Self::Aot => Bf16SwigluWeightLayout::CheckpointRowMajor,
            Self::Cuda => Bf16SwigluWeightLayout::PackedPairInterleaved,
        }
    }

    /// Whether this build links this implementation at all.
    pub const fn linked(self) -> bool {
        match self {
            Self::Aot => cfg!(apxinf_aot_gemm_swiglu_bf16),
            Self::Cuda => cfg!(apxinf_cutlass_gemm),
        }
    }
}

/// Which fused implementation this build will use for the given operands, if
/// any. `None` means the generic cuBLAS + activation route.
pub fn bf16_swiglu_impl(
    ctx: &CudaContext,
    input_dims: &[usize],
    weight_elements: usize,
) -> Option<Bf16SwigluImpl> {
    if input_dims != [M, K] {
        return None;
    }
    device_impl(ctx, weight_elements)
}

/// The fused implementation this build and device would use for a weight of
/// this element count, ignoring `M`.
///
/// `M` decides whether a *call* takes the fused route; it must not decide
/// which weight layout gets *materialized*, because one buffer has to serve
/// every `M` the model runs. See [`bf16_swiglu_weight_layout`].
fn device_impl(ctx: &CudaContext, weight_elements: usize) -> Option<Bf16SwigluImpl> {
    if ctx.caps().sm != 110
        || ctx.caps().multiprocessor_count != 20
        || weight_elements != FULL_N * K
    {
        return None;
    }
    // AOT wins when linked: it is the previously measured baseline, and the
    // arm exists so an end-to-end regression can be attributed to this one
    // operator. The default build links only the CUDA implementation.
    for candidate in [Bf16SwigluImpl::Aot, Bf16SwigluImpl::Cuda] {
        if candidate.linked() {
            return Some(candidate);
        }
    }
    None
}

/// The one physical weight layout the caller must materialize for
/// [`bf16_swiglu_checkpoint`] on this build and device.
///
/// Deliberately independent of `M`: the model holds exactly one gate/up
/// buffer and runs it at every sequence length, so a layout that depended on
/// `M` would force a second resident copy (`+94.4 MB` per backbone layer,
/// `+3.02 GB` at 32 layers — see
/// `devlocal/aot2cuda/reports/23-swiglu-weight-memory.md`).
/// `PackedPairInterleaved` is served by the fused kernel at the pinned `M`
/// and by the generic `gemm + swiglu_bf16_rounded_pair_interleaved` route at
/// every other `M`.
pub fn bf16_swiglu_weight_layout(
    ctx: &CudaContext,
    weight_elements: usize,
) -> Bf16SwigluWeightLayout {
    device_impl(ctx, weight_elements)
        .map(Bf16SwigluImpl::weight_layout)
        .unwrap_or(Bf16SwigluWeightLayout::CheckpointRowMajor)
}

/// Source index into a `CheckpointRowMajor` `[2I, K]` weight for destination
/// index `index` of the `PackedPairInterleaved` `[K, 2I]` weight.
///
/// This is the single definition of the packed layout on the host side; it
/// mirrors `pack_weight_from_aot` in the kernel's own documentation. Keeping
/// it as an index map rather than a buffer transform lets the caller gather in
/// whatever element type it already holds, without a second copy.
pub fn bf16_swiglu_packed_source_index(index: usize, full_n: usize, k: usize) -> usize {
    debug_assert!(full_n % (2 * PAIR_COLUMNS) == 0);
    let half_n = full_n / 2;
    // Destination walks [K, full_n] with full_n contiguous.
    let row_k = index / full_n;
    let dst_col = index - row_k * full_n;
    let tile = dst_col / (2 * PAIR_COLUMNS);
    let within = dst_col - tile * (2 * PAIR_COLUMNS);
    let src_n = if within < PAIR_COLUMNS {
        tile * PAIR_COLUMNS + within
    } else {
        half_n + tile * PAIR_COLUMNS + (within - PAIR_COLUMNS)
    };
    // Source is [2I, K] with K contiguous.
    src_n * k + row_k
}

/// Gather a `CheckpointRowMajor` `[2I, K]` weight into the
/// `PackedPairInterleaved` `[K, 2I]` layout.
///
/// Takes the exactly-widened `f32` view a host `Tensor` hands out and returns
/// BF16, because `bf16 -> f32 -> bf16` is the identity and the alternative is
/// another full-size copy. This is the one gather the model load path performs,
/// and it is the same one
/// `tests/qwen_drive_bf16_swiglu_equivalence.rs` validates against the AOT
/// kernel, so the production permutation cannot drift from the verified one.
pub fn pack_bf16_swiglu_weight_host(
    source: &[f32],
    full_n: usize,
    k: usize,
) -> Result<Vec<half::bf16>> {
    if full_n == 0
        || k == 0
        || full_n % (2 * PAIR_COLUMNS) != 0
        || source.len() != full_n * k
    {
        return Err(Error::Other(format!(
            "BF16 SwiGLU weight pack expects {full_n} x {k} values with {full_n} \
             divisible by {}, got {}",
            2 * PAIR_COLUMNS,
            source.len()
        )));
    }
    Ok((0..full_n * k)
        .map(|index| {
            half::bf16::from_f32(source[bf16_swiglu_packed_source_index(index, full_n, k)])
        })
        .collect())
}

/// Compute SwiGLU from input `[M,K]` and a gate-then-up weight.
///
/// The weight must be in the layout [`bf16_swiglu_weight_layout`] reports for
/// these operands: `[2N,K]` for `CheckpointRowMajor`, `[K,2N]` for
/// `PackedPairInterleaved`.
///
/// The first M rows of the result are the logical output. A fused
/// implementation may return additional zero rows so a following projection
/// can run at a tile-aligned M; callers must slice back to M before the
/// residual add. The generic route materializes a BF16 projection first, so
/// its rounding boundaries differ from the fused routes' and cross-route
/// outputs are not promised bit-identical.
pub fn bf16_swiglu_checkpoint(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let selected = bf16_swiglu_impl(ctx, input.shape().dims(), weight.numel());
    dispatch(ctx, input, weight, selected)
}

/// [`bf16_swiglu_checkpoint`] with the implementation named explicitly.
///
/// Only the equivalence test uses this: it is the one caller that wants both
/// arms in one process. Production goes through [`bf16_swiglu_checkpoint`] so
/// a build can never run two different implementations in one pass.
pub fn bf16_swiglu_checkpoint_with(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    implementation: Bf16SwigluImpl,
) -> Result<Tensor> {
    if !implementation.linked() {
        return Err(Error::Other(format!(
            "BF16 SwiGLU implementation {implementation:?} is not linked by this build"
        )));
    }
    if bf16_swiglu_impl(ctx, input.shape().dims(), weight.numel()).is_none() {
        return Err(Error::Other(
            "BF16 SwiGLU fused route rejects this device or shape".into(),
        ));
    }
    dispatch(ctx, input, weight, Some(implementation))
}

fn dispatch(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    selected: Option<Bf16SwigluImpl>,
) -> Result<Tensor> {
    let x = input.shape().dims();
    let w = weight.shape().dims();
    if x.len() != 2 || w.len() != 2 || x.contains(&0) || w.contains(&0) {
        return Err(Error::Other(
            "BF16 SwiGLU expects input[M,K] and a 2-D weight".into(),
        ));
    }
    // The packed layout is recognized by its exact pinned shape. `K != FULL_N`
    // (asserted in the tests), so this can never be confused with a
    // `CheckpointRowMajor` `[2N, K]` weight, and the two layouts' equal
    // element count cannot produce a silently-wrong result: the generic route
    // below is correct for *either* layout, and which one it got is decided by
    // the shape rather than by a cfg or a capability test.
    let packed = w == [K, FULL_N];
    let expected_weight: &[usize] = match selected.map(Bf16SwigluImpl::weight_layout) {
        Some(Bf16SwigluWeightLayout::PackedPairInterleaved) => &[K, FULL_N],
        Some(Bf16SwigluWeightLayout::CheckpointRowMajor) => &[FULL_N, K],
        None => &[],
    };
    if expected_weight.is_empty() {
        // CheckpointRowMajor: [2N, K], K shared with the activation.
        if !packed && (x[1] != w[1] || w[0] % 2 != 0) {
            return Err(Error::Other(
                "BF16 SwiGLU expects input[M,K], weight[2N,K] or the packed \
                 [K,2N]"
                    .into(),
            ));
        }
    } else if w != expected_weight {
        // The layouts have equal element counts, so this is the only cheap
        // guard against being handed the other one.
        return Err(Error::Other(format!(
            "BF16 SwiGLU {selected:?} expects a weight {expected_weight:?}, got {w:?}"
        )));
    }
    for tensor in [input, weight] {
        if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "BF16 SwiGLU requires BF16 inputs on the context device".into(),
            ));
        }
    }
    let k = i32::try_from(x[1]).map_err(|_| Error::Other("SwiGLU K exceeds i32".into()))?;
    i32::try_from(x[0]).map_err(|_| Error::Other("SwiGLU rows exceed i32".into()))?;
    let xp = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let wp = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "BF16 SwiGLU",
        &[
            ("input", &xp, checked_bytes(DType::BF16, x, "SwiGLU input")?),
            (
                "weight",
                &wp,
                checked_bytes(DType::BF16, w, "SwiGLU weight")?,
            ),
        ],
    )?;

    if let Some(implementation) = selected {
        #[cfg(apxinf_aot_gemm_swiglu_bf16)]
        if implementation == Bf16SwigluImpl::Aot {
            return fused_aot(ctx, &xp, &wp);
        }
        #[cfg(apxinf_cutlass_gemm)]
        if implementation == Bf16SwigluImpl::Cuda {
            return fused_cuda(ctx, &xp, &wp);
        }
        return Err(Error::Other(format!(
            "BF16 SwiGLU implementation {implementation:?} is not linked by this build"
        )));
    }

    // Generic route. Both weight layouts describe the same logical
    // `[2N, K]` matrix, so the GEMM is the same product either way; only the
    // operand's physical orientation and the projection's column order
    // differ, and the column order is undone by the activation's pairing.
    let full_n = if packed { w[1] } else { w[0] };
    let n2 = i32::try_from(full_n).map_err(|_| Error::Other("SwiGLU width exceeds i32".into()))?;
    let projection = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::BF16, &[x[0], full_n], "SwiGLU projection")?,
    )?;
    ctx.cublas()
        .gemm_ex(
            DType::BF16,
            CublasTranspose::None,
            if packed {
                CublasTranspose::None
            } else {
                CublasTranspose::Transpose
            },
            x[0],
            full_n,
            x[1],
            1.0,
            &xp,
            k,
            &wp,
            if packed { n2 } else { k },
            0.0,
            &projection,
            n2,
        )
        .map_err(Error::Cuda)?;
    let projection = projection.into_tensor(Shape::new(vec![x[0], full_n]), DType::BF16);
    if packed {
        super::super::activation::swiglu_bf16_rounded_pair_interleaved(
            ctx,
            &projection,
            PAIR_COLUMNS,
        )
    } else {
        super::super::activation::swiglu_bf16_rounded(ctx, &projection)
    }
}

/// Padded output buffer plus the zeroed tail both fused routes promise.
fn fused_output(ctx: &CudaContext) -> Result<CudaBuffer> {
    let output = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::BF16, &[PADDED_M, INNER], "SwiGLU padded output")?,
    )?;
    let prefix = M * INNER * DType::BF16.size_in_bytes();
    let tail = (PADDED_M - M) * INNER * DType::BF16.size_in_bytes();
    output
        .view(prefix, tail)
        .map_err(Error::Cuda)?
        .memset_async(0, tail, ctx.stream())
        .map_err(Error::Cuda)?;
    Ok(output)
}

fn fused_tensor(output: CudaBuffer) -> Tensor {
    output.into_tensor(Shape::new(vec![PADDED_M, INNER]), DType::BF16)
}

#[cfg(apxinf_aot_gemm_swiglu_bf16)]
fn fused_aot(ctx: &CudaContext, xp: &CudaBuffer, wp: &CudaBuffer) -> Result<Tensor> {
    if crate::workspace::may_prepare_native_resources() {
        let status = unsafe { crate::ffi::apxinf_quack_m256n256_init() };
        if status != 0 {
            return Err(Error::Other(format!(
                "BF16 SwiGLU AOT prepare failed: {status}"
            )));
        }
    }
    let output = fused_output(ctx)?;
    // `max_active_clusters` is part of the pinned contract, not a tunable.
    let status = unsafe {
        crate::ffi::apxinf_quack_m256n256_forward(
            xp.ptr(),
            wp.ptr(),
            output.ptr(),
            M as i32,
            20,
            ctx.stream().handle(),
        )
    };
    if status != 0 {
        return Err(Error::Other(format!(
            "BF16 SwiGLU AOT enqueue failed: {status}"
        )));
    }
    Ok(fused_tensor(output))
}

#[cfg(apxinf_cutlass_gemm)]
fn fused_cuda(ctx: &CudaContext, xp: &CudaBuffer, wp: &CudaBuffer) -> Result<Tensor> {
    let output = fused_output(ctx)?;
    let status = unsafe {
        crate::ffi::apxinf_static_cutlass_bf16_dual_swiglu(
            xp.ptr(),
            wp.ptr(),
            output.ptr(),
            M as i32,
            INNER as i32,
            K as i32,
            FULL_N as i32,
            ctx.stream().handle(),
        )
    };
    if status != 0 {
        return Err(Error::Other(format!(
            "BF16 SwiGLU CUDA enqueue failed: {status}"
        )));
    }
    Ok(fused_tensor(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_weight_layouts_hold_the_same_number_of_elements() {
        // The property that makes a mismatch silent, asserted so nobody
        // "optimizes" the layout guard away on the grounds that a size check
        // would catch it.
        assert_eq!(FULL_N * K, K * FULL_N);
    }

    #[test]
    fn the_two_weight_layouts_have_distinguishable_shapes() {
        // What the dispatch's shape-based layout recognition rests on. If a
        // future shape had K == FULL_N the recognition would be ambiguous and
        // the generic route could apply the wrong gate/up pairing, so this is
        // a load-bearing assertion, not a tautology.
        assert_ne!(K, FULL_N);
    }

    #[test]
    fn the_interleaved_pairing_inverts_the_packed_column_order() {
        // The generic route's activation reads, for output column `col`,
        // projection columns (gate_col, gate_col + PAIR_COLUMNS). Those must
        // be the projection columns the packed weight produces for logical
        // gate column `col` and up column `col`, i.e. destination columns
        // whose packed source index lands on source rows `col` and
        // `half_n + col`. Checked over every column of the shipping shape,
        // so the kernel's map and the packer's map cannot drift apart.
        let (full_n, k) = (FULL_N, K);
        let half_n = full_n / 2;
        for col in 0..INNER {
            let tile = col / PAIR_COLUMNS;
            let within = col - tile * PAIR_COLUMNS;
            let gate_col = tile * (2 * PAIR_COLUMNS) + within;
            let up_col = gate_col + PAIR_COLUMNS;
            // Row 0 of the packed weight: destination index == column index.
            assert_eq!(
                bf16_swiglu_packed_source_index(gate_col, full_n, k),
                col * k,
                "gate column {col}"
            );
            assert_eq!(
                bf16_swiglu_packed_source_index(up_col, full_n, k),
                (half_n + col) * k,
                "up column {col}"
            );
        }
    }

    #[test]
    fn packed_source_index_is_a_permutation() {
        let (full_n, k) = (FULL_N, K);
        let total = full_n * k;
        let mut seen = vec![false; total];
        for index in 0..total {
            let source = bf16_swiglu_packed_source_index(index, full_n, k);
            assert!(source < total, "source {source} out of range at {index}");
            assert!(!seen[source], "source {source} visited twice");
            seen[source] = true;
        }
    }

    #[test]
    fn packed_source_index_transposes_and_interleaves() {
        let (full_n, k) = (FULL_N, K);
        let half_n = full_n / 2;
        // Destination row 0, column 0 is gate column 0 at depth 0.
        assert_eq!(bf16_swiglu_packed_source_index(0, full_n, k), 0);
        // Column 256 of the first tile is the first up column.
        assert_eq!(
            bf16_swiglu_packed_source_index(PAIR_COLUMNS, full_n, k),
            half_n * k
        );
        // Column 512 opens the second tile: gate column 256.
        assert_eq!(
            bf16_swiglu_packed_source_index(2 * PAIR_COLUMNS, full_n, k),
            PAIR_COLUMNS * k
        );
        // Destination row 1 is depth 1 of the same gate column.
        assert_eq!(bf16_swiglu_packed_source_index(full_n, full_n, k), 1);
    }

    #[test]
    fn implementations_declare_distinct_weight_layouts() {
        assert_eq!(
            Bf16SwigluImpl::Aot.weight_layout(),
            Bf16SwigluWeightLayout::CheckpointRowMajor
        );
        assert_eq!(
            Bf16SwigluImpl::Cuda.weight_layout(),
            Bf16SwigluWeightLayout::PackedPairInterleaved
        );
    }
}
