use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::context::CudaContext;
use crate::ffi::abi::preprocess as preprocess_abi;
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::ops::validation::{
    bf16_output, check_cuda, expect_bf16, gpu_ptr, make_gpu_tensor, matrix_shape,
};

fn storage_ranges_overlap(first: &crate::CudaBuffer, second: &crate::CudaBuffer) -> bool {
    let first_start = first.ptr() as usize;
    let second_start = second.ptr() as usize;
    let Some(first_end) = first_start.checked_add(first.len()) else {
        return true;
    };
    let Some(second_end) = second_start.checked_add(second.len()) else {
        return true;
    };
    first_start < second_end && second_start < first_end
}

/// Fused learned-position preparation for a spatial-merge vision tower.
///
/// The kernel bilinearly interpolates a square BF16 position table to
/// `height x width`, writes the embeddings in spatial-merge order, and emits
/// the matching `[row, column]` axial position IDs in the same launch.
pub struct VisionPositionArgs<'a> {
    pub table: &'a Tensor,
    pub embeddings: &'a mut Tensor,
    pub position_ids: &'a mut Tensor,
    pub height: usize,
    pub width: usize,
    pub merge_size: usize,
}

pub fn vision_position(ctx: &CudaContext, args: VisionPositionArgs<'_>) -> Result<()> {
    vision_position_impl(ctx, args, false)
}

/// Bilinear position interpolation with separately rounded F32 tap weights,
/// products, and left-to-right four-tap summation before BF16 output.
pub fn vision_position_staged(ctx: &CudaContext, args: VisionPositionArgs<'_>) -> Result<()> {
    vision_position_impl(ctx, args, true)
}

fn vision_position_impl(
    ctx: &CudaContext,
    args: VisionPositionArgs<'_>,
    staged: bool,
) -> Result<()> {
    let table_dims = args.table.shape().dims();
    if table_dims.len() != 2 {
        return Err(invalid("vision position table must be rank 2"));
    }
    let (source_positions, hidden) = (table_dims[0], table_dims[1]);
    let source_side = (source_positions as f64).sqrt() as usize;
    let tokens = args
        .height
        .checked_mul(args.width)
        .ok_or_else(|| invalid("vision position target size overflow"))?;
    if source_side == 0
        || source_side.checked_mul(source_side) != Some(source_positions)
        || hidden == 0
        || args.height == 0
        || args.width == 0
        || args.merge_size == 0
        || args.height % args.merge_size != 0
        || args.width % args.merge_size != 0
    {
        return Err(invalid(
            "vision position requires a square table and merge-aligned target grid",
        ));
    }
    for extent in [
        source_side,
        args.height,
        args.width,
        hidden,
        args.merge_size,
        tokens,
    ] {
        i32::try_from(extent)
            .map_err(|_| invalid("vision position dimension exceeds native limits"))?;
    }

    let table = tensor_storage(ctx, args.table, DType::BF16, table_dims)?;
    let embeddings = tensor_storage(ctx, args.embeddings, DType::BF16, &[tokens, hidden])?;
    let position_ids = tensor_storage(ctx, args.position_ids, DType::I32, &[tokens, 2])?;
    if storage_ranges_overlap(&table, &embeddings)
        || storage_ranges_overlap(&table, &position_ids)
        || storage_ranges_overlap(&embeddings, &position_ids)
    {
        return Err(invalid("vision position tensor storage must not overlap"));
    }
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    let launch = if staged {
        preprocess_abi::apxinf_preprocess_vision_position_staged_bf16
    } else {
        preprocess_abi::apxinf_preprocess_vision_position_bf16
    };
    unsafe {
        check_cuda(launch(
            table.ptr(),
            embeddings.ptr(),
            position_ids.ptr(),
            source_side as i32,
            args.height as i32,
            args.width as i32,
            hidden as i32,
            args.merge_size as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Generate Fourier features with per-channel sin/cos layout.
pub fn fourier_features(
    ctx: &CudaContext,
    waypoints: &Tensor,
    freqs: &Tensor,
    point_dim: usize,
    num_features: usize,
) -> Result<Tensor> {
    let (rows, width) = matrix_shape(waypoints, "fourier features")?;
    if rows == 0
        || width != point_dim
        || point_dim == 0
        || num_features == 0
        || freqs.shape().dims() != [num_features]
    {
        return Err(Error::Other("fourier features shape mismatch".into()));
    }
    expect_bf16(waypoints, "fourier features")?;
    expect_bf16(freqs, "fourier features")?;
    let out_width = point_dim * num_features * 2;
    let output = bf16_output(ctx, rows, out_width)?;
    unsafe {
        check_cuda(preprocess_abi::apxinf_preprocess_fourier_features_bf16(
            gpu_ptr(waypoints)?,
            gpu_ptr(freqs)?,
            output.ptr(),
            rows as i32,
            point_dim as i32,
            num_features as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, out_width]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Column-concatenate seven `[rows|1, cols]` sources into `[rows, 7 * cols]`;
/// `broadcast_mask` bit `i` broadcasts source `i` row 0 across rows.
pub fn concat7_cols(ctx: &CudaContext, srcs: [&Tensor; 7], broadcast_mask: u32) -> Result<Tensor> {
    let (first_rows, cols) = matrix_shape(srcs[0], "concat7")?;
    if first_rows == 0 || cols == 0 {
        return Err(Error::Other("concat7 shape mismatch".into()));
    }
    for (index, src) in srcs.iter().enumerate() {
        let (rows, width) = matrix_shape(src, "concat7")?;
        let expect_rows = if (broadcast_mask >> index) & 1 == 1 {
            1
        } else {
            first_rows
        };
        if rows != expect_rows || width != cols {
            return Err(Error::Other("concat7 shape mismatch".into()));
        }
        expect_bf16(src, "concat7")?;
    }
    let output = bf16_output(ctx, first_rows, 7 * cols)?;
    unsafe {
        check_cuda(preprocess_abi::apxinf_preprocess_concat7_cols_bf16(
            gpu_ptr(srcs[0])?,
            gpu_ptr(srcs[1])?,
            gpu_ptr(srcs[2])?,
            gpu_ptr(srcs[3])?,
            gpu_ptr(srcs[4])?,
            gpu_ptr(srcs[5])?,
            gpu_ptr(srcs[6])?,
            output.ptr(),
            first_rows as i32,
            cols as i32,
            broadcast_mask as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![first_rows, 7 * cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// In-place flow-matching Euler update on an F32 state:
/// `w += (endpoint - w) / remaining * step`.
pub fn flow_update(
    ctx: &CudaContext,
    w: &Tensor,
    endpoint: &Tensor,
    remaining: f32,
    step: f32,
) -> Result<()> {
    if w.dtype() != DType::F32
        || endpoint.dtype() != DType::F32
        || w.shape().dims() != endpoint.shape().dims()
        || w.numel() == 0
        || !remaining.is_finite()
        || remaining <= 0.0
        || !step.is_finite()
    {
        return Err(Error::Other("flow update shape mismatch".into()));
    }
    unsafe {
        check_cuda(preprocess_abi::apxinf_preprocess_flow_update_f32(
            gpu_ptr(w)?,
            gpu_ptr(endpoint)?,
            remaining,
            step,
            w.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Transpose contiguous BF16 `[batch, channels, spatial]` into
/// `[batch, spatial, padded_channels]`, filling extra channels with zero.
/// Caller retains output storage; input and output must not overlap.
pub fn transpose_pad_channels(
    ctx: &CudaContext,
    input: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let x = input.shape().dims();
    let y = output.shape().dims();
    if x.len() != 3
        || y.len() != 3
        || x.contains(&0)
        || y.contains(&0)
        || x[0] != y[0]
        || x[2] != y[1]
        || x[1] > y[2]
    {
        return Err(invalid(
            "transpose/pad requires [B,C,S] -> [B,S,Cp], Cp >= C",
        ));
    }
    for &extent in x.iter().chain(y.iter()) {
        i32::try_from(extent)
            .map_err(|_| invalid("transpose/pad dimension exceeds native range"))?;
    }
    for dims in [x, y] {
        dims.iter()
            .try_fold(2usize, |size, &v| size.checked_mul(v))
            .filter(|&size| size <= i64::MAX as usize)
            .ok_or_else(|| invalid("transpose/pad size overflow"))?;
    }
    let a = tensor_storage(ctx, input, DType::BF16, x)?;
    let b = tensor_storage(ctx, output, DType::BF16, y)?;
    if storage_ranges_overlap(&a, &b) {
        return Err(invalid("transpose/pad storage must not overlap"));
    }
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    unsafe {
        check_cuda(
            preprocess_abi::apxinf_preprocess_transpose_pad_channels_bf16(
                a.ptr(),
                b.ptr(),
                x[0] as i32,
                x[1] as i32,
                x[2] as i32,
                y[2] as i32,
                ctx.stream().handle(),
            ),
        )
    }
}
