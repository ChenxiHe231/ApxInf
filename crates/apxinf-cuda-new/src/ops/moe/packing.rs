use apxinf_core::{Error, Result};

use crate::ops::qwen35_attention::{decode_e4m3, encode_e4m3};

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

/// Interleave checkpoint FC1 bytes in the TensorRT-Edge-LLM 64-row layout.
///
/// The returned order is `up[0:64], gate[0:64], up[64:128], ...`.
pub fn edge_nvfp4_interleave_fc1(
    gate: &[u8],
    up: &[u8],
    intermediate: usize,
    row_bytes: usize,
) -> Result<Vec<u8>> {
    if intermediate == 0 || intermediate % 64 != 0 || row_bytes == 0 {
        return Err(invalid(
            "Edge NVFP4 FC1 requires non-zero row bytes and intermediate multiple of 64",
        ));
    }
    let projection_bytes = intermediate
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Edge NVFP4 FC1 byte size overflow"))?;
    if gate.len() != projection_bytes || up.len() != projection_bytes {
        return Err(invalid(format!(
            "Edge NVFP4 FC1 projection byte size mismatch: expected {projection_bytes}, got gate={} up={}",
            gate.len(),
            up.len()
        )));
    }
    let chunk_bytes = 64 * row_bytes;
    let mut packed = Vec::with_capacity(2 * projection_bytes);
    for chunk in 0..intermediate / 64 {
        let begin = chunk * chunk_bytes;
        let end = begin + chunk_bytes;
        packed.extend_from_slice(&up[begin..end]);
        packed.extend_from_slice(&gate[begin..end]);
    }
    Ok(packed)
}

/// Swizzle linear `[rows, scale_columns]` E4M3 bytes to CuTe's 128x4 MMA
/// layout. The returned physical shape is
/// `[ceil(rows/128), ceil(scale_columns/4), 32, 4, 4]`.
pub fn edge_nvfp4_swizzle_scales(
    linear: &[u8],
    rows: usize,
    scale_columns: usize,
) -> Result<(Vec<u8>, [usize; 5])> {
    if rows == 0 || scale_columns == 0 {
        return Err(invalid("Edge NVFP4 scale matrix must be non-empty"));
    }
    let linear_bytes = rows
        .checked_mul(scale_columns)
        .ok_or_else(|| invalid("Edge NVFP4 scale byte size overflow"))?;
    if linear.len() != linear_bytes {
        return Err(invalid(format!(
            "Edge NVFP4 scale byte size mismatch: expected {linear_bytes}, got {}",
            linear.len()
        )));
    }
    let m_tiles = rows.div_ceil(128);
    let k_tiles = scale_columns.div_ceil(4);
    let shape = [m_tiles, k_tiles, 32, 4, 4];
    let mut swizzled = vec![0u8; m_tiles * k_tiles * 32 * 4 * 4];
    for row in 0..rows {
        let m_tile = row / 128;
        let row_in_tile = row % 128;
        let inner_m = row_in_tile / 32;
        let outer_m = row_in_tile % 32;
        for column in 0..scale_columns {
            let k_tile = column / 4;
            let inner_k = column % 4;
            let destination =
                ((((m_tile * k_tiles + k_tile) * 32 + outer_m) * 4 + inner_m) * 4) + inner_k;
            swizzled[destination] = linear[row * scale_columns + column];
        }
    }
    Ok((swizzled, shape))
}

/// Re-express E4M3 block scales under a larger shared FC1 alpha.
/// `factor` is `projection_alpha / shared_alpha` and must be in `(0, 1]`.
pub fn edge_nvfp4_rescale_e4m3(scales: &[u8], factor: f32) -> Result<Vec<u8>> {
    if !factor.is_finite() || factor <= 0.0 || factor > 1.0 {
        return Err(invalid(
            "Edge NVFP4 E4M3 rescale factor must be finite and in (0, 1]",
        ));
    }
    if factor == 1.0 {
        return Ok(scales.to_vec());
    }
    Ok(scales
        .iter()
        .map(|&scale| encode_e4m3(decode_e4m3(scale) * factor))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fc1_interleave_is_up_then_gate_in_64_row_chunks() {
        let gate = (0..128u8).collect::<Vec<_>>();
        let up = (128..=255u8).collect::<Vec<_>>();
        let packed = edge_nvfp4_interleave_fc1(&gate, &up, 128, 1).unwrap();
        assert_eq!(&packed[0..64], &up[0..64]);
        assert_eq!(&packed[64..128], &gate[0..64]);
        assert_eq!(&packed[128..192], &up[64..128]);
        assert_eq!(&packed[192..256], &gate[64..128]);
    }

    #[test]
    fn scale_swizzle_matches_cute_128x4_coordinates() {
        let linear = (0..128 * 4).map(|value| value as u8).collect::<Vec<_>>();
        let (swizzled, shape) = edge_nvfp4_swizzle_scales(&linear, 128, 4).unwrap();
        assert_eq!(shape, [1, 1, 32, 4, 4]);
        for row in 0..128 {
            let inner_m = row / 32;
            let outer_m = row % 32;
            for column in 0..4 {
                let offset = ((outer_m * 4 + inner_m) * 4) + column;
                assert_eq!(swizzled[offset], linear[row * 4 + column]);
            }
        }
    }

    #[test]
    fn shared_alpha_rescale_rounds_back_to_e4m3() {
        let one = encode_e4m3(1.0);
        let half = edge_nvfp4_rescale_e4m3(&[one], 0.5).unwrap();
        assert_eq!(decode_e4m3(half[0]), 0.5);
        assert_eq!(edge_nvfp4_rescale_e4m3(&[one], 1.0).unwrap(), [one]);
    }
}
