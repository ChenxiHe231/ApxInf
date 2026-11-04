//! Independent, allocation-bounded scalar oracle for prepared Q/K/V attention.
//! No CUDA, library attention implementation, RoPE or quantization is reused.
//! Inputs are f32 values decoded from the same BF16 bit patterns as the device.
//! The oracle accumulates in f64 to make reference reduction error negligible;
//! production FP32 accumulation and final BF16 storage are measured against it.

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub queries: usize,
    pub valid_keys: usize,
    pub capacity: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub query_start: usize,
}

fn size(parts: &[usize]) -> Result<usize, String> {
    parts.iter().try_fold(1usize, |value, part| {
        value.checked_mul(*part).ok_or_else(|| "shape overflow".into())
    })
}

pub fn causal_attention(
    g: Geometry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f32,
) -> Result<Vec<f32>, String> {
    if [g.queries, g.valid_keys, g.capacity, g.query_heads, g.kv_heads, g.head_dim]
        .contains(&0)
        || g.query_heads % g.kv_heads != 0
        || g.valid_keys > g.capacity
        || g.query_start.checked_add(g.queries).is_none_or(|end| end > g.valid_keys)
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err("invalid attention geometry/scale".into());
    }
    let query_elements = size(&[g.queries, g.query_heads, g.head_dim])?;
    let cache_elements = size(&[g.capacity, g.kv_heads, g.head_dim])?;
    if q.len() != query_elements || k.len() != cache_elements || v.len() != cache_elements {
        return Err("attention input shape mismatch".into());
    }
    if q.iter().chain(k).chain(v).any(|value| !value.is_finite()) {
        return Err("attention input contains NaN/Inf".into());
    }
    let mut output = vec![0.0f32; query_elements];
    for row in 0..g.queries {
        let visible = g.query_start + row + 1;
        for head in 0..g.query_heads {
            let kv_head = head / (g.query_heads / g.kv_heads);
            let q_base = (row * g.query_heads + head) * g.head_dim;
            let mut weights = Vec::with_capacity(visible);
            for position in 0..visible {
                let cache_base = (position * g.kv_heads + kv_head) * g.head_dim;
                let dot = (0..g.head_dim)
                    .map(|dim| f64::from(q[q_base + dim]) * f64::from(k[cache_base + dim]))
                    .sum::<f64>();
                weights.push(dot * f64::from(scale));
            }
            let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for weight in &mut weights {
                *weight = (*weight - maximum).exp();
            }
            let denominator = weights.iter().sum::<f64>();
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err("non-finite reference softmax denominator".into());
            }
            for dim in 0..g.head_dim {
                let result = weights.iter().enumerate().map(|(position, weight)| {
                    let index = (position * g.kv_heads + kv_head) * g.head_dim + dim;
                    *weight * f64::from(v[index])
                }).sum::<f64>() / denominator;
                if !result.is_finite() {
                    return Err("non-finite reference output".into());
                }
                output[q_base + dim] = result as f32;
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    pub count: usize,
    pub max_abs: f64,
    pub relative_l2: f64,
    pub cosine: f64,
    pub max_head_row_abs: f64,
    pub max_head_row_relative_l2: f64,
    pub worst_head_row: usize,
}

pub fn metrics(got: &[f32], expected: &[f32], head_dim: usize) -> Result<Metrics, String> {
    if got.is_empty() || got.len() != expected.len() || head_dim == 0 || got.len() % head_dim != 0 {
        return Err("metric shape mismatch".into());
    }
    if got.iter().chain(expected).any(|value| !value.is_finite()) {
        return Err("metric tensor contains NaN/Inf".into());
    }
    let (mut error, mut reference_norm, mut got_norm, mut dot) = (0.0, 0.0, 0.0, 0.0);
    let (mut max_abs, mut max_row_abs, mut max_row_rel, mut worst_row) = (0.0f64, 0.0f64, 0.0f64, 0);
    for (row, (actual_row, reference_row)) in got.chunks_exact(head_dim).zip(expected.chunks_exact(head_dim)).enumerate() {
        let (mut row_error, mut row_norm, mut row_abs) = (0.0f64, 0.0f64, 0.0f64);
        for (&actual, &reference) in actual_row.iter().zip(reference_row) {
            let (actual, reference) = (f64::from(actual), f64::from(reference));
            let difference = actual - reference;
            row_error += difference * difference;
            row_norm += reference * reference;
            row_abs = row_abs.max(difference.abs());
            got_norm += actual * actual;
            dot += actual * reference;
        }
        let row_relative = if row_norm == 0.0 {
            if row_error == 0.0 { 0.0 } else { f64::INFINITY }
        } else { (row_error / row_norm).sqrt() };
        if row_relative > max_row_rel { max_row_rel = row_relative; worst_row = row; }
        max_abs = max_abs.max(row_abs);
        max_row_abs = max_row_abs.max(row_abs);
        reference_norm += row_norm;
        error += row_error;
    }
    let relative_l2 = if reference_norm == 0.0 {
        if error == 0.0 { 0.0 } else { f64::INFINITY }
    } else { (error / reference_norm).sqrt() };
    let cosine = if reference_norm == 0.0 || got_norm == 0.0 {
        if error == 0.0 { 1.0 } else { 0.0 }
    } else { dot / (reference_norm * got_norm).sqrt() };
    Ok(Metrics { count: got.len(), max_abs, relative_l2, cosine,
                 max_head_row_abs: max_row_abs, max_head_row_relative_l2: max_row_rel,
                 worst_head_row: worst_row })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Geometry {
        Geometry { queries: 2, valid_keys: 3, capacity: 4, query_heads: 4, kv_heads: 2, head_dim: 1, query_start: 0 }
    }

    #[test]
    fn causal_gqa_uses_only_visible_keys_and_correct_head_group() {
        let q = vec![0.0; 8];
        let k = vec![0.0; 8];
        let v = vec![2.0, 10.0, 4.0, 20.0, 30.0, 90.0, 500.0, 700.0];
        let got = causal_attention(small(), &q, &k, &v, 1.0).unwrap();
        assert_eq!(got, vec![2.0, 2.0, 10.0, 10.0, 3.0, 3.0, 15.0, 15.0]);
    }

    #[test]
    fn shifted_queries_apply_global_cache_positions() {
        let mut g = small(); g.query_start = 1;
        let q = vec![0.0; 8]; let k = vec![0.0; 8];
        let v = vec![2.0, 10.0, 4.0, 20.0, 30.0, 90.0, 500.0, 700.0];
        let got = causal_attention(g, &q, &k, &v, 1.0).unwrap();
        assert_eq!(got, vec![3.0, 3.0, 15.0, 15.0, 12.0, 12.0, 40.0, 40.0]);
    }

    #[test]
    fn stable_softmax_matches_an_analytic_two_key_case() {
        let g = Geometry { queries: 1, valid_keys: 2, capacity: 2, query_heads: 1, kv_heads: 1, head_dim: 1, query_start: 1 };
        let expected = (2.0 + 8.0 * std::f64::consts::E) / (1.0 + std::f64::consts::E);
        let got = causal_attention(g, &[1.0], &[1000.0, 1001.0], &[2.0, 8.0], 1.0).unwrap();
        assert!((f64::from(got[0]) - expected).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_geometry_nonfinite_and_shape_are_rejected() {
        let mut g = small(); g.query_start = 2;
        assert!(causal_attention(g, &[0.0; 8], &[0.0; 8], &[0.0; 8], 1.0).is_err());
        assert!(causal_attention(small(), &[0.0; 7], &[0.0; 8], &[0.0; 8], 1.0).is_err());
        assert!(causal_attention(small(), &[f32::NAN; 8], &[0.0; 8], &[0.0; 8], 1.0).is_err());
        assert!(causal_attention(small(), &[0.0; 8], &[0.0; 8], &[0.0; 8], f32::INFINITY).is_err());
    }

    #[test]
    fn metrics_expose_bad_head_row_and_zero_reference() {
        let mut got = vec![1.0; 8]; got[7] = 2.0;
        let m = metrics(&got, &[1.0; 8], 2).unwrap();
        assert_eq!(m.worst_head_row, 3); assert_eq!(m.max_abs, 1.0);
        assert!(m.max_head_row_relative_l2 > m.relative_l2);
        assert!(!metrics(&[0.0, 0.001], &[0.0; 2], 2).unwrap().relative_l2.is_finite());
        assert_eq!(metrics(&[0.0; 2], &[0.0; 2], 2).unwrap().cosine, 1.0);
        assert!(metrics(&[f32::NAN], &[0.0], 1).is_err());
    }
}
