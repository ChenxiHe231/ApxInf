//! Independent numeric E4M3 reference, with no CUDA/half/dependency imports.
//! Enumerate representable numbers, then choose nearest/ties-even. This does
//! not reproduce a floating-point bit-conversion ladder or GPU intrinsic.

#[derive(Clone, Debug)]
pub struct Oracle {
    positive: Vec<(u8, f64)>,
}

#[derive(Clone, Copy, Debug)]
pub struct Stages {
    pub reciprocal_f32: f32,
    pub multiplied_f32: f32,
    pub divided_f32: f32,
    pub mathematical_f64: f64,
    pub reciprocal_product_code: u8,
    pub direct_division_code: u8,
    pub mathematical_code: u8,
}

impl Oracle {
    pub fn new() -> Self {
        let mut positive = Vec::new();
        for exponent in 0u8..16 {
            for fraction in 0u8..8 {
                // E4M3's final exponent/fraction combination is NaN, not a
                // finite value. The last finite positive value is448.
                if exponent == 15 && fraction == 7 { continue; }
                let value = if exponent == 0 {
                    f64::from(fraction) * 2.0f64.powi(-9)
                } else {
                    f64::from(8 + fraction) * 2.0f64.powi(i32::from(exponent) - 10)
                };
                positive.push((exponent * 8 + fraction, value));
            }
        }
        assert_eq!(positive.len(), 127); // +zero plus126 positive finite codes
        assert_eq!(positive.last().unwrap().1, 448.0);
        Self { positive }
    }

    /// SATFINITE, round nearest/ties-even, preserving negative zero. NaN
    /// inputs are excluded from this contract; infinite scaled intermediates
    /// caused by overflow of finite BF16 are saturated rather than rejected.
    pub fn nearest(&self, value: f64) -> u8 {
        assert!(!value.is_nan(), "NaN is outside the finite-input precision gate");
        let sign = if value.is_sign_negative() { 128 } else { 0 };
        let magnitude = value.abs();
        if magnitude >= 448.0 { return sign | 126; }
        let mut low = 0;
        let mut high = self.positive.len();
        while low < high {
            let mid = (low + high) / 2;
            if self.positive[mid].1 < magnitude { low = mid + 1; } else { high = mid; }
        }
        if low == 0 { return sign; }
        let (left, left_value) = self.positive[low - 1];
        let (right, right_value) = self.positive[low];
        let dl = magnitude - left_value;
        let dr = right_value - magnitude;
        let code = if dl < dr { left } else if dr < dl { right }
                   else if left % 2 == 0 { left } else { right };
        sign | code
    }

    /// The decomposed public cache quantizer uses host F32 reciprocal, then
    /// device F32 multiplication. Exact-division and direct-F32-division are
    /// separate diagnostics; they are not an allowance for GPU byte errors.
    pub fn stages(&self, prepared: f32, scale: f32) -> Stages {
        assert!(prepared.is_finite());
        assert!(scale.is_finite() && scale > 0.0);
        let reciprocal_f32 = 1.0f32 / scale;
        let multiplied_f32 = prepared * reciprocal_f32;
        let divided_f32 = prepared / scale;
        let mathematical_f64 = f64::from(prepared) / f64::from(scale);
        Stages {
            reciprocal_f32, multiplied_f32, divided_f32, mathematical_f64,
            reciprocal_product_code: self.nearest(f64::from(multiplied_f32)),
            direct_division_code: self.nearest(f64::from(divided_f32)),
            mathematical_code: self.nearest(mathematical_f64),
        }
    }
}

pub fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerated_values_and_signed_zero_are_exact() {
        let o = Oracle::new();
        for &(code, value) in &o.positive {
            assert_eq!(o.nearest(value), code);
            assert_eq!(o.nearest(-value), code | 128);
        }
        assert_eq!(o.nearest(0.0), 0);
        assert_eq!(o.nearest(-0.0), 128);
        assert_eq!(o.positive[1].1, 1.0 / 512.0);
        assert_eq!(o.positive[8].1, 1.0 / 64.0);
    }

    #[test]
    fn every_adjacent_tie_uses_even_code_for_both_signs() {
        let o = Oracle::new();
        for pair in o.positive.windows(2) {
            let midpoint = (pair[0].1 + pair[1].1) / 2.0;
            let expected = if pair[0].0 % 2 == 0 { pair[0].0 } else { pair[1].0 };
            assert_eq!(o.nearest(midpoint), expected);
            assert_eq!(o.nearest(-midpoint), expected | 128);
        }
    }

    #[test]
    fn saturation_never_produces_nan_codes() {
        let o = Oracle::new();
        for x in [448.0, 449.0, 65504.0, f64::INFINITY] {
            assert_eq!(o.nearest(x), 126);
            assert_eq!(o.nearest(-x), 254);
        }
    }

    #[test]
    fn finite_bf16_power2_stages_agree_on_every_code() {
        let o = Oracle::new();
        for bits in 0..=u16::MAX {
            if bits & 0x7f80 == 0x7f80 { continue; }
            for scale in [0.25, 0.5, 1.0, 2.0, 4.0] {
                let s = o.stages(bf16_value(bits), scale);
                assert_eq!(s.reciprocal_product_code, s.mathematical_code, "bf16={bits:04x} {scale} {s:?}");
                assert_eq!(s.direct_division_code, s.mathematical_code, "bf16={bits:04x} {scale} {s:?}");
            }
        }
    }

    #[test]
    fn nonpower_reciprocal_and_direct_division_are_distinct_contracts() {
        let s = Oracle::new().stages(bf16_value(0x3b90), 0.3);
        assert_eq!(s.multiplied_f32, 15.0 / 1024.0); // exact E4M3 midpoint
        assert!(s.divided_f32 < s.multiplied_f32);
        assert_eq!(s.reciprocal_product_code, 8); // midpoint -> even
        assert_eq!(s.direct_division_code, 7);    // below midpoint
        assert_eq!(s.mathematical_code, 7);
    }

    #[test]
    fn nonpower_scale_stage_differences_are_reported_separately() {
        let o = Oracle::new();
        for scale in [0.1f32, 0.3, 0.11216518, 0.0322265625] {
            let mut product_vs_math = 0;
            let mut product_vs_division = 0;
            for bits in 0..=u16::MAX {
                if bits & 0x7f80 == 0x7f80 { continue; }
                let s = o.stages(bf16_value(bits), scale);
                product_vs_math += usize::from(s.reciprocal_product_code != s.mathematical_code);
                product_vs_division += usize::from(s.reciprocal_product_code != s.direct_division_code);
            }
            println!("nonpower scale={scale:?} bits={:08x} reciprocal_vs_math={product_vs_math} reciprocal_vs_direct_f32={product_vs_division}", scale.to_bits());
        }
    }
}
