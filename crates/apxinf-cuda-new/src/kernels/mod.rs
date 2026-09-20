//! Compatibility facade for direct-launch operators migrated from the old
//! CUDA crate. New code should prefer [`crate::ops`]; these modules keep
//! existing model call sites buildable while they move family by family.

pub mod activation;
pub mod attention;
pub mod cache;
mod contracts;
pub mod elementwise;
pub mod embedding;
pub mod norm;
pub mod preprocess;
pub mod quantization;
pub mod rope;
