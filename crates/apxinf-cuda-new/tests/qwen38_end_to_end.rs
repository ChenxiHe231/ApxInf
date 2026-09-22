//! The whole Qwen3.8-27B-NVFP4 text model, forward, on one Thor.
//!
//! 64 layers: 48 Gated DeltaNet and 16 full attention, mixed NVFP4/FP8/BF16,
//! embedding through lm_head, greedy token selection. This is the first point
//! where a decode and prefill rate for the *model* exists rather than for a
//! component.
//!
//! What this establishes and what it does not: the model runs, its memory fits,
//! and the token rate is measured. It does **not** establish that the tokens
//! are the right ones. Four modelling assumptions are still unvalidated -- the
//! GDN recurrence, the rotary pairing convention, the mRoPE collapse, and the
//! conv/SiLU ordering -- and each would produce plausible output if wrong. See
//! devlocal/qwen38-nvfp4/reports/STATUS.md. Validating them needs a reference
//! engine that can execute this checkpoint, which thor-3 does not have.
//!
//! ```text
//! APXINF_QWEN38_CHECKPOINT=/path/to/Qwen3.8-27B-NVFP4 \
//!   bash crates/apxinf-cuda-new/test-new.sh \
//!     test -p apxinf-cuda --test qwen38_end_to_end --release \
//!     -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 17408;
const VOCAB: usize = 248320;
const LAYERS: usize = 64;
const FULL_ATTENTION_INTERVAL: usize = 4;
const BLOCK: u32 = 16;
const EPSILON: f32 = 1e-6;

// Full attention
const HEADS: usize = 24;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const ROPE_THETA: f32 = 1.0e7;
const PARTIAL_ROTARY: f32 = 0.25;

// Gated DeltaNet
const GDN_K_HEADS: usize = 16;
const GDN_V_HEADS: usize = 48;
const GDN_HEAD_DIM: usize = 128;
const CONV_WIDTH: usize = 4;
const QKV_WIDTH: usize = 10240; // 16*128 q + 16*128 k + 48*128 v
const Z_WIDTH: usize = 6144;

fn is_full_attention(layer: usize) -> bool {
    (layer + 1) % FULL_ATTENTION_INTERVAL == 0
}

fn checkpoint() -> HashMap<String, Tensor> {
    let path: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
        .expect("set APXINF_QWEN38_CHECKPOINT")
        .into();
    apxinf_loader::safetensors::load_native_path(&path)
        .expect("checkpoint failed to load")
        .0
}

fn cpu_bytes(tensor: &Tensor) -> &[u8] {
    match tensor.storage() {
        apxinf_core::Storage::Cpu(data) => data,
        _ => panic!("expected a CPU tensor"),
    }
}

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes.max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes.max(1)]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn scalar(tensors: &HashMap<String, Tensor>, name: &str) -> f32 {
    tensors[name].to_f32_vec().unwrap()[0]
}

/// An NVFP4 weight with its relaid-out scales and the alpha folding both
/// per-tensor scales.
struct Nvfp4Weight {
    packed: Tensor,
    scales: Tensor,
    input_scale: f32,
    alpha: f32,
}

fn relayout(ctx: &CudaContext, source: &Tensor, rows: usize, k: usize) -> Tensor {
    let bytes = ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap();
    let destination = zeros(ctx, vec![bytes], DType::F8E4M3);
    ops::nvfp4_pack_block_scales(ctx, source, &destination, rows, k, BLOCK).unwrap();
    destination
}

fn load_nvfp4(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    n: usize,
    k: usize,
) -> Nvfp4Weight {
    let packed = upload(
        ctx,
        cpu_bytes(&tensors[&format!("{prefix}.weight")]),
        vec![n, k / 2],
        DType::E2M1Pair,
    );
    let checkpoint_scales = upload(
        ctx,
        cpu_bytes(&tensors[&format!("{prefix}.weight_scale")]),
        vec![n, k / BLOCK as usize],
        DType::F8E4M3,
    );
    let input_scale = scalar(tensors, &format!("{prefix}.input_scale"));
    let weight_scale_2 = scalar(tensors, &format!("{prefix}.weight_scale_2"));
    Nvfp4Weight {
        packed,
        scales: relayout(ctx, &checkpoint_scales, n, k),
        input_scale,
        alpha: input_scale * weight_scale_2,
    }
}

/// Concatenate gate and up into one [2N, K/2] operand.
///
/// Both per-tensor scales are identical in every layer of this checkpoint, so
/// the fused GEMM is exact. Asserted rather than assumed.
fn load_fused_gate_up(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    layer: usize,
) -> Nvfp4Weight {
    let prefix = format!("model.language_model.layers.{layer}.mlp");
    let input_scale = scalar(tensors, &format!("{prefix}.gate_proj.input_scale"));
    let weight_scale_2 = scalar(tensors, &format!("{prefix}.gate_proj.weight_scale_2"));
    assert_eq!(
        input_scale,
        scalar(tensors, &format!("{prefix}.up_proj.input_scale"))
    );
    assert_eq!(
        weight_scale_2,
        scalar(tensors, &format!("{prefix}.up_proj.weight_scale_2"))
    );

    let mut weight = Vec::new();
    weight.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.gate_proj.weight")]));
    weight.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.up_proj.weight")]));
    let packed = upload(
        ctx,
        &weight,
        vec![2 * INTERMEDIATE, HIDDEN / 2],
        DType::E2M1Pair,
    );

    let mut scales = Vec::new();
    scales.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.gate_proj.weight_scale")]));
    scales.extend_from_slice(cpu_bytes(&tensors[&format!("{prefix}.up_proj.weight_scale")]));
    let checkpoint_scales = upload(
        ctx,
        &scales,
        vec![2 * INTERMEDIATE, HIDDEN / BLOCK as usize],
        DType::F8E4M3,
    );

    Nvfp4Weight {
        packed,
        scales: relayout(ctx, &checkpoint_scales, 2 * INTERMEDIATE, HIDDEN),
        input_scale,
        alpha: input_scale * weight_scale_2,
    }
}

/// An FP8 weight transposed to the GEMM's [K, N] contract, with both scalar
/// scales folded into alpha.
struct Fp8Weight {
    weight: Tensor,
    input_scale: f32,
    alpha: f32,
}

fn load_fp8(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    n: usize,
    k: usize,
) -> Fp8Weight {
    // Kept as [N, K], the checkpoint's own orientation and the one the GEMV
    // reads. No transpose: the [K, N] form exists for the GEMM contract, and
    // decode does not use it.
    let weight_scale = scalar(tensors, &format!("{prefix}.weight_scale"));
    let input_scale = scalar(tensors, &format!("{prefix}.input_scale"));
    Fp8Weight {
        weight: upload(
            ctx,
            cpu_bytes(&tensors[&format!("{prefix}.weight")]),
            vec![n, k],
            DType::F8E4M3,
        ),
        input_scale,
        alpha: weight_scale * input_scale,
    }
}

fn load_bf16(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
    dims: Vec<usize>,
) -> Tensor {
    upload(ctx, cpu_bytes(&tensors[name]), dims, DType::BF16)
}

struct AttentionLayer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Fp8Weight,
    k: Fp8Weight,
    v: Fp8Weight,
    o: Fp8Weight,
    q_norm: Tensor,
    k_norm: Tensor,
    gate_up: Nvfp4Weight,
    down: Nvfp4Weight,
}

struct GdnLayer {
    input_norm: Tensor,
    post_norm: Tensor,
    qkv: Fp8Weight,
    z: Fp8Weight,
    out: Fp8Weight,
    in_proj_a: Tensor,
    in_proj_b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    conv_weight: Tensor,
    norm_weight: Tensor,
    gate_up: Nvfp4Weight,
    down: Nvfp4Weight,
}

enum Layer {
    Attention(Box<AttentionLayer>),
    Gdn(Box<GdnLayer>),
}

struct Model {
    embedding: Tensor,
    layers: Vec<Layer>,
    final_norm: Tensor,
    lm_head: Nvfp4Weight,
}

fn load_model(ctx: &CudaContext, tensors: &HashMap<String, Tensor>) -> Model {
    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let prefix = format!("model.language_model.layers.{layer}");
        let input_norm = load_bf16(ctx, tensors, &format!("{prefix}.input_layernorm.weight"), vec![HIDDEN]);
        let post_norm = load_bf16(
            ctx,
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            vec![HIDDEN],
        );
        let gate_up = load_fused_gate_up(ctx, tensors, layer);
        let down = load_nvfp4(
            ctx,
            tensors,
            &format!("{prefix}.mlp.down_proj"),
            HIDDEN,
            INTERMEDIATE,
        );

        if is_full_attention(layer) {
            layers.push(Layer::Attention(Box::new(AttentionLayer {
                input_norm,
                post_norm,
                q: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.q_proj"), 2 * HEADS * HEAD_DIM, HIDDEN),
                k: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.k_proj"), KV_HEADS * HEAD_DIM, HIDDEN),
                v: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.v_proj"), KV_HEADS * HEAD_DIM, HIDDEN),
                o: load_fp8(ctx, tensors, &format!("{prefix}.self_attn.o_proj"), HIDDEN, HEADS * HEAD_DIM),
                q_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.q_norm.weight"), vec![HEAD_DIM]),
                k_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.k_norm.weight"), vec![HEAD_DIM]),
                gate_up,
                down,
            })));
        } else {
            layers.push(Layer::Gdn(Box::new(GdnLayer {
                input_norm,
                post_norm,
                qkv: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_qkv"), QKV_WIDTH, HIDDEN),
                z: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_z"), Z_WIDTH, HIDDEN),
                out: load_fp8(ctx, tensors, &format!("{prefix}.linear_attn.out_proj"), HIDDEN, Z_WIDTH),
                in_proj_a: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"), vec![GDN_V_HEADS, HIDDEN]),
                in_proj_b: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"), vec![GDN_V_HEADS, HIDDEN]),
                a_log: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.A_log"), vec![GDN_V_HEADS]),
                dt_bias: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.dt_bias"), vec![GDN_V_HEADS]),
                conv_weight: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.conv1d.weight"), vec![QKV_WIDTH, CONV_WIDTH]),
                norm_weight: load_bf16(ctx, tensors, &format!("{prefix}.linear_attn.norm.weight"), vec![GDN_HEAD_DIM]),
                gate_up,
                down,
            })));
        }
    }

    Model {
        embedding: load_bf16(ctx, tensors, "model.language_model.embed_tokens.weight", vec![VOCAB, HIDDEN]),
        layers,
        final_norm: load_bf16(ctx, tensors, "model.language_model.norm.weight", vec![HIDDEN]),
        lm_head: load_nvfp4(ctx, tensors, "lm_head", VOCAB, HIDDEN),
    }
}

/// Per-token working buffers, sized for one token of decode.
struct Scratch {
    hidden: Tensor,
    normalized: Tensor,
    fp8_activation: Tensor,
    nvfp4_activation: Tensor,
    nvfp4_scales: Tensor,
    mlp_fused: Tensor,
    mlp_activation: Tensor,
    mlp_scales: Tensor,
    mlp_out: Tensor,
    // attention
    qkv_fused: Tensor,
    query: Tensor,
    query_gate: Tensor,
    attention_out: Tensor,
    attention_fp8: Tensor,
    projected: Tensor,
    positions: Tensor,
    // gdn
    gdn_qkv: Tensor,
    gdn_conv: Tensor,
    gdn_z: Tensor,
    gdn_a: Tensor,
    gdn_b: Tensor,
    gdn_decay: Tensor,
    gdn_beta: Tensor,
    gdn_readout: Tensor,
    gdn_gated: Tensor,
    gdn_fp8: Tensor,
    token: Tensor,
    logits: Tensor,
    next_token: Tensor,
}

impl Scratch {
    fn next_token_host(&self) -> i32 {
        let mut id = [0u8; 4];
        CudaBuffer::from_tensor(&self.next_token)
            .unwrap()
            .copy_to_host(&mut id)
            .unwrap();
        i32::from_le_bytes(id)
    }

    fn new(ctx: &CudaContext) -> Scratch {
        let scale_bytes = |rows: usize, k: usize| {
            vec![ops::nvfp4_scale_buffer_bytes(rows, k, BLOCK).unwrap()]
        };
        Scratch {
            hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            normalized: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            fp8_activation: zeros(ctx, vec![1, HIDDEN], DType::F8E4M3),
            nvfp4_activation: zeros(ctx, vec![1, HIDDEN / 2], DType::E2M1Pair),
            nvfp4_scales: zeros(ctx, scale_bytes(1, HIDDEN), DType::F8E4M3),
            mlp_fused: zeros(ctx, vec![1, 2 * INTERMEDIATE], DType::BF16),
            mlp_activation: zeros(ctx, vec![1, INTERMEDIATE / 2], DType::E2M1Pair),
            mlp_scales: zeros(ctx, scale_bytes(1, INTERMEDIATE), DType::F8E4M3),
            mlp_out: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            qkv_fused: zeros(ctx, vec![1, 2 * HEADS * HEAD_DIM], DType::BF16),
            query: zeros(ctx, vec![1, HEADS, HEAD_DIM], DType::BF16),
            query_gate: zeros(ctx, vec![1, HEADS, HEAD_DIM], DType::BF16),
            attention_out: zeros(ctx, vec![1, HEADS * HEAD_DIM], DType::BF16),
            attention_fp8: zeros(ctx, vec![1, HEADS * HEAD_DIM], DType::F8E4M3),
            projected: zeros(ctx, vec![1, HIDDEN], DType::BF16),
            positions: zeros(ctx, vec![1], DType::I32),
            gdn_qkv: zeros(ctx, vec![1, QKV_WIDTH], DType::BF16),
            gdn_conv: zeros(ctx, vec![QKV_WIDTH], DType::BF16),
            gdn_z: zeros(ctx, vec![1, Z_WIDTH], DType::BF16),
            gdn_a: zeros(ctx, vec![GDN_V_HEADS], DType::BF16),
            gdn_b: zeros(ctx, vec![GDN_V_HEADS], DType::BF16),
            gdn_decay: zeros(ctx, vec![GDN_V_HEADS], DType::F32),
            gdn_beta: zeros(ctx, vec![GDN_V_HEADS], DType::F32),
            gdn_readout: zeros(ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16),
            gdn_gated: zeros(ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16),
            gdn_fp8: zeros(ctx, vec![1, Z_WIDTH], DType::F8E4M3),
            token: zeros(ctx, vec![1], DType::I32),
            logits: zeros(ctx, vec![1, VOCAB], DType::BF16),
            next_token: zeros(ctx, vec![1], DType::I32),
        }
    }
}

/// Recurrent state a GDN layer carries between tokens.
struct GdnState {
    recurrent: Tensor,
    conv_window: Tensor,
}

/// Key/value cache for one full-attention layer.
struct KvCache {
    keys: Tensor,
    values: Tensor,
}

/// Single-token FP8 projection: quantize against the checkpoint's
/// `input_scale`, then one GEMV with both per-tensor scales in alpha.
///
/// The GEMV reaches 249.7 GB/s on the qkv shape against the general GEMM's
/// 146.8 GB/s. A GEMM's tiling is built for large M and leaves half this
/// device's bandwidth on the table at M=1.
fn fp8_projection(
    ctx: &CudaContext,
    weight: &Fp8Weight,
    source: &Tensor,
    quantized: &Tensor,
    output: &mut Tensor,
) {
    ops::quantize_fp8_per_tensor(ctx, source, quantized, weight.input_scale).unwrap();
    ops::fp8_gemv(ctx, &weight.weight, quantized, output, weight.alpha).unwrap();
}

/// residual = residual + MLP(RMSNorm(residual)), with the residual being the
/// running hidden state. Taking it from `scratch` rather than as a separate
/// argument keeps the borrow disjoint.
fn nvfp4_mlp(
    ctx: &CudaContext,
    gate_up: &Nvfp4Weight,
    down: &Nvfp4Weight,
    norm_weight: &Tensor,
    scratch: &mut Scratch,
) {
    ops::nvfp4_quantize_rms_norm(
        ctx,
        &scratch.hidden,
        norm_weight,
        &scratch.nvfp4_activation,
        &scratch.nvfp4_scales,
        EPSILON,
        gate_up.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    ops::gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.nvfp4_activation,
            &scratch.nvfp4_scales,
            &gate_up.packed,
            &gate_up.scales,
            BLOCK,
            gate_up.alpha,
            &mut scratch.mlp_fused,
        ),
    )
    .unwrap();
    ops::nvfp4_quantize_swiglu(
        ctx,
        &scratch.mlp_fused,
        &scratch.mlp_activation,
        &scratch.mlp_scales,
        down.input_scale,
        BLOCK,
        ops::ScaleLayout::GemmAtom,
    )
    .unwrap();
    ops::gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.mlp_activation,
            &scratch.mlp_scales,
            &down.packed,
            &down.scales,
            BLOCK,
            down.alpha,
            &mut scratch.mlp_out,
        ),
    )
    .unwrap();
    ops::add_into(ctx, &scratch.mlp_out, &scratch.hidden).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn decode_step(
    ctx: &CudaContext,
    model: &Model,
    scratch: &mut Scratch,
    gdn_states: &mut [GdnState],
    kv_caches: &mut [KvCache],
    position: usize,
) {
    ops::embedding_gather(ctx, &model.embedding, &scratch.token, &scratch.hidden).unwrap();

    let rotary = ops::rotary_dim(HEAD_DIM, PARTIAL_ROTARY);
    let mut gdn_index = 0usize;
    let mut attention_index = 0usize;

    for (layer_index, layer) in model.layers.iter().enumerate() {
        match layer {
            Layer::Attention(attention) => {
                let cache = &mut kv_caches[attention_index];
                attention_index += 1;

                ops::rms_norm(ctx, &scratch.hidden, &attention.input_norm, &scratch.normalized, EPSILON).unwrap();
                fp8_projection(ctx, &attention.q, &scratch.normalized, &scratch.fp8_activation, &mut scratch.qkv_fused);
                let fused_heads = view(&scratch.qkv_fused, vec![1, HEADS, 2 * HEAD_DIM], DType::BF16);
                ops::split_query_and_gate(ctx, &fused_heads, &scratch.query, &scratch.query_gate).unwrap();

                // k and v project directly into this token's cache slot.
                let mut key_slot = cache_slot(&cache.keys, position, vec![1, KV_HEADS * HEAD_DIM]);
                let mut value_slot = cache_slot(&cache.values, position, vec![1, KV_HEADS * HEAD_DIM]);
                fp8_projection(ctx, &attention.k, &scratch.normalized, &scratch.fp8_activation, &mut key_slot);
                fp8_projection(ctx, &attention.v, &scratch.normalized, &scratch.fp8_activation, &mut value_slot);

                let key_heads = cache_slot(&cache.keys, position, vec![KV_HEADS, HEAD_DIM]);
                let query_heads = view(&scratch.query, vec![HEADS, HEAD_DIM], DType::BF16);
                ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, EPSILON).unwrap();
                ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, EPSILON).unwrap();

                let query_tokens = view(&scratch.query, vec![1, HEADS, HEAD_DIM], DType::BF16);
                let key_tokens = cache_slot(&cache.keys, position, vec![1, KV_HEADS, HEAD_DIM]);
                ops::partial_rope(ctx, &query_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();
                ops::partial_rope(ctx, &key_tokens, &scratch.positions, rotary, ROPE_THETA).unwrap();

                // The cache is allocated at full capacity; attention reads only
                // the tokens written so far.
                let valid = position + 1;
                let keys = view(&cache.keys, vec![1, capacity_of(&cache.keys), KV_HEADS, HEAD_DIM], DType::BF16);
                let values = view(&cache.values, vec![1, capacity_of(&cache.values), KV_HEADS, HEAD_DIM], DType::BF16);
                let query_4d = view(&scratch.query, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
                let mut out_4d = view(&scratch.attention_out, vec![1, 1, HEADS, HEAD_DIM], DType::BF16);
                let mut args = ops::KvCacheAttentionArgs::new(&query_4d, &keys, &values, &mut out_4d);
                args.valid_key_tokens = valid;
                args.query_start = position;
                ops::kv_cache_attention(ctx, args).unwrap();

                let gate_flat = view(&scratch.query_gate, vec![1, HEADS * HEAD_DIM], DType::BF16);
                ops::apply_swish_gate(ctx, &scratch.attention_out, &gate_flat).unwrap();
                fp8_projection(ctx, &attention.o, &scratch.attention_out, &scratch.attention_fp8, &mut scratch.projected);
                ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();

                nvfp4_mlp(ctx, &attention.gate_up, &attention.down, &attention.post_norm, scratch);
            }
            Layer::Gdn(gdn) => {
                let state = &mut gdn_states[gdn_index];
                gdn_index += 1;

                ops::rms_norm(ctx, &scratch.hidden, &gdn.input_norm, &scratch.normalized, EPSILON).unwrap();
                fp8_projection(ctx, &gdn.qkv, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_qkv);
                fp8_projection(ctx, &gdn.z, &scratch.normalized, &scratch.fp8_activation, &mut scratch.gdn_z);

                let qkv_flat = view(&scratch.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
                ops::gdn_causal_conv_step(ctx, &state.conv_window, &qkv_flat, &gdn.conv_weight, &scratch.gdn_conv).unwrap();

                let (q, k, v) = split_gdn_qkv(ctx, &scratch.gdn_conv);
                ops::gdn_l2_normalize_heads(ctx, &q, EPSILON).unwrap();
                ops::gdn_l2_normalize_heads(ctx, &k, EPSILON).unwrap();

                bf16_matvec(ctx, &gdn.in_proj_a, &scratch.normalized, &scratch.gdn_a);
                bf16_matvec(ctx, &gdn.in_proj_b, &scratch.normalized, &scratch.gdn_b);
                ops::gdn_decay_and_beta(ctx, &scratch.gdn_a, &scratch.gdn_b, &gdn.a_log, &gdn.dt_bias, &scratch.gdn_decay, &scratch.gdn_beta).unwrap();

                ops::gdn_recurrent_step(ctx, &state.recurrent, &q, &k, &v, &scratch.gdn_decay, &scratch.gdn_beta, &scratch.gdn_readout, GDN_K_HEADS).unwrap();
                ops::gdn_gated_norm(ctx, &scratch.gdn_readout, &gdn_z_heads(ctx, &scratch.gdn_z), &gdn.norm_weight, &scratch.gdn_gated, EPSILON).unwrap();

                let flat = flatten(ctx, &scratch.gdn_gated, Z_WIDTH);
                fp8_projection(ctx, &gdn.out, &flat, &scratch.gdn_fp8, &mut scratch.projected);
                ops::add_into(ctx, &scratch.projected, &scratch.hidden).unwrap();

                nvfp4_mlp(ctx, &gdn.gate_up, &gdn.down, &gdn.post_norm, scratch);
            }
        }
        let _ = layer_index;
    }

    ops::rms_norm(ctx, &scratch.hidden, &model.final_norm, &scratch.normalized, EPSILON).unwrap();
    ops::nvfp4_quantize_activation(ctx, &scratch.normalized, &scratch.nvfp4_activation, &scratch.nvfp4_scales, model.lm_head.input_scale, BLOCK, ops::ScaleLayout::GemmAtom).unwrap();
    ops::gemm(
        ctx,
        ops::GemmArgs::nvfp4(
            &scratch.nvfp4_activation,
            &scratch.nvfp4_scales,
            &model.lm_head.packed,
            &model.lm_head.scales,
            BLOCK,
            model.lm_head.alpha,
            &mut scratch.logits,
        ),
    )
    .unwrap();
    ops::argmax(ctx, &scratch.logits, &scratch.next_token).unwrap();
}

// --- small helpers that reinterpret existing device storage -----------------

fn capacity_of(cache: &Tensor) -> usize {
    cache.shape().dims()[1]
}

fn view(tensor: &Tensor, dims: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

fn flatten(_ctx: &CudaContext, tensor: &Tensor, width: usize) -> Tensor {
    view(tensor, vec![1, width], DType::BF16)
}

fn gdn_z_heads(_ctx: &CudaContext, z: &Tensor) -> Tensor {
    view(z, vec![GDN_V_HEADS, GDN_HEAD_DIM], DType::BF16)
}

/// q, k and v live in one [10240] projection: 16*128 q, then 16*128 k, then
/// 48*128 v. The views share storage rather than copying.
fn split_gdn_qkv(_ctx: &CudaContext, qkv: &Tensor) -> (Tensor, Tensor, Tensor) {
    let buffer = CudaBuffer::from_tensor(qkv).unwrap();
    let element = DType::BF16.size_in_bytes();
    let q_len = GDN_K_HEADS * GDN_HEAD_DIM;
    let v_len = GDN_V_HEADS * GDN_HEAD_DIM;
    let q = buffer
        .view(0, q_len * element)
        .unwrap()
        .as_tensor(Shape::new(vec![GDN_K_HEADS, GDN_HEAD_DIM]), DType::BF16)
        .unwrap();
    let k = buffer
        .view(q_len * element, q_len * element)
        .unwrap()
        .as_tensor(Shape::new(vec![GDN_K_HEADS, GDN_HEAD_DIM]), DType::BF16)
        .unwrap();
    let v = buffer
        .view(2 * q_len * element, v_len * element)
        .unwrap()
        .as_tensor(Shape::new(vec![GDN_V_HEADS, GDN_HEAD_DIM]), DType::BF16)
        .unwrap();
    (q, k, v)
}

/// [heads, hidden] x [1, hidden] -> [1, heads], in BF16.
fn bf16_matvec(ctx: &CudaContext, weight: &Tensor, input: &Tensor, output: &Tensor) {
    let dims = weight.shape().dims().to_vec();
    let transposed = view(weight, vec![dims[1], dims[0]], DType::BF16);
    let mut out = view(output, vec![1, dims[0]], DType::BF16);
    ops::gemm(ctx, ops::GemmArgs::new(input, &transposed, &mut out)).unwrap();
}

/// A view of one token's slot in a KV cache, shaped for the projection that
/// fills it. Writing the projection straight into the cache avoids a
/// device-to-device copy per layer per token.
fn cache_slot(cache_tensor: &Tensor, position: usize, dims: Vec<usize>) -> Tensor {
    let element = DType::BF16.size_in_bytes();
    let stride = KV_HEADS * HEAD_DIM * element;
    CudaBuffer::from_tensor(cache_tensor)
        .unwrap()
        .view(position * stride, stride)
        .unwrap()
        .as_tensor(Shape::new(dims), DType::BF16)
        .unwrap()
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn full_model_decodes_and_is_timed() {
    let ctx = CudaContext::new(0).unwrap();

    let start = Instant::now();
    let tensors = checkpoint();
    println!("checkpoint read:     {:6.2} s", start.elapsed().as_secs_f64());

    let start = Instant::now();
    let model = load_model(&ctx, &tensors);
    ctx.synchronize().unwrap();
    println!("weights on device:   {:6.2} s", start.elapsed().as_secs_f64());
    drop(tensors);

    let capacity = 64usize;
    let mut gdn_states: Vec<GdnState> = (0..LAYERS - LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| GdnState {
            recurrent: zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
            conv_window: zeros(&ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
        })
        .collect();
    let mut kv_caches: Vec<KvCache> = (0..LAYERS / FULL_ATTENTION_INTERVAL)
        .map(|_| KvCache {
            keys: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
            values: zeros(&ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        })
        .collect();
    let mut scratch = Scratch::new(&ctx);

    // Warm up: the first step tunes every distinct GEMM shape.
    decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, 0);
    ctx.synchronize().unwrap();

    // Sanity before timing. A forward pass that finishes is not the same as a
    // forward pass that computed anything: NaNs propagate silently, a dead
    // layer yields constant logits, and a broken recurrence still returns a
    // number. None of that shows up in a latency measurement.
    {
        let mut bytes = vec![0u8; VOCAB * 2];
        CudaBuffer::from_tensor(&scratch.logits)
            .unwrap()
            .copy_to_host(&mut bytes)
            .unwrap();
        let logits: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|v| half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
            .collect();

        let non_finite = logits.iter().filter(|v| !v.is_finite()).count();
        let finite: Vec<f32> = logits.iter().copied().filter(|v| v.is_finite()).collect();
        let max = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = finite.iter().copied().fold(f32::INFINITY, f32::min);
        let mean = finite.iter().sum::<f32>() / finite.len().max(1) as f32;
        let variance = finite.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / finite.len().max(1) as f32;
        let distinct = {
            let mut sorted = finite.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted.dedup();
            sorted.len()
        };
        println!(
            "\nlogits: min={min:.3} max={max:.3} mean={mean:.3} sd={:.3}\n        \
             non-finite={non_finite}  distinct values={distinct} of {VOCAB}",
            variance.sqrt()
        );

        assert_eq!(non_finite, 0, "logits contain NaN or Inf");
        assert!(distinct > 1000, "logits are nearly constant -- a layer is dead");
        assert!(
            max.abs() < 1.0e4 && variance.sqrt() > 1.0e-3,
            "logit scale is implausible: sd={}, max={max}",
            variance.sqrt()
        );
    }

    // Greedy-decode a short run and look at what comes out. Repeating a single
    // token forever is the classic signature of a model that runs but does not
    // compute -- worth catching here rather than in a latency table.
    let mut produced = Vec::new();
    for position in 1..=8usize {
        // Autoregressive: feed the previous step's token and advance the
        // position, since decode_step reads scratch.token and scratch.positions.
        CudaBuffer::from_tensor(&scratch.token)
            .unwrap()
            .copy_from_host(&scratch.next_token_host().to_le_bytes())
            .unwrap();
        CudaBuffer::from_tensor(&scratch.positions)
            .unwrap()
            .copy_from_host(&(position as i32).to_le_bytes())
            .unwrap();
        decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, position);
        ctx.synchronize().unwrap();
        produced.push(scratch.next_token_host());
    }
    println!("greedy token ids: {produced:?}");
    let unique: std::collections::HashSet<_> = produced.iter().collect();
    println!("  {} distinct of {}", unique.len(), produced.len());
    for id in &produced {
        assert!(
            *id >= 0 && (*id as usize) < VOCAB,
            "token id {id} is outside the vocabulary"
        );
    }

    let steps = 16usize;
    let start = Instant::now();
    for position in 1..=steps {
        decode_step(&ctx, &model, &mut scratch, &mut gdn_states, &mut kv_caches, position);
    }
    ctx.synchronize().unwrap();
    let per_token = start.elapsed().as_secs_f64() / steps as f64;

    println!(
        "\ndecode: {:7.2} ms/token   {:6.2} tok/s   ({steps} steps, batch 1)",
        per_token * 1e3,
        1.0 / per_token
    );

    // Where the time goes. Each phase synchronizes, so the sum exceeds the
    // pipelined total above; the point is the ratio between phases.
    macro_rules! phase {
        ($label:expr, $iterations:expr, $body:block) => {{
            $body
            ctx.synchronize().unwrap();
            let start = Instant::now();
            for _ in 0..$iterations {
                $body
            }
            ctx.synchronize().unwrap();
            let each = start.elapsed().as_secs_f64() / $iterations as f64;
            println!("  {:30} {:8.3} ms  x{:3} = {:7.2} ms",
                     $label, each * 1e3, $iterations, each * 1e3);
            each
        }};
    }

    println!("\nper-phase (synchronized, so these over-count):");
    let mut probe = Scratch::new(&ctx);
    let gdn_layer = model.layers.iter().find_map(|l| match l {
        Layer::Gdn(g) => Some(g),
        _ => None,
    }).unwrap();

    let qkv_quantized = view(&probe.fp8_activation, vec![1, HIDDEN], DType::F8E4M3);
    let mut qkv_out = view(&probe.gdn_qkv, vec![1, QKV_WIDTH], DType::BF16);
    // The GEMM contract reads [K, N]; reinterpreting the [N, K] weight gives
    // wrong numbers but the same sequential sweep, which is what is timed.
    let gemm_weight = view(&gdn_layer.qkv.weight, vec![HIDDEN, QKV_WIDTH], DType::F8E4M3);
    let qkv_each = phase!("fp8 qkv via GEMM", 50, {
        let mut args = ops::GemmArgs::new(&qkv_quantized, &gemm_weight, &mut qkv_out);
        args.quantization = ops::GemmQuantization::Fp8UnitScale;
        args.alpha = gdn_layer.qkv.alpha;
        ops::gemm(&ctx, args).unwrap();
    });

    // The decode path now uses the GEMV; keep the comparison so a regression
    // in either shows up.
    let gemv_weight = view(&gdn_layer.qkv.weight, vec![QKV_WIDTH, HIDDEN], DType::F8E4M3);
    let gemv_activation = view(&probe.fp8_activation, vec![HIDDEN], DType::F8E4M3);
    let gemv_out = view(&probe.gdn_qkv, vec![QKV_WIDTH], DType::BF16);
    let gemv_each = phase!("fp8 qkv via vectorized GEMV", 50, {
        ops::fp8_gemv(&ctx, &gemv_weight, &gemv_activation, &gemv_out,
                      gdn_layer.qkv.alpha).unwrap();
    });
    let weight_bytes = (QKV_WIDTH * HIDDEN) as f64;
    println!(
        "    GEMM {:6.1} GB/s   GEMV {:6.1} GB/s   ({:+.0}%)",
        weight_bytes / qkv_each / 1e9,
        weight_bytes / gemv_each / 1e9,
        (qkv_each / gemv_each - 1.0) * 100.0
    );

    let state = zeros(&ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32);
    let (gq, gk, gv) = split_gdn_qkv(&ctx, &probe.gdn_conv);
    let recurrent_each = phase!("gdn recurrent step", 50, {
        ops::gdn_recurrent_step(&ctx, &state, &gq, &gk, &gv, &probe.gdn_decay,
                                &probe.gdn_beta, &probe.gdn_readout, GDN_K_HEADS).unwrap();
    });

    let mlp_each = phase!("nvfp4 mlp block", 50, {
        nvfp4_mlp(&ctx, &gdn_layer.gate_up, &gdn_layer.down, &gdn_layer.post_norm, &mut probe);
    });

    let mut logits = zeros(&ctx, vec![1, VOCAB], DType::BF16);
    let head_activation = zeros(&ctx, vec![1, HIDDEN / 2], DType::E2M1Pair);
    let head_scales = zeros(&ctx, vec![ops::nvfp4_scale_buffer_bytes(1, HIDDEN, BLOCK).unwrap()], DType::F8E4M3);
    let head_each = phase!("lm_head", 20, {
        ops::gemm(&ctx, ops::GemmArgs::nvfp4(&head_activation, &head_scales,
                  &model.lm_head.packed, &model.lm_head.scales, BLOCK,
                  model.lm_head.alpha, &mut logits)).unwrap();
    });

    println!(
        "\n  64 MLP blocks              {:7.2} ms\n  \
         48 gdn recurrent steps     {:7.2} ms\n  \
         48 qkv projections         {:7.2} ms\n  \
         1 lm_head                  {:7.2} ms",
        mlp_each * 64.0 * 1e3,
        recurrent_each * 48.0 * 1e3,
        qkv_each * 48.0 * 1e3,
        head_each * 1e3
    );
}
