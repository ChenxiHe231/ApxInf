//! `LlmTrait` and `AutoModel` surface for the exact Qwen3.5-35B-A3B runtime.
//!
//! This wrapper owns one transactional sequence session. The serving worker is
//! batch one, so a request resets and reuses that storage while model weights
//! and cache allocations remain resident on CUDA.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{
    Backend, DType, Device, Error, Result, Shape, Tensor, TokenSampler, TokenSamplingSpec,
};
use apxinf_cuda_new::{transfers, CudaBuffer, CudaContext};
use apxinf_loader::ModelConfig;

use crate::llm_trait::{
    generate_streaming_with_options, GeneratedToken, GenerationOutput, GenerationRequest, LlmInput,
    LlmTrait, SpeculativeGenerationStats,
};
use crate::profiling::GenerationProfile;
use crate::GenerationOptions;

use super::attention_backend::{Qwen35AttentionCache, Qwen35Fp8AttentionBackend};
use super::config::Qwen35MoeConfig;
use super::decoder::Qwen35Decoder;
use super::device_weights::{
    load_qwen35_device_weights_from_path_with_options, upload_qwen35_device_weights_with_options,
    DeviceMixerWeights, Qwen35DeviceWeights,
};
use super::execution::{Qwen35ExecutionOptions, Qwen35KvCacheDType};
use super::loader::load_qwen35_runtime_weights;
use super::mtp_runtime::{
    MtpGreedyPolicy, Qwen35ContextSnapshot, Qwen35MtpCoordinator, Qwen35MtpRuntime, Qwen35MtpState,
};
use super::mtp_schedule::QWEN35_MTP_DRAFT_STEPS;
use super::runtime_weights::KvCacheScales;
use super::session::{Qwen35Session, Qwen35SessionSnapshot};

/// Default cache admission for direct `AutoModel` callers. OpenAI serving
/// supplies its prompt/output admission plus MTP lookahead through
/// `LoadOptions::text_kv_capacity` instead of relying on this default.
pub const DEFAULT_KV_CAPACITY: usize = 8_192;
const SCALAR_PREFIX_CACHE_ENTRIES: usize = 4;
/// One exact first-turn checkpoint plus one exact tool-follow-up checkpoint.
/// Each entry owns a complete device snapshot; the two-entry LRU recycles the
/// prior tool-turn allocation instead of stalling on a third allocation.
// Retain the stable first-turn checkpoint plus the common tool-call variants.
// Greedy NVFP4 execution can differ by one token near close logits; a two-entry
// LRU made the corresponding tool-result checkpoints evict each other and
// forced an otherwise avoidable ~430-token prefill every other request.
const MTP_PROMPT_PREFIX_CACHE_ENTRIES: usize = 4;
const MTP_VOLATILE_TOOL_TAIL_TOKENS: usize = 64;
const MTP_TOOL_PROMPT_MIN_TOKENS: usize = 9_000;
const ADAPTIVE_MTP_HISTORY: usize = 16;
const ADAPTIVE_MTP_WARMUP_ROUNDS: usize = 3;
const ADAPTIVE_MTP_SCALAR_DECISIONS: usize = 6;

#[derive(Clone, Copy, Debug)]
struct MtpRoundObservation {
    depth: usize,
    accepted: usize,
    elapsed_ms: f64,
}

/// Request-local cost policy. Acceptance probabilities use every round that
/// actually proposed the corresponding depth; latency uses direct samples
/// when available and a shallow linear projection while a depth is untried.
/// The scalar threshold is configurable so the decision remains tied to a
/// measured Thor scalar baseline instead of a hard-coded acceptance heuristic.
struct AdaptiveMtpController {
    observations: VecDeque<MtpRoundObservation>,
    rounds: usize,
    last_depth: usize,
    scalar_tpot_ms: f64,
    fixed_depth: Option<usize>,
}

impl AdaptiveMtpController {
    fn new(thor_dynamic_fp8_decode: bool) -> Self {
        let (fixed_depth, scalar_tpot_ms) = if thor_dynamic_fp8_decode {
            (Some(QWEN35_MTP_DRAFT_STEPS), 19.42)
        } else {
            let fixed_depth = std::env::var("APXINF_QWEN35_MTP_FIXED_DEPTH")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|depth| (1..=QWEN35_MTP_DRAFT_STEPS).contains(depth));
            let scalar_tpot_ms = std::env::var("APXINF_QWEN35_SCALAR_TPOT_MS")
                .ok()
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(19.42);
            (fixed_depth, scalar_tpot_ms)
        };
        Self {
            observations: VecDeque::with_capacity(ADAPTIVE_MTP_HISTORY),
            rounds: 0,
            last_depth: QWEN35_MTP_DRAFT_STEPS,
            scalar_tpot_ms,
            fixed_depth,
        }
    }

    fn next_depth(&self) -> usize {
        if let Some(depth) = self.fixed_depth {
            return depth;
        }
        if self.rounds < ADAPTIVE_MTP_WARMUP_ROUNDS {
            return QWEN35_MTP_DRAFT_STEPS;
        }
        // A periodic full-depth probe prevents an early low-acceptance window
        // from permanently hiding later high-confidence spans.
        if self.rounds % 12 == 11 {
            return QWEN35_MTP_DRAFT_STEPS;
        }

        let mut best_depth = 1usize;
        let mut best_tpot = f64::INFINITY;
        for depth in 1..=QWEN35_MTP_DRAFT_STEPS {
            let expected_tokens = 1.0
                + (0..depth)
                    .map(|proposal_depth| self.prefix_acceptance(proposal_depth))
                    .sum::<f64>();
            let tpot = self.estimated_round_ms(depth) / expected_tokens;
            if tpot < best_tpot {
                best_tpot = tpot;
                best_depth = depth;
            }
        }
        if self.rounds >= ADAPTIVE_MTP_SCALAR_DECISIONS
            && best_tpot >= self.scalar_tpot_ms
        {
            0
        } else {
            best_depth
        }
    }

    fn observe(&mut self, depth: usize, accepted: usize, elapsed_ms: f64) {
        if self.observations.len() == ADAPTIVE_MTP_HISTORY {
            self.observations.pop_front();
        }
        self.observations.push_back(MtpRoundObservation {
            depth,
            accepted,
            elapsed_ms,
        });
        self.rounds += 1;
        self.last_depth = depth;
    }

    fn prefix_acceptance(&self, proposal_depth: usize) -> f64 {
        let mut opportunities = 0usize;
        let mut accepted = 0usize;
        for observation in &self.observations {
            if observation.depth > proposal_depth {
                opportunities += 1;
                accepted += usize::from(observation.accepted > proposal_depth);
            }
        }
        // Beta(1,1) smoothing avoids selecting a depth from one lucky round.
        (accepted as f64 + 1.0) / (opportunities as f64 + 2.0)
    }

    fn estimated_round_ms(&self, depth: usize) -> f64 {
        let direct = self
            .observations
            .iter()
            .filter(|observation| observation.depth == depth)
            .map(|observation| observation.elapsed_ms)
            .collect::<Vec<_>>();
        if !direct.is_empty() {
            return direct.iter().sum::<f64>() / direct.len() as f64;
        }
        let normalized = self
            .observations
            .iter()
            .map(|observation| observation.elapsed_ms / depth_cost(observation.depth))
            .collect::<Vec<_>>();
        if normalized.is_empty() {
            return self.scalar_tpot_ms * depth_cost(depth);
        }
        normalized.iter().sum::<f64>() / normalized.len() as f64 * depth_cost(depth)
    }
}

fn depth_cost(depth: usize) -> f64 {
    // The target verifier dominates the round; draft work adds a smaller
    // approximately linear component. Online direct samples replace this
    // projection as soon as a depth is exercised.
    1.0 + 0.24 * depth as f64
}

type ContextSnapshot = Qwen35ContextSnapshot<Qwen35AttentionCache, Qwen35MtpState>;
type ScalarContextSnapshot = Qwen35SessionSnapshot<Qwen35AttentionCache>;

struct SystemPrefixCache {
    tokens: Vec<u32>,
    snapshot: ContextSnapshot,
    capacity: usize,
}

struct ScalarSystemPrefixCache {
    tokens: Vec<u32>,
    snapshot: ScalarContextSnapshot,
    capacity: usize,
}

/// Exact text runtime for `txn545/Qwen3.5-35B-A3B-NVFP4`.
///
/// MTP weights are loaded and device-resident. Exact neutral greedy generation
/// uses the native MTP coordinator; sampling, filters, penalties and log-prob
/// requests reuse the same target session through the scalar generation path.
pub struct Qwen35Moe {
    ctx: CudaContext,
    backend: Arc<dyn Backend>,
    decoder: Qwen35Decoder,
    weights: Qwen35DeviceWeights,
    runtime: Qwen35MtpCoordinator<Qwen35Fp8AttentionBackend, Qwen35MtpRuntime>,
    vocab_size: usize,
    reset_error: Option<String>,
    context_reuse_enabled: bool,
    system_prefix_cache: Option<SystemPrefixCache>,
    /// Exact prompt checkpoints in most-recently-used order. Checkpoints stop
    /// one token before the rendered prompt end: restoring one and appending
    /// that final token reconstructs the first target token and a fresh MTP
    /// schedule without re-evaluating the stable prompt body.
    mtp_prompt_prefix_caches: VecDeque<SystemPrefixCache>,
    /// Token identity for the live MTP endpoint. The batch-one serving worker
    /// leaves target/GDN/MTP state resident after generation, so an immediate
    /// tool-result follow-up can append without copying a second KV snapshot.
    mtp_live_conversation_prefix: Option<Vec<u32>>,
    /// Sampling-only prefix snapshots in most-recently-used order. AIOS tool
    /// bindings can render different system prefixes; retaining a small,
    /// bounded set prevents those bindings from evicting one another on every
    /// request without allowing device memory to grow without limit.
    scalar_system_prefix_caches: VecDeque<ScalarSystemPrefixCache>,
    last_reused_tokens: usize,
    /// Reused destination for decode logits. This used to be a fresh
    /// cudaMalloc plus memset per decode step, freed on drop; on a HugeTLB
    /// pool that free synchronizes, so it was a device stall on every token.
    /// Keyed by row count so a shape change still reallocates.
    decode_logits_cache: RefCell<Option<(usize, Tensor)>>,
}

impl Qwen35Moe {
    /// Validate the exact checkpoint contract, upload all text/MTP tensors and
    /// allocate one batch-one transactional target session.
    pub fn from_checkpoint(
        config: Qwen35MoeConfig,
        tensors: HashMap<String, Tensor>,
        device: Device,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
    ) -> Result<Self> {
        Self::from_checkpoint_with_options(
            config,
            tensors,
            device,
            backend,
            kv_capacity,
            Qwen35ExecutionOptions::default(),
        )
    }

    pub fn from_checkpoint_with_options(
        config: Qwen35MoeConfig,
        tensors: HashMap<String, Tensor>,
        device: Device,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
        execution: Qwen35ExecutionOptions,
    ) -> Result<Self> {
        config.validate()?;
        let Device::Cuda(ordinal) = device else {
            return Err(Error::Other(
                "Qwen3.5-35B-A3B NVFP4 requires a CUDA device".into(),
            ));
        };
        if kv_capacity == 0 || kv_capacity > config.text_config.max_position_embeddings {
            return Err(Error::Other(format!(
                "Qwen3.5 KV capacity must be in 1..={}, got {kv_capacity}",
                config.text_config.max_position_embeddings
            )));
        }

        let ctx = CudaContext::new(ordinal).map_err(Error::Cuda)?;
        let host_weights = load_qwen35_runtime_weights(&config, tensors)?;
        let weights =
            upload_qwen35_device_weights_with_options(&ctx, &config, host_weights, execution)?;
        Self::from_device_weights(config, ctx, weights, backend, kv_capacity, execution, None)
    }

    /// Load the exact checkpoint through the layer-streaming SafeTensors path.
    /// This validates all headers first, never materializes visual tensors and
    /// releases each host-side language layer after its CUDA upload.
    pub fn from_checkpoint_path(
        config: Qwen35MoeConfig,
        path: &Path,
        device: Device,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
    ) -> Result<Self> {
        Self::from_checkpoint_path_with_options(
            config,
            path,
            device,
            backend,
            kv_capacity,
            Qwen35ExecutionOptions::default(),
        )
    }

    pub fn from_checkpoint_path_with_options(
        config: Qwen35MoeConfig,
        path: &Path,
        device: Device,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
        execution: Qwen35ExecutionOptions,
    ) -> Result<Self> {
        Self::from_checkpoint_path_with_runtime_options(
            config,
            path,
            device,
            backend,
            kv_capacity,
            execution,
            true,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_checkpoint_path_with_runtime_options(
        config: Qwen35MoeConfig,
        path: &Path,
        device: Device,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
        execution: Qwen35ExecutionOptions,
        autotune: bool,
        tuning_path: Option<PathBuf>,
        kv_cache_scales: Option<KvCacheScales>,
    ) -> Result<Self> {
        config.validate()?;
        let Device::Cuda(ordinal) = device else {
            return Err(Error::Other(
                "Qwen3.5-35B-A3B NVFP4 requires a CUDA device".into(),
            ));
        };
        if kv_capacity == 0 || kv_capacity > config.text_config.max_position_embeddings {
            return Err(Error::Other(format!(
                "Qwen3.5 KV capacity must be in 1..={}, got {kv_capacity}",
                config.text_config.max_position_embeddings
            )));
        }

        let ctx = CudaContext::new_with_options(ordinal, autotune, tuning_path.as_deref())
            .map_err(Error::Cuda)?;
        let weights =
            load_qwen35_device_weights_from_path_with_options(&ctx, &config, path, execution)?;
        Self::from_device_weights(
            config,
            ctx,
            weights,
            backend,
            kv_capacity,
            execution,
            kv_cache_scales,
        )
    }

    fn from_device_weights(
        config: Qwen35MoeConfig,
        ctx: CudaContext,
        mut weights: Qwen35DeviceWeights,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
        execution: Qwen35ExecutionOptions,
        kv_cache_scales: Option<KvCacheScales>,
    ) -> Result<Self> {
        match (execution.kv_cache, kv_cache_scales) {
            (Qwen35KvCacheDType::Bf16, None) => {}
            (Qwen35KvCacheDType::Fp8Dynamic, None) => {}
            (Qwen35KvCacheDType::Fp8, Some(_)) => {}
            (Qwen35KvCacheDType::Fp8, None) => {
                return Err(Error::Other(
                    "Qwen3.6 FP8 KV cache requires TensorRT-compatible static scales".into(),
                ));
            }
            (Qwen35KvCacheDType::Bf16, Some(_)) => {
                return Err(Error::Other(
                    "Qwen3.6 BF16 KV cache must not carry FP8 scales".into(),
                ));
            }
            (Qwen35KvCacheDType::Fp8Dynamic, Some(_)) => {
                return Err(Error::Other(
                    "Qwen3.6 dynamic FP8 KV cache must not carry fixed scales".into(),
                ));
            }
        }
        for layer in &mut weights.layers {
            if let DeviceMixerWeights::FullAttention(attention) = &mut layer.mixer {
                attention.kv_cache_scales = kv_cache_scales;
            }
        }
        let thor_dynamic_fp8_decode =
            execution.uses_thor_dynamic_fp8_decode(ctx.device_id(), kv_capacity);
        let decoder = Qwen35Decoder::with_execution(&config, thor_dynamic_fp8_decode)?;
        let attention =
            Qwen35Fp8AttentionBackend::with_kv_cache(&config, execution.kv_cache, kv_cache_scales)?;
        let target = Qwen35Session::new(&ctx, &decoder, &weights, attention, kv_capacity)?;
        let mtp = Qwen35MtpRuntime::new(&config)?;
        let runtime = Qwen35MtpCoordinator::new(&ctx, target, mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;

        Ok(Self {
            ctx,
            backend,
            decoder,
            weights,
            runtime,
            vocab_size: config.text_config.vocab_size,
            reset_error: None,
            context_reuse_enabled: false,
            system_prefix_cache: None,
            mtp_prompt_prefix_caches: VecDeque::new(),
            mtp_live_conversation_prefix: None,
            scalar_system_prefix_caches: VecDeque::new(),
            last_reused_tokens: 0,
            decode_logits_cache: RefCell::new(None),
        })
    }

    pub fn kv_capacity(&self) -> usize {
        self.runtime.target().capacity()
    }

    fn configure_context_reuse(&mut self, enabled: bool) -> Result<()> {
        self.last_reused_tokens = 0;
        self.context_reuse_enabled = enabled;
        if !enabled {
            self.system_prefix_cache = None;
            self.mtp_prompt_prefix_caches.clear();
            self.mtp_live_conversation_prefix = None;
            self.scalar_system_prefix_caches.clear();
        }
        Ok(())
    }

    /// Sampling requests use the scalar target runtime rather than the MTP
    /// coordinator. Keep a target-only checkpoint for that path so a non-zero
    /// temperature from AIOS does not silently disable system-prefix reuse.
    fn prefill_scalar_with_system_reuse(
        &mut self,
        prompt: &[u32],
        reusable_prefix_tokens: usize,
    ) -> Result<Tensor> {
        self.last_reused_tokens = 0;
        if reusable_prefix_tokens > prompt.len() {
            return Err(Error::Other(format!(
                "reusable system prefix has {reusable_prefix_tokens} tokens, but prompt has {}",
                prompt.len()
            )));
        }
        if !self.context_reuse_enabled
            || reusable_prefix_tokens == 0
            || reusable_prefix_tokens == prompt.len()
        {
            return Ok(self
                .runtime
                .target_mut()
                .prefill(&self.ctx, &self.decoder, &self.weights, prompt)?
                .logits);
        }

        let prefix = &prompt[..reusable_prefix_tokens];
        let hit_index = self
            .scalar_system_prefix_caches
            .iter()
            .position(|cache| cache.tokens.as_slice() == prefix);
        let mut cache = if let Some(index) = hit_index {
            self.scalar_system_prefix_caches
                .remove(index)
                .expect("located sampling prefix cache entry exists")
        } else {
            let reusable = if self.scalar_system_prefix_caches.len() >= SCALAR_PREFIX_CACHE_ENTRIES
            {
                self.scalar_system_prefix_caches.pop_back()
            } else {
                None
            };
            match reusable.filter(|cache| cache.capacity >= reusable_prefix_tokens) {
                Some(mut cache) => {
                    cache.tokens.clear();
                    cache
                }
                None => ScalarSystemPrefixCache {
                    tokens: Vec::new(),
                    snapshot: self.runtime.target_mut().allocate_snapshot(
                        &self.ctx,
                        &self.decoder,
                        &self.weights,
                        reusable_prefix_tokens,
                    )?,
                    capacity: reusable_prefix_tokens,
                },
            }
        };
        let result = (|| {
            if hit_index.is_some() {
                self.runtime
                    .target_mut()
                    .restore_snapshot(&self.ctx, &cache.snapshot)?;
                self.last_reused_tokens = reusable_prefix_tokens;
            } else {
                self.runtime.reset(&self.ctx)?;
                self.runtime.target_mut().prefill(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    &prompt[..reusable_prefix_tokens],
                )?;
                cache.tokens.clear();
                self.runtime
                    .target_mut()
                    .capture_snapshot(&self.ctx, &mut cache.snapshot)?;
                cache.tokens.extend_from_slice(prefix);
                eprintln!(
                    "scalar system prefix cache miss: prepared {reusable_prefix_tokens} prompt tokens"
                );
            }

            let output = self.runtime.target_mut().append_prefill(
                &self.ctx,
                &self.decoder,
                &self.weights,
                &prompt[reusable_prefix_tokens..],
            )?;
            if self.last_reused_tokens > 0 {
                eprintln!(
                    "scalar system prefix cache hit: reused {} of {} prompt tokens; evaluated {}",
                    self.last_reused_tokens,
                    prompt.len(),
                    prompt.len() - self.last_reused_tokens
                );
            }
            Ok(output.logits)
        })();
        self.scalar_system_prefix_caches.push_front(cache);
        result
    }

    /// Prepare an MTP request from the longest exact checkpoint available.
    /// Tool-result follow-ups can extend the previous conversation endpoint;
    /// unrelated requests fall back to the caller-designated system/tool prefix.
    fn prefill_mtp_with_system_reuse(
        &mut self,
        prompt: &[u32],
        reusable_prefix_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<super::mtp_runtime::MtpPrefillOutcome> {
        self.last_reused_tokens = 0;
        if reusable_prefix_tokens > prompt.len() {
            return Err(Error::Other(format!(
                "reusable system prefix has {reusable_prefix_tokens} tokens, but prompt has {}",
                prompt.len()
            )));
        }

        let exact_hit = longest_reusable_prefix_index(
            self.mtp_prompt_prefix_caches
                .iter()
                .map(|cache| cache.tokens.as_slice()),
            prompt,
        );
        let exact_hit_tokens = exact_hit
            .and_then(|index| self.mtp_prompt_prefix_caches.get(index))
            .map_or(0, |cache| cache.tokens.len());

        // A tool-result turn contains the prior rendered prompt and generated
        // assistant tool call verbatim. Prefer the still-live endpoint when it
        // is at least as deep as a stored checkpoint; otherwise restore the
        // deeper exact checkpoint. The suffix is always non-empty because
        // prompt checkpoints intentionally stop one token before the end.
        if self.context_reuse_enabled {
            let live_hit_tokens = self
                .mtp_live_conversation_prefix
                .as_ref()
                .filter(|tokens| {
                    live_conversation_suffix(tokens, prompt, self.runtime.target().position())
                        .is_some()
                })
                .map_or(0, Vec::len);

            if live_hit_tokens >= exact_hit_tokens && live_hit_tokens > 0 {
                let tokens = self
                    .mtp_live_conversation_prefix
                    .take()
                    .expect("a measured live-prefix hit remains present");
                let cached_tokens = tokens.len();
                let outcome =
                    self.append_mtp_with_prompt_checkpoint(prompt, cached_tokens, options)?;
                self.last_reused_tokens = cached_tokens;
                eprintln!(
                    "live conversation prefix hit: reused {} of {} prompt tokens; evaluated {}",
                    cached_tokens,
                    prompt.len(),
                    prompt.len() - cached_tokens
                );
                return Ok(outcome);
            }

            if let Some(index) = exact_hit {
                self.mtp_live_conversation_prefix = None;
                let cache = self
                    .mtp_prompt_prefix_caches
                    .remove(index)
                    .expect("located MTP prompt prefix cache entry exists");
                let cached_tokens = cache.tokens.len();
                self.runtime.restore_context(&self.ctx, &cache.snapshot)?;
                let outcome = self.runtime.append_prefill_greedy(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    &prompt[cached_tokens..],
                    options,
                )?;
                self.last_reused_tokens = cached_tokens;
                self.mtp_prompt_prefix_caches.push_front(cache);
                eprintln!(
                    "exact prompt prefix cache hit: reused {} of {} prompt tokens; evaluated {}",
                    cached_tokens,
                    prompt.len(),
                    prompt.len() - cached_tokens
                );
                return Ok(outcome);
            }

            if let Some(tokens) = self.mtp_live_conversation_prefix.take() {
                let cached_tokens = tokens.len();
                let runtime_position = self.runtime.target().position();
                let common_tokens = tokens
                    .iter()
                    .zip(prompt)
                    .take_while(|(cached, requested)| cached == requested)
                    .count();
                eprintln!(
                    "live conversation prefix miss: cached_tokens={cached_tokens} prompt_tokens={} runtime_position={runtime_position} common_tokens={common_tokens}",
                    prompt.len()
                );
            }
        }

        if !self.context_reuse_enabled
            || reusable_prefix_tokens == 0
            || reusable_prefix_tokens == prompt.len()
        {
            self.runtime.reset(&self.ctx)?;
            return self.runtime.prefill_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                prompt,
                options,
            );
        }

        let mut cache = match self
            .system_prefix_cache
            .take()
            .filter(|cache| cache.capacity >= reusable_prefix_tokens)
        {
            Some(cache) => cache,
            None => SystemPrefixCache {
                tokens: Vec::new(),
                snapshot: self.runtime.allocate_context_snapshot(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    reusable_prefix_tokens,
                )?,
                capacity: reusable_prefix_tokens,
            },
        };
        let result = (|| {
            let hit = cache.tokens.len() == reusable_prefix_tokens
                && prompt.get(..reusable_prefix_tokens) == Some(cache.tokens.as_slice());
            if hit {
                self.runtime.restore_context(&self.ctx, &cache.snapshot)?;
                self.last_reused_tokens = reusable_prefix_tokens;
            } else {
                self.runtime.reset(&self.ctx)?;
                self.runtime.prefill_greedy(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    &prompt[..reusable_prefix_tokens],
                    options,
                )?;
                // Do not publish a token identity until every device copy is
                // complete. A failed capture can never become a false hit.
                cache.tokens.clear();
                self.runtime
                    .capture_context(&self.ctx, &mut cache.snapshot)?;
                cache
                    .tokens
                    .extend_from_slice(&prompt[..reusable_prefix_tokens]);
                eprintln!(
                    "system prefix cache miss: prepared {reusable_prefix_tokens} prompt tokens"
                );
            }

            let outcome =
                self.append_mtp_with_prompt_checkpoint(prompt, reusable_prefix_tokens, options)?;
            if self.last_reused_tokens > 0 {
                eprintln!(
                    "system prefix cache hit: reused {} of {} prompt tokens; evaluated {}",
                    self.last_reused_tokens,
                    prompt.len(),
                    prompt.len() - self.last_reused_tokens
                );
            }
            Ok(outcome)
        })();
        self.system_prefix_cache = Some(cache);
        result
    }

    /// Extend the currently restored context while retaining a prompt
    /// checkpoint. First turns stop one token before the end. AIOS tool-result
    /// turns keep one scan-aligned uncached tail because their final
    /// elapsed-time context changes on every request. This trades a 64-token
    /// Cake append for a stable hit instead of alternating between one and
    /// ~430 evaluated rows.
    fn append_mtp_with_prompt_checkpoint(
        &mut self,
        prompt: &[u32],
        restored_tokens: usize,
        options: &GenerationOptions,
    ) -> Result<super::mtp_runtime::MtpPrefillOutcome> {
        if restored_tokens >= prompt.len() {
            return Err(Error::Other(
                "Qwen3.5 prompt checkpoint append requires a non-empty suffix".into(),
            ));
        }
        let checkpoint_tokens = if prompt.len() >= MTP_TOOL_PROMPT_MIN_TOKENS {
            prompt.len().saturating_sub(MTP_VOLATILE_TOOL_TAIL_TOKENS)
        } else {
            prompt.len() - 1
        };
        if checkpoint_tokens < restored_tokens {
            return Err(Error::Other(
                "Qwen3.5 restored prefix extends past the prompt checkpoint".into(),
            ));
        }

        if checkpoint_tokens > restored_tokens {
            self.runtime.append_prefill_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                &prompt[restored_tokens..checkpoint_tokens],
                options,
            )?;
        }

        if checkpoint_tokens > 0 {
            let mut cache = self.take_mtp_prompt_prefix_cache(checkpoint_tokens)?;
            self.runtime
                .capture_context(&self.ctx, &mut cache.snapshot)?;
            cache.tokens.clear();
            cache.tokens.extend_from_slice(&prompt[..checkpoint_tokens]);
            self.mtp_prompt_prefix_caches.push_front(cache);
        }

        self.runtime.append_prefill_greedy(
            &self.ctx,
            &self.decoder,
            &self.weights,
            &prompt[checkpoint_tokens..],
            options,
        )
    }

    fn take_mtp_prompt_prefix_cache(&mut self, capacity: usize) -> Result<SystemPrefixCache> {
        // Allocate every prompt checkpoint at the session capacity. Exact-size
        // snapshots grow with each AIOS turn, so an evicted smaller snapshot
        // could never be reused and each request carved a new, larger block
        // out of the HugeTLB pool until generation failed with CUDA OOM.
        let capacity = capacity.max(self.runtime.target().capacity());
        let reusable = if self.mtp_prompt_prefix_caches.len() >= MTP_PROMPT_PREFIX_CACHE_ENTRIES {
            self.mtp_prompt_prefix_caches.pop_back()
        } else {
            None
        };
        match reusable.filter(|cache| cache.capacity >= capacity) {
            Some(mut cache) => {
                cache.tokens.clear();
                Ok(cache)
            }
            None => Ok(SystemPrefixCache {
                tokens: Vec::new(),
                snapshot: self.runtime.allocate_context_snapshot(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    capacity,
                )?,
                capacity,
            }),
        }
    }

    /// Publish the committed token identity for the live MTP endpoint. The
    /// final emitted token is usually still the schedule's pending lookahead,
    /// so identity is truncated to the target's authoritative position.
    fn publish_mtp_live_conversation_prefix(
        &mut self,
        prompt: &[u32],
        generated: &[GeneratedToken],
    ) -> Result<()> {
        if !self.context_reuse_enabled {
            return Ok(());
        }
        let position = self.runtime.target().position();
        let Some(tokens) = committed_context_identity(prompt, generated, position)? else {
            self.mtp_live_conversation_prefix = None;
            return Ok(());
        };
        eprintln!(
            "live conversation prefix published {} committed tokens",
            tokens.len()
        );
        self.mtp_live_conversation_prefix = Some(tokens);
        Ok(())
    }

    /// Visible target KV length for checking benchmark timing boundaries.
    #[doc(hidden)]
    pub fn benchmark_context_position(&self) -> usize {
        self.runtime.target().position()
    }

    /// Benchmark-only setup for a scalar chunk-prefill case. The prefix is
    /// committed outside the measured region; an empty prefix leaves the
    /// target session reset for a normal prefill.
    #[doc(hidden)]
    pub fn benchmark_prepare_scalar_prefix(&mut self, prefix: &[u32]) -> Result<()> {
        self.runtime.reset(&self.ctx)?;
        if !prefix.is_empty() {
            self.runtime
                .target_mut()
                .prefill(&self.ctx, &self.decoder, &self.weights, prefix)?;
        }
        Ok(())
    }

    /// Run the measured scalar prefill chunk and return its greedy first token.
    #[doc(hidden)]
    pub fn benchmark_scalar_prefill_chunk(&mut self, chunk: &[u32]) -> Result<u32> {
        let output = if self.runtime.target().position() == 0 {
            self.runtime
                .target_mut()
                .prefill(&self.ctx, &self.decoder, &self.weights, chunk)?
        } else {
            self.runtime.target_mut().append_prefill(
                &self.ctx,
                &self.decoder,
                &self.weights,
                chunk,
            )?
        };
        output
            .greedy_token_ids(&self.ctx)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Other("Qwen3.5 scalar prefill returned no token".into()))
    }

    /// Continue exact scalar greedy decode from a measured prefill result.
    #[doc(hidden)]
    pub fn benchmark_scalar_decode(
        &mut self,
        first_token: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        if max_new_tokens == 0 {
            return Err(Error::Other(
                "Qwen3.5 scalar benchmark requires output tokens".into(),
            ));
        }
        // Match ordinary non-speculative generation: every scalar step is
        // committed, so no rollback shadow is required.
        self.runtime.target_mut().set_decode_in_place(true);
        // Reuse the compact device result. Allocating and freeing a CUDA
        // tensor for every argmax makes the benchmark measure the allocator
        // (seconds per token under the inference HugeTLB split), not decode.
        let next_token = CudaBuffer::alloc_zeros_async(
            std::mem::size_of::<i32>(),
            self.ctx.device_id(),
            self.ctx.stream(),
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1]), DType::I32)
        .map_err(Error::Cuda)?;
        let mut tokens = Vec::with_capacity(max_new_tokens);
        tokens.push(first_token);
        while tokens.len() < max_new_tokens {
            let input = *tokens.last().expect("one scalar token is present");
            let output =
                self.runtime
                    .target_mut()
                    .decode(&self.ctx, &self.decoder, &self.weights, input)?;
            output.enqueue_greedy_token_ids(&self.ctx, &next_token, 0)?;
            self.ctx.synchronize().map_err(Error::Cuda)?;
            let next = transfers::to_cpu(&next_token)?
                .as_i32()?
                .first()
                .copied()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| Error::Other("Qwen3.5 scalar decode returned no token".into()))?;
            tokens.push(next);
        }
        Ok(tokens)
    }

    /// Benchmark-only setup for an MTP chunk-prefill case. A non-empty prefix
    /// prepares both target and proposer state outside the measured region.
    #[doc(hidden)]
    pub fn benchmark_prepare_mtp_prefix(
        &mut self,
        prefix: &[u32],
        options: &GenerationOptions,
    ) -> Result<()> {
        self.runtime.reset(&self.ctx)?;
        if !prefix.is_empty() {
            self.runtime.prefill_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                prefix,
                options,
            )?;
        }
        Ok(())
    }

    /// Run a measured MTP-aware prefill chunk. This extends both target KV/GDN
    /// state and shifted MTP teacher-forcing state when a prefix is present.
    #[doc(hidden)]
    pub fn benchmark_mtp_prefill_chunk(
        &mut self,
        chunk: &[u32],
        options: &GenerationOptions,
    ) -> Result<u32> {
        let outcome = if self.runtime.target().position() == 0 {
            self.runtime
                .prefill_greedy(&self.ctx, &self.decoder, &self.weights, chunk, options)?
        } else {
            self.runtime.append_prefill_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                chunk,
                options,
            )?
        };
        Ok(outcome.first_token)
    }

    /// Continue the already prepared speculative schedule without repeating
    /// prompt work. Returned token IDs include `first_token`.
    #[doc(hidden)]
    pub fn benchmark_mtp_decode(
        &mut self,
        first_token: u32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        if max_new_tokens == 0 {
            return Err(Error::Other(
                "Qwen3.5 MTP benchmark requires output tokens".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(max_new_tokens);
        tokens.push(first_token);
        while tokens.len() < max_new_tokens && !self.runtime.is_finished() {
            let round = self
                .runtime
                .run_greedy_round(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    QWEN35_MTP_DRAFT_STEPS,
                )?;
            for token in round.outcome.emitted {
                if tokens.len() == max_new_tokens {
                    break;
                }
                tokens.push(token);
            }
        }
        if tokens.len() != max_new_tokens {
            return Err(Error::Other(format!(
                "Qwen3.5 MTP benchmark produced {} tokens, expected {max_new_tokens}",
                tokens.len()
            )));
        }
        Ok(tokens)
    }

    fn mtp_capacity_admits(&self, prompt_tokens: usize, max_new_tokens: usize) -> bool {
        prompt_tokens
            .checked_add(max_new_tokens)
            // A final budget-truncated round still verifies four scalar target
            // rows before committing only its admitted prefix.
            .and_then(|tokens| tokens.checked_add(QWEN35_MTP_DRAFT_STEPS))
            .is_some_and(|tokens| tokens <= self.kv_capacity())
    }

    fn generate_mtp(
        &mut self,
        request: GenerationRequest<'_>,
        policy: MtpGreedyPolicy,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        let prompt = request.input.token_ids;
        if request.input.image.is_some() {
            return Err(Error::Other("Qwen3.5-35B-A3B runtime is text-only".into()));
        }
        if prompt.is_empty() {
            return Err(Error::Other("Qwen3.5 generation requires a prompt".into()));
        }
        if let Some(error) = self.reset_error.take() {
            return Err(Error::Other(format!(
                "Qwen3.5 session reset failed before inference: {error}"
            )));
        }

        // A verified round can be rejected, so the pre-round state must survive.
        self.runtime.target_mut().set_decode_in_place(false);

        let mut profile = GenerationProfile::new();
        let prefill = self.prefill_mtp_with_system_reuse(
            prompt,
            request.input.reusable_prefix_tokens,
            request.options,
        )?;
        let first = GeneratedToken {
            token_id: prefill.first_token,
            logprob: None,
        };
        let mut tokens = Vec::with_capacity(policy.max_new_tokens);
        tokens.push(first);
        profile.record_first_token();
        profile.run_unprofiled(|| on_token(first));

        let mut adaptive = AdaptiveMtpController::new(self.decoder.thor_dynamic_fp8_decode());
        let mut scalar_fallback = false;
        while tokens.len() < policy.max_new_tokens && !self.runtime.is_finished() {
            if policy
                .eos_token_ids
                .contains(&tokens.last().expect("generated token exists").token_id)
            {
                break;
            }
            let depth = if scalar_fallback {
                0
            } else {
                adaptive.next_depth()
            };
            if depth == 0 {
                if !scalar_fallback {
                    eprintln!(
                        "adaptive MTP: scalar fallback after {} rounds; scalar_tpot_ms={:.3}",
                        adaptive.rounds, adaptive.scalar_tpot_ms
                    );
                    scalar_fallback = true;
                }
                let token_id = self.runtime.run_scalar_fallback_step(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                )?;
                let token = GeneratedToken {
                    token_id,
                    logprob: None,
                };
                tokens.push(token);
                profile.run_unprofiled(|| on_token(token));
                continue;
            }
            if depth != adaptive.last_depth {
                eprintln!(
                    "adaptive MTP: depth {} -> {} after {} rounds",
                    adaptive.last_depth, depth, adaptive.rounds
                );
            }
            let started = Instant::now();
            let round = self.runtime.run_greedy_round(
                &self.ctx,
                &self.decoder,
                &self.weights,
                depth,
            )?;
            adaptive.observe(
                depth,
                round.outcome.accepted_proposals,
                started.elapsed().as_secs_f64() * 1e3,
            );
            for token_id in round.outcome.emitted {
                if tokens.len() == policy.max_new_tokens {
                    return Err(Error::Other(
                        "Qwen3.5 MTP schedule exceeded the output budget".into(),
                    ));
                }
                let token = GeneratedToken {
                    token_id,
                    logprob: None,
                };
                tokens.push(token);
                profile.run_unprofiled(|| on_token(token));
            }
        }
        if scalar_fallback {
            // The target advanced without maintaining proposer KV. Keep the
            // static prefix snapshots, but do not advertise the live endpoint
            // as an exact target+MTP continuation.
            self.mtp_live_conversation_prefix = None;
        } else if let Err(error) = self.publish_mtp_live_conversation_prefix(prompt, &tokens) {
            // Generation has already streamed successfully. A best-effort
            // optimization must not turn that response into an API error.
            self.mtp_live_conversation_prefix = None;
            eprintln!("live conversation prefix publication skipped: {error}");
        }
        profile.finalize_with_reuse(prompt.len(), tokens.len(), self.last_reused_tokens);
        Ok(GenerationOutput { tokens, profile })
    }

    fn generate_dispatch(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        // Validate the same ranges and contradictory settings as the shared
        // sampler before considering the model-local fast path.
        request.options.resolve()?;
        let policy = MtpGreedyPolicy::from_generation_options(request.options).ok();
        if let Some(policy) = policy.filter(|policy| {
            request.input.image.is_none()
                && self.mtp_capacity_admits(request.input.token_ids.len(), policy.max_new_tokens)
        }) {
            return self.generate_mtp(request, policy, on_token);
        }

        // This is an intentional semantic fallback, not an error path. The
        // shared sampler owns random selection, top-k/top-p and penalties;
        // MTP's exact argmax acceptance protocol cannot preserve them.
        generate_streaming_with_options(self, request, on_token)
    }

    fn allocate_decode_logits(&self, rows: usize) -> Result<Tensor> {
        // Every row is fully overwritten by copy_decode_logits_row before any
        // reader sees it, so the reused buffer does not need re-zeroing.
        if let Some((cached_rows, tensor)) = self.decode_logits_cache.borrow().as_ref() {
            if *cached_rows == rows {
                return Ok(tensor.clone());
            }
        }
        let (_, total_bytes) = decode_logits_layout(rows, self.vocab_size)?;
        let output =
            CudaBuffer::alloc_zeros_async(total_bytes, self.ctx.device_id(), self.ctx.stream())
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(vec![rows, self.vocab_size]), DType::BF16)
                .map_err(Error::Cuda)?;
        *self.decode_logits_cache.borrow_mut() = Some((rows, output.clone()));
        Ok(output)
    }

    fn copy_decode_logits_row(&self, output: &Tensor, row: usize, logits: &Tensor) -> Result<()> {
        let rows = output
            .shape()
            .dims()
            .first()
            .copied()
            .ok_or_else(|| Error::Other("Qwen3.5 logits output must be rank-2".into()))?;
        let (row_bytes, _) = decode_logits_layout(rows, self.vocab_size)?;
        if output.dtype() != DType::BF16
            || output.shape().dims() != [rows, self.vocab_size]
            || logits.dtype() != DType::BF16
            || logits.shape().dims() != [1, self.vocab_size]
            || row >= rows
        {
            return Err(Error::Other(
                "Qwen3.5 scalar decode returned malformed logits".into(),
            ));
        }
        let destination = CudaBuffer::from_tensor(output)
            .map_err(Error::Cuda)?
            .view(row * row_bytes, row_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![1, self.vocab_size]), DType::BF16)
            .map_err(Error::Cuda)?;
        transfers::copy_cuda_to_cuda(&self.ctx, logits, &destination)
    }
}

fn committed_context_identity(
    prompt: &[u32],
    generated: &[GeneratedToken],
    committed_position: usize,
) -> Result<Option<Vec<u32>>> {
    if committed_position < prompt.len() {
        return Err(Error::Other(format!(
            "Qwen3.5 committed position {committed_position} precedes prompt length {}",
            prompt.len()
        )));
    }
    let committed_generated = committed_position - prompt.len();
    if committed_generated > generated.len() {
        return Err(Error::Other(format!(
            "Qwen3.5 committed position includes {committed_generated} generated tokens, but only {} were emitted",
            generated.len()
        )));
    }
    if committed_generated == 0 {
        return Ok(None);
    }
    let mut tokens = Vec::with_capacity(committed_position);
    tokens.extend_from_slice(prompt);
    tokens.extend(
        generated
            .iter()
            .take(committed_generated)
            .map(|token| token.token_id),
    );
    debug_assert_eq!(tokens.len(), committed_position);
    Ok(Some(tokens))
}

fn live_conversation_suffix<'a>(
    cached: &[u32],
    prompt: &'a [u32],
    runtime_position: usize,
) -> Option<&'a [u32]> {
    (!cached.is_empty()
        && cached.len() < prompt.len()
        && runtime_position == cached.len()
        && prompt.get(..cached.len()) == Some(cached))
    .then(|| &prompt[cached.len()..])
}

fn longest_reusable_prefix_index<'a>(
    caches: impl Iterator<Item = &'a [u32]>,
    prompt: &[u32],
) -> Option<usize> {
    caches
        .enumerate()
        .filter(|(_, cached)| {
            !cached.is_empty()
                && cached.len() < prompt.len()
                && prompt.get(..cached.len()) == Some(*cached)
        })
        .max_by_key(|(_, cached)| cached.len())
        .map(|(index, _)| index)
}

fn decode_logits_layout(rows: usize, vocab_size: usize) -> Result<(usize, usize)> {
    if rows == 0 || vocab_size == 0 {
        return Err(Error::Other(
            "Qwen3.5 decode logits dimensions must be non-zero".into(),
        ));
    }
    let row_bytes = vocab_size
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("Qwen3.5 logits row size overflow".into()))?;
    let total_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Other("Qwen3.5 logits allocation overflow".into()))?;
    Ok((row_bytes, total_bytes))
}

impl LlmTrait for Qwen35Moe {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self> {
        Err(Error::Contract(
            "Qwen3.5 MoE requires its exact nested config; load it through AutoModel",
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if let Some(error) = self.reset_error.take() {
            return Err(Error::Other(format!(
                "Qwen3.5 session reset failed before inference: {error}"
            )));
        }
        if token_ids.is_empty() {
            return Err(Error::Other("Qwen3.5 forward requires token IDs".into()));
        }
        let start = usize::try_from(start_pos)
            .map_err(|_| Error::Other("Qwen3.5 start position does not fit usize".into()))?;
        if start != 0 && self.runtime.target().position() != start {
            return Err(Error::Other(format!(
                "Qwen3.5 forward start position {start} does not match committed position {}",
                self.runtime.target().position()
            )));
        }

        if start == 0 {
            return Ok(self
                .runtime
                .target_mut()
                .prefill_all_logits(&self.ctx, &self.decoder, &self.weights, token_ids)?
                .logits);
        }

        // Non-speculative decode: every step commits, so the per-step
        // transactional shadow copy only reproduces state the step already
        // owns. Worth 0.844 ms of a 19.42 ms step.
        self.runtime.target_mut().set_decode_in_place(true);

        // Decode uses one persistent scratch row. Copy each row to the stable
        // result immediately, on the same CUDA stream, before the next decode
        // overwrites that scratch.
        let logits = self.allocate_decode_logits(token_ids.len())?;
        for (row, &token_id) in token_ids.iter().enumerate() {
            let output = self.runtime.target_mut().decode(
                &self.ctx,
                &self.decoder,
                &self.weights,
                token_id,
            )?;
            self.copy_decode_logits_row(&logits, row, &output.logits)?;
        }
        self.ctx.synchronize().map_err(Error::Cuda)?;
        Ok(logits)
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        apxinf_cuda_new::sampling::create_token_sampler(&self.ctx, spec)
    }

    fn speculative_generation_stats(&self) -> Option<SpeculativeGenerationStats> {
        let stats = self.runtime.schedule()?.stats();
        let target_verified_rows = stats.target_verified_rows;
        let target_rolled_back_rows = target_verified_rows.checked_sub(stats.emitted_tokens)?;
        Some(SpeculativeGenerationStats {
            implementation: "qwen3.6_moe_adaptive_mtp",
            draft_steps: QWEN35_MTP_DRAFT_STEPS,
            rounds: stats.rounds,
            zero_accept_rounds: stats.zero_accept_rounds,
            partial_accept_rounds: stats.partial_accept_rounds,
            full_accept_rounds: stats.full_accept_rounds,
            proposed_tokens: stats.proposed_tokens,
            verified_accepted_tokens: stats.verified_accepted_proposals,
            committed_accepted_tokens: stats.committed_accepted_proposals,
            correction_tokens: stats.correction_tokens,
            bonus_tokens: stats.bonus_tokens,
            emitted_tokens: stats.emitted_tokens,
            target_verified_rows,
            target_committed_rows: stats.emitted_tokens,
            target_rolled_back_rows,
            proposed_by_depth: stats.proposed_by_depth.to_vec(),
            matched_by_depth: stats.matched_by_depth.to_vec(),
            accepted_prefix_by_depth: stats.accepted_prefix_by_depth.to_vec(),
        })
    }

    fn reused_input_tokens(&self) -> usize {
        self.last_reused_tokens
    }

    fn set_context_reuse(&mut self, enabled: bool) -> Result<()> {
        self.configure_context_reuse(enabled)
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Other("Qwen3.5-35B-A3B runtime is text-only".into()));
        }
        if input.token_ids.is_empty() {
            return Err(Error::Other("Qwen3.5 prefill requires token IDs".into()));
        }
        if let Some(error) = self.reset_error.take() {
            return Err(Error::Other(format!(
                "Qwen3.5 session reset failed before inference: {error}"
            )));
        }
        self.prefill_scalar_with_system_reuse(input.token_ids, input.reusable_prefix_tokens)
    }

    fn reset(&mut self) {
        self.mtp_live_conversation_prefix = None;
        self.reset_error = self
            .runtime
            .reset(&self.ctx)
            .err()
            .map(|error| error.to_string());
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn generate_streaming_with_options(
        &mut self,
        request: GenerationRequest<'_>,
        mut on_token: impl FnMut(GeneratedToken),
    ) -> Result<GenerationOutput>
    where
        Self: Sized,
    {
        self.generate_dispatch(request, &mut on_token)
    }

    fn generate_streaming_with_options_dyn(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        self.generate_dispatch(request, on_token)
    }

    fn generate_streaming_dyn(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        let options = GenerationOptions::greedy(max_new_tokens, eos_token_id);
        let mut callback = |token: GeneratedToken| on_token(token.token_id);
        let output = self.generate_dispatch(
            GenerationRequest {
                input,
                options: &options,
            },
            &mut callback,
        )?;
        Ok((output.token_ids(), output.profile))
    }

    fn generate_streaming(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        mut on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)>
    where
        Self: Sized,
    {
        self.generate_streaming_dyn(input, max_new_tokens, &mut on_token, eos_token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multirow_decode_logits_use_distinct_contiguous_rows() {
        let (row_bytes, total_bytes) = decode_logits_layout(4, 248_320).unwrap();
        assert_eq!(row_bytes, 248_320 * DType::BF16.size_in_bytes());
        assert_eq!(total_bytes, 4 * row_bytes);
        let offsets = (0..4).map(|row| row * row_bytes).collect::<Vec<_>>();
        assert_eq!(offsets, vec![0, row_bytes, 2 * row_bytes, 3 * row_bytes]);
    }

    #[test]
    fn decode_logits_layout_rejects_empty_and_overflow() {
        assert!(decode_logits_layout(0, 248_320).is_err());
        assert!(decode_logits_layout(1, 0).is_err());
        assert!(decode_logits_layout(usize::MAX, 2).is_err());
    }

    #[test]
    fn committed_context_identity_excludes_pending_lookahead() {
        let prompt = [10, 11, 12];
        let generated = [
            GeneratedToken {
                token_id: 20,
                logprob: None,
            },
            GeneratedToken {
                token_id: 21,
                logprob: None,
            },
            GeneratedToken {
                token_id: 22,
                logprob: None,
            },
        ];
        let identity = committed_context_identity(&prompt, &generated, 5)
            .unwrap()
            .unwrap();
        assert_eq!(identity, vec![10, 11, 12, 20, 21]);
    }

    #[test]
    fn committed_context_identity_validates_runtime_position() {
        let generated = [GeneratedToken {
            token_id: 20,
            logprob: None,
        }];
        assert!(committed_context_identity(&[10, 11], &generated, 1).is_err());
        assert!(committed_context_identity(&[10, 11], &generated, 4).is_err());
        assert!(committed_context_identity(&[10, 11], &generated, 2)
            .unwrap()
            .is_none());
    }

    #[test]
    fn live_conversation_suffix_requires_exact_resident_prefix() {
        let prompt = [10, 11, 12, 20, 21];
        assert_eq!(
            live_conversation_suffix(&[10, 11, 12], &prompt, 3),
            Some(&[20, 21][..])
        );
        assert!(live_conversation_suffix(&[], &prompt, 0).is_none());
        assert!(live_conversation_suffix(&prompt, &prompt, prompt.len()).is_none());
        assert!(live_conversation_suffix(&[10, 99, 12], &prompt, 3).is_none());
        assert!(live_conversation_suffix(&[10, 11, 12], &prompt, 2).is_none());
    }

    #[test]
    fn exact_prompt_cache_selects_the_longest_compatible_prefix() {
        let caches = [vec![10, 11], vec![10, 11, 12, 13], vec![99], vec![]];
        let prompt = [10, 11, 12, 13, 14];
        assert_eq!(
            longest_reusable_prefix_index(caches.iter().map(Vec::as_slice), &prompt),
            Some(1)
        );
        assert_eq!(
            longest_reusable_prefix_index(caches.iter().map(Vec::as_slice), &[7, 8]),
            None
        );
        assert_eq!(
            longest_reusable_prefix_index(caches.iter().map(Vec::as_slice), &[10, 11]),
            None
        );
    }

    fn adaptive_controller() -> AdaptiveMtpController {
        AdaptiveMtpController {
            observations: VecDeque::with_capacity(ADAPTIVE_MTP_HISTORY),
            rounds: 0,
            last_depth: QWEN35_MTP_DRAFT_STEPS,
            scalar_tpot_ms: 19.42,
            fixed_depth: None,
        }
    }

    #[test]
    fn thor_dynamic_fp8_keeps_depth_three_at_low_acceptance() {
        let mut controller = AdaptiveMtpController::new(true);
        for _ in 0..16 {
            controller.observe(3, 0, 45.0);
            assert_eq!(controller.next_depth(), QWEN35_MTP_DRAFT_STEPS);
        }
    }

    #[test]
    fn adaptive_mtp_keeps_depth_three_for_high_acceptance() {
        let mut controller = adaptive_controller();
        for _ in 0..6 {
            controller.observe(3, 3, 40.0);
        }
        assert_eq!(controller.next_depth(), 3);
    }

    #[test]
    fn adaptive_mtp_shrinks_to_one_when_only_first_draft_matches() {
        let mut controller = adaptive_controller();
        for _ in 0..6 {
            controller.observe(3, 1, 35.0);
        }
        assert_eq!(controller.next_depth(), 1);
    }

    #[test]
    fn adaptive_mtp_falls_back_when_speculation_loses_to_scalar() {
        let mut controller = adaptive_controller();
        for _ in 0..6 {
            controller.observe(3, 0, 45.0);
        }
        assert_eq!(controller.next_depth(), 0);
    }
}
