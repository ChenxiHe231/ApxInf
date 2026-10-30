//! `LlmTrait` and `AutoModel` surface for the exact Qwen3.5-35B-A3B runtime.
//!
//! This wrapper owns one transactional sequence session. The serving worker is
//! batch one, so a request resets and reuses that storage while model weights
//! and cache allocations remain resident on CUDA.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    Qwen35DeviceWeights,
};
use super::execution::Qwen35ExecutionOptions;
use super::loader::load_qwen35_runtime_weights;
use super::mtp_runtime::{
    MtpGreedyPolicy, Qwen35ContextSnapshot, Qwen35MtpCoordinator, Qwen35MtpRuntime, Qwen35MtpState,
};
use super::mtp_schedule::QWEN35_MTP_DRAFT_STEPS;
use super::session::{Qwen35Session, Qwen35SessionSnapshot};

/// Default cache admission for direct `AutoModel` callers. OpenAI serving
/// supplies its prompt/output admission plus MTP lookahead through
/// `LoadOptions::text_kv_capacity` instead of relying on this default.
pub const DEFAULT_KV_CAPACITY: usize = 8_192;

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
    scalar_system_prefix_cache: Option<ScalarSystemPrefixCache>,
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
        Self::from_device_weights(config, ctx, weights, backend, kv_capacity, execution)
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
        let weights =
            load_qwen35_device_weights_from_path_with_options(&ctx, &config, path, execution)?;
        Self::from_device_weights(config, ctx, weights, backend, kv_capacity, execution)
    }

    fn from_device_weights(
        config: Qwen35MoeConfig,
        ctx: CudaContext,
        weights: Qwen35DeviceWeights,
        backend: Arc<dyn Backend>,
        kv_capacity: usize,
        execution: Qwen35ExecutionOptions,
    ) -> Result<Self> {
        let decoder = Qwen35Decoder::new(&config)?;
        let attention = Qwen35Fp8AttentionBackend::with_kv_cache(&config, execution.kv_cache)?;
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
            scalar_system_prefix_cache: None,
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
            self.scalar_system_prefix_cache = None;
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

        let mut cache = match self
            .scalar_system_prefix_cache
            .take()
            .filter(|cache| cache.capacity >= reusable_prefix_tokens)
        {
            Some(cache) => cache,
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
        };
        let result = (|| {
            let hit = cache.tokens.len() == reusable_prefix_tokens
                && prompt.get(..reusable_prefix_tokens) == Some(cache.tokens.as_slice());
            if hit {
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
                cache
                    .tokens
                    .extend_from_slice(&prompt[..reusable_prefix_tokens]);
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
        self.scalar_system_prefix_cache = Some(cache);
        result
    }

    /// Prepare an MTP request while retaining only the caller-designated
    /// system/tool prefix. Conversation history and user content are always
    /// evaluated again for each request.
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

            let outcome = self.runtime.append_prefill_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                &prompt[reusable_prefix_tokens..],
                options,
            )?;
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
        let mut tokens = Vec::with_capacity(max_new_tokens);
        tokens.push(first_token);
        while tokens.len() < max_new_tokens {
            let input = *tokens.last().expect("one scalar token is present");
            let output =
                self.runtime
                    .target_mut()
                    .decode(&self.ctx, &self.decoder, &self.weights, input)?;
            let next = output
                .greedy_token_ids(&self.ctx)?
                .into_iter()
                .next()
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
                .run_greedy_round(&self.ctx, &self.decoder, &self.weights)?;
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

        while tokens.len() < policy.max_new_tokens && !self.runtime.is_finished() {
            let round = self
                .runtime
                .run_greedy_round(&self.ctx, &self.decoder, &self.weights)?;
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
        let target_verified_rows = stats
            .rounds
            .checked_mul(super::mtp_schedule::QWEN35_MTP_VERIFY_ROWS)?;
        let target_rolled_back_rows = target_verified_rows.checked_sub(stats.emitted_tokens)?;
        Some(SpeculativeGenerationStats {
            implementation: "qwen3.5_moe_mtp3",
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
}
