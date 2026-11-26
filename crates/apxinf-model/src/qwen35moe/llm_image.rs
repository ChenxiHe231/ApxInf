//! Bounded visual-result and transactional multimodal prompt reuse.
use super::super::image_reuse::ImageKey;
use super::super::vision_input::{image_patch_offsets, ImagePositions};
use super::*;
use apxinf_core::Storage;

#[cfg(test)]
#[path = "../../examples/support/qwen35_cuda_profile.rs"]
mod feature_profile;

const FEATURE_CACHE_BYTES: usize = 128 * 1024 * 1024;
const FEATURE_CACHE_ENTRIES: usize = 64;

pub(super) struct CachedImageFeatures {
    key: ImageKey,
    features: Tensor,
}

fn rows(tensor: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let width = tensor.shape().dims()[1];
    let stride = width * tensor.dtype().size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(start * stride, len * stride)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![len, width]), tensor.dtype())
        .map_err(Error::Cuda)
}

fn allocate(ctx: &CudaContext, rows: usize, width: usize) -> Result<Tensor> {
    CudaBuffer::alloc(rows * width * 2, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![rows, width]), DType::BF16)
        .map_err(Error::Cuda)
}

/// Select whole processed images, preserving their bytes, dtype and patch order.
fn pack_image_pixels(
    ctx: &CudaContext,
    pixels: &Tensor,
    grids: &[[u32; 3]],
    selected: &[usize],
) -> Result<Tensor> {
    let offsets = image_patch_offsets(grids)?;
    let count: usize = selected
        .iter()
        .map(|&i| (offsets[i + 1] - offsets[i]) as usize)
        .sum();
    let row_bytes = 1536 * pixels.dtype().size_in_bytes();
    match pixels.storage() {
        Storage::Cpu(bytes) if pixels.device() == Device::Cpu => {
            let mut packed = Vec::with_capacity(count * row_bytes);
            for &i in selected {
                packed.extend_from_slice(
                    &bytes[offsets[i] as usize * row_bytes..offsets[i + 1] as usize * row_bytes],
                );
            }
            Tensor::from_raw(
                Shape::new(vec![count, 1536]),
                pixels.dtype(),
                Device::Cpu,
                packed,
            )
        }
        _ if pixels.device() == Device::Cuda(ctx.device_id()) => {
            let packed = CudaBuffer::alloc(count * row_bytes, ctx.device_id())
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(vec![count, 1536]), pixels.dtype())
                .map_err(Error::Cuda)?;
            let mut start = 0;
            for &i in selected {
                let len = (offsets[i + 1] - offsets[i]) as usize;
                transfers::copy_cuda_to_cuda(
                    ctx,
                    &rows(pixels, offsets[i] as usize, len)?,
                    &rows(&packed, start, len)?,
                )?;
                start += len;
            }
            ctx.synchronize().map_err(Error::Cuda)?;
            Ok(packed)
        }
        _ => Err(Error::UnsupportedDevice(pixels.device())),
    }
}

fn stable_checkpoint(tokens: &[u32]) -> Option<usize> {
    if tokens.len() >= MTP_TOOL_PROMPT_MIN_TOKENS {
        Some(tokens.len() - MTP_VOLATILE_TOOL_TAIL_TOKENS)
    } else if tokens.ends_with(&[248068, 271, 248069, 271]) {
        Some(tokens.len() - 4)
    } else {
        None
    }
}

impl Qwen35Moe {
    pub(super) fn prepare_image_embeddings(
        &mut self,
        input: LlmInput<'_>,
    ) -> Result<(Tensor, ImageIdentity)> {
        let image = input.image.expect("image prefill requires media");
        if input.token_ids.len() > self.kv_capacity()
            || input
                .token_ids
                .iter()
                .any(|&id| id as usize >= self.vocab_size)
            || input.reusable_prefix_tokens > input.token_ids.len()
        {
            return Err(Error::Other(
                "Qwen3.5 image prompt exceeds token or KV bounds".into(),
            ));
        }
        if !matches!(image.pixel_values.device(), Device::Cpu)
            && image.pixel_values.device() != Device::Cuda(self.ctx.device_id())
        {
            return Err(Error::UnsupportedDevice(image.pixel_values.device()));
        }
        super::super::vision::validate_input(image.pixel_values, image.grid_thw)?;
        let identity = if self.context_reuse_enabled {
            let host;
            let pixels = if image.pixel_values.device() == Device::Cpu {
                image.pixel_values
            } else {
                host = transfers::to_cpu(image.pixel_values)?;
                super::super::vision::validate_input(&host, image.grid_thw)?;
                &host
            };
            ImageIdentity::new(input.token_ids, pixels, image.grid_thw)?
        } else {
            ImageIdentity::uncached(input.token_ids, image.grid_thw)?
        };
        let visual = self.image_features(&identity, image.pixel_values, image.grid_thw)?;
        let embeddings = super::super::vision::embed_and_replace(
            &self.ctx,
            &self.weights.embedding,
            input.token_ids,
            &visual,
        )?;
        Ok((embeddings, identity))
    }

    fn image_features(
        &mut self,
        identity: &ImageIdentity,
        pixels: &Tensor,
        grids: &[[u32; 3]],
    ) -> Result<Tensor> {
        let hits: Vec<Option<Tensor>> = if self.context_reuse_enabled {
            identity
                .keys
                .iter()
                .map(|key| {
                    self.image_features
                        .iter()
                        .find(|e| e.key == *key)
                        .map(|e| e.features.clone())
                })
                .collect()
        } else {
            Vec::new()
        };
        if hits.iter().any(Option::is_some) {
            // Retain owners until all copies finish, even if admission evicts
            // a hit. Deduplicate missing media without changing request order.
            let mut missing = Vec::new();
            for (i, hit) in hits.iter().enumerate() {
                if hit.is_none()
                    && !missing
                        .iter()
                        .any(|&j| identity.keys[j] == identity.keys[i])
                {
                    missing.push(i);
                }
            }
            let missing_keys: Vec<_> = missing.iter().map(|&i| identity.keys[i].clone()).collect();
            let missing_visual = if missing.is_empty() {
                None
            } else {
                let packed = pack_image_pixels(&self.ctx, pixels, grids, &missing)?;
                let grids: Vec<_> = missing_keys.iter().map(|k| k.grid).collect();
                Some(self.run_image_encoder(&packed, &grids)?)
            };
            let offsets: Vec<_> = std::iter::once(0)
                .chain(missing_keys.iter().scan(0, |n, k| {
                    *n += (k.grid[1] * k.grid[2] / 4) as usize;
                    Some(*n)
                }))
                .collect();
            let count = identity
                .keys
                .iter()
                .map(|k| (k.grid[1] * k.grid[2] / 4) as usize)
                .sum();
            let visual = allocate(&self.ctx, count, 2048)?;
            let mut start = 0;
            for (i, hit) in hits.iter().enumerate() {
                let len = (identity.keys[i].grid[1] * identity.keys[i].grid[2] / 4) as usize;
                let source = if let Some(hit) = hit {
                    hit.clone()
                } else {
                    let index = missing_keys
                        .iter()
                        .position(|k| *k == identity.keys[i])
                        .unwrap();
                    rows(missing_visual.as_ref().unwrap(), offsets[index], len)?
                };
                transfers::copy_cuda_to_cuda(&self.ctx, &source, &rows(&visual, start, len)?)?;
                start += len;
            }
            self.ctx.synchronize().map_err(Error::Cuda)?;
            for (key, hit) in identity.keys.iter().zip(&hits) {
                if hit.is_some() {
                    let index = self
                        .image_features
                        .iter()
                        .position(|e| e.key == *key)
                        .unwrap();
                    let entry = self.image_features.remove(index).unwrap();
                    self.image_features.push_front(entry);
                }
            }
            if let Some(missing_visual) = missing_visual {
                self.admit_image_features(&missing_keys, &missing_visual)?;
                eprintln!(
                    "image feature cache partial hit: {} of {} images; encoded {} unique images",
                    hits.iter().filter(|h| h.is_some()).count(),
                    hits.len(),
                    missing.len()
                );
            } else {
                eprintln!("image feature cache hit: {} images", hits.len());
            }
            return Ok(visual);
        }
        let visual = self.run_image_encoder(pixels, grids)?;
        if self.context_reuse_enabled {
            self.admit_image_features(&identity.keys, &visual)?;
        }
        Ok(visual)
    }

    fn run_image_encoder(&mut self, pixels: &Tensor, grids: &[[u32; 3]]) -> Result<Tensor> {
        // Use the original packed encoder for this ordered image subset.
        // Admission makes independent copies of reusable scratch outputs.
        if self.vision.is_none() {
            let source = self.vision_source.as_ref().ok_or_else(|| {
                Error::Other("checkpoint does not include a supported visual tower".into())
            })?;
            self.vision = Some(source.load(&self.ctx)?);
            self.vision_source = None;
        }
        let pixels = super::super::vision::upload_pixels(&self.ctx, pixels)?;
        if let Some(index) = self
            .vision_sessions
            .iter()
            .position(|(s, _, _)| s.matches_grids(grids))
        {
            let entry = self.vision_sessions.remove(index).unwrap();
            self.vision_sessions.push_front(entry);
        } else {
            if self.vision_sessions.len() == 4 {
                self.vision_sessions.pop_back();
            }
            self.vision_sessions.push_front((
                super::super::vision::VisionScratch::new(&self.ctx, grids)?,
                ops::ExecutionSession::with_capacity(64 * 1024 * 1024, self.ctx.device_id())?,
                false,
            ));
        }
        let (scratch, session, prepared) = self.vision_sessions.front_mut().unwrap();
        let visual = super::super::vision::forward(
            self.vision.as_ref().unwrap(),
            &self.ctx,
            scratch,
            session,
            *prepared,
            &pixels,
            grids,
        )?;
        *prepared = true;
        Ok(visual)
    }

    fn admit_image_features(&mut self, keys: &[ImageKey], visual: &Tensor) -> Result<()> {
        let mut start = 0;
        for key in keys {
            let len = (key.grid[1] * key.grid[2] / 4) as usize;
            let bytes = len * 2048 * 2;
            if bytes <= FEATURE_CACHE_BYTES && !self.image_features.iter().any(|e| e.key == *key) {
                while self.image_features.len() >= FEATURE_CACHE_ENTRIES
                    || self
                        .image_features
                        .iter()
                        .map(|e| e.features.size_in_bytes())
                        .sum::<usize>()
                        + bytes
                        > FEATURE_CACHE_BYTES
                {
                    self.image_features.pop_back();
                }
                let features = allocate(&self.ctx, len, 2048)?;
                transfers::copy_cuda_to_cuda(&self.ctx, &rows(visual, start, len)?, &features)?;
                self.ctx.synchronize().map_err(Error::Cuda)?;
                self.image_features.push_front(CachedImageFeatures {
                    key: key.clone(),
                    features,
                });
            }
            start += len;
        }
        Ok(())
    }

    fn take_scalar_image_checkpoint(&mut self) -> Result<ScalarSystemPrefixCache> {
        let capacity = self.kv_capacity();
        let reusable = if self.scalar_system_prefix_caches.len() >= SCALAR_PREFIX_CACHE_ENTRIES {
            // Keep the leading text checkpoint available when an image is
            // replaced/resized. Reserve two slots for stable+complete image
            // endpoints, then recycle image entries before evicting systems.
            let image_entries = self
                .scalar_system_prefix_caches
                .iter()
                .filter(|c| c.image.is_some())
                .count();
            let index = self
                .scalar_system_prefix_caches
                .iter()
                .rposition(|c| {
                    if image_entries < 2 {
                        c.image.is_none()
                    } else {
                        c.image.is_some()
                    }
                })
                .unwrap_or(self.scalar_system_prefix_caches.len() - 1);
            self.scalar_system_prefix_caches.remove(index)
        } else {
            None
        };
        if let Some(mut cache) = reusable.filter(|c| c.capacity >= capacity) {
            cache.tokens.clear();
            cache.image = None;
            cache.image_logits = None;
            return Ok(cache);
        }
        Ok(ScalarSystemPrefixCache {
            tokens: Vec::new(),
            image: None,
            image_logits: None,
            capacity,
            snapshot: self.runtime.target_mut().allocate_snapshot(
                &self.ctx,
                &self.decoder,
                &self.weights,
                capacity,
            )?,
        })
    }

    pub(super) fn prefill_image(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        let (embeddings, identity) = self.prepare_image_embeddings(input)?;
        self.mtp_live_conversation_prefix = None;
        self.mtp_live_image = None;
        self.last_reused_tokens = 0;
        let previous = self
            .runtime
            .target_mut()
            .attention_mut()
            .replace_image_positions(Some(identity.positions.clone()));
        let result = self.prefill_scalar_image_reuse(input, &embeddings, &identity);
        if result.is_err() {
            self.runtime
                .target_mut()
                .attention_mut()
                .set_image_positions(previous);
        }
        result
    }

    fn scalar_image_chunk(
        &mut self,
        tokens: &[u32],
        embeddings: &Tensor,
        start: usize,
        end: usize,
    ) -> Result<Tensor> {
        Ok(self
            .runtime
            .target_mut()
            .prefill_image(
                &self.ctx,
                &self.decoder,
                &self.weights,
                &tokens[start..end],
                &rows(embeddings, start, end - start)?,
                start,
            )?
            .logits)
    }

    fn prefill_scalar_image_reuse(
        &mut self,
        input: LlmInput<'_>,
        embeddings: &Tensor,
        identity: &ImageIdentity,
    ) -> Result<Tensor> {
        let prompt = input.token_ids;
        if !self.context_reuse_enabled {
            return self.scalar_image_chunk(prompt, embeddings, 0, prompt.len());
        }
        let hit = self
            .scalar_system_prefix_caches
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                (c.tokens.len() < prompt.len()
                    || (c.tokens.len() == prompt.len() && c.image_logits.is_some()))
                    && !c.tokens.is_empty()
                    && prompt.starts_with(&c.tokens)
                    && identity.matches_prefix(c.image.as_ref(), c.tokens.len())
            })
            .max_by_key(|(_, c)| c.tokens.len())
            .map(|(i, _)| i);
        let mut start = 0;
        if let Some(index) = hit {
            let cache = self.scalar_system_prefix_caches.remove(index).unwrap();
            self.runtime
                .target_mut()
                .restore_snapshot(&self.ctx, &cache.snapshot)?;
            start = cache.tokens.len();
            self.last_reused_tokens = start;
            let logits = if start == prompt.len() {
                cache.image_logits.clone()
            } else {
                None
            };
            self.scalar_system_prefix_caches.push_front(cache);
            if let Some(logits) = logits {
                eprintln!("image scalar prefix cache hit: reused {start} of {} prompt tokens; evaluated 0", prompt.len());
                return Ok(logits);
            }
        } else {
            let prefix = input
                .reusable_prefix_tokens
                .min(identity.text_prefix_limit())
                .min(prompt.len() - 1);
            if prefix > 0 {
                self.runtime.reset(&self.ctx)?;
                self.runtime.target_mut().prefill(
                    &self.ctx,
                    &self.decoder,
                    &self.weights,
                    &prompt[..prefix],
                )?;
                let mut cache = self.take_scalar_image_checkpoint()?;
                self.runtime
                    .target_mut()
                    .capture_snapshot(&self.ctx, &mut cache.snapshot)?;
                cache.tokens.extend_from_slice(&prompt[..prefix]);
                self.scalar_system_prefix_caches.push_front(cache);
                start = prefix;
            }
        }
        if let Some(checkpoint) = stable_checkpoint(prompt)
            .map(|n| n.max(start))
            .filter(|&n| n > 0 && n < prompt.len())
        {
            if checkpoint > start {
                self.scalar_image_chunk(prompt, embeddings, start, checkpoint)?;
            }
            if !self.scalar_system_prefix_caches.iter().any(|c| {
                c.tokens == prompt[..checkpoint]
                    && identity.matches_prefix(c.image.as_ref(), checkpoint)
            }) {
                let mut cache = self.take_scalar_image_checkpoint()?;
                self.runtime
                    .target_mut()
                    .capture_snapshot(&self.ctx, &mut cache.snapshot)?;
                cache.tokens.extend_from_slice(&prompt[..checkpoint]);
                cache.image = Some(identity.clone());
                self.scalar_system_prefix_caches.push_front(cache);
            }
            start = checkpoint;
        }
        let logits = self.scalar_image_chunk(prompt, embeddings, start, prompt.len())?;
        let mut cache = self.take_scalar_image_checkpoint()?;
        self.runtime
            .target_mut()
            .capture_snapshot(&self.ctx, &mut cache.snapshot)?;
        let frozen = CudaBuffer::alloc(logits.size_in_bytes(), self.ctx.device_id())
            .map_err(Error::Cuda)?
            .as_tensor(logits.shape().clone(), logits.dtype())
            .map_err(Error::Cuda)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &logits, &frozen)?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        cache.tokens.extend_from_slice(prompt);
        cache.image = Some(identity.clone());
        cache.image_logits = Some(frozen);
        self.scalar_system_prefix_caches.push_front(cache);
        eprintln!(
            "image scalar prefix reuse: reused {} of {} prompt tokens; evaluated {}",
            self.last_reused_tokens,
            prompt.len(),
            prompt.len() - self.last_reused_tokens
        );
        Ok(logits)
    }

    fn mtp_image_chunk(
        &mut self,
        prompt: &[u32],
        embeddings: &Tensor,
        positions: &std::rc::Rc<ImagePositions>,
        start: usize,
        end: usize,
        options: &GenerationOptions,
    ) -> Result<super::super::mtp_runtime::MtpPrefillOutcome> {
        let embeddings = rows(embeddings, start, end - start)?;
        if start == 0 {
            // Full image prefill initializes shadow target/proposer state
            // transactionally; resetting committed banks first is redundant.
            self.runtime.prefill_image_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                &prompt[..end],
                &embeddings,
                positions.clone(),
                options,
            )
        } else {
            self.runtime.append_image_prefill_greedy(
                &self.ctx,
                &self.decoder,
                &self.weights,
                &prompt[start..end],
                &embeddings,
                positions.clone(),
                options,
            )
        }
    }

    pub(super) fn prefill_mtp_image_reuse(
        &mut self,
        prompt: &[u32],
        embeddings: &Tensor,
        identity: &ImageIdentity,
        reusable: usize,
        options: &GenerationOptions,
    ) -> Result<u32> {
        self.last_reused_tokens = 0;
        if !self.context_reuse_enabled {
            self.mtp_live_conversation_prefix = None;
            self.mtp_live_image = None;
            return self
                .mtp_image_chunk(
                    prompt,
                    embeddings,
                    &identity.positions,
                    0,
                    prompt.len(),
                    options,
                )
                .map(|o| o.first_token);
        }
        let hit = self
            .mtp_prompt_prefix_caches
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                !c.tokens.is_empty()
                    && (c.tokens.len() < prompt.len()
                        || (c.tokens.len() == prompt.len() && c.image_first_token.is_some()))
                    && prompt.starts_with(&c.tokens)
                    && identity.matches_prefix(c.image.as_ref(), c.tokens.len())
            })
            .max_by_key(|(_, c)| c.tokens.len())
            .map(|(i, c)| (i, c.tokens.len()));
        let live = self
            .mtp_live_conversation_prefix
            .as_ref()
            .filter(|tokens| {
                live_conversation_suffix(tokens, prompt, self.runtime.target().position()).is_some()
                    && identity.matches_prefix(self.mtp_live_image.as_ref(), tokens.len())
            })
            .map_or(0, Vec::len);
        // A restore can fail after changing one of the state trees. Do not
        // leave the previous live identity advertised during that operation.
        self.mtp_live_conversation_prefix = None;
        self.mtp_live_image = None;
        let mut start = 0;
        if live > 0 && live >= hit.map_or(0, |(_, n)| n) {
            start = live;
            self.last_reused_tokens = start;
        } else if let Some((index, len)) = hit {
            let cache = self.mtp_prompt_prefix_caches.remove(index).unwrap();
            self.runtime.restore_context(&self.ctx, &cache.snapshot)?;
            let first = cache.image_first_token;
            self.mtp_prompt_prefix_caches.push_front(cache);
            if len == prompt.len() {
                let first = first
                    .ok_or_else(|| Error::Other("missing image checkpoint first token".into()))?;
                self.runtime.restart_prefill_greedy(first, options)?;
                self.last_reused_tokens = len;
                eprintln!(
                    "image MTP prefix cache hit: reused {len} of {} prompt tokens; evaluated 0",
                    prompt.len()
                );
                return Ok(first);
            }
            start = len;
            self.last_reused_tokens = start;
        } else {
            let prefix = reusable
                .min(identity.text_prefix_limit())
                .min(prompt.len() - 1);
            if prefix > 0 {
                let mut cache = self.system_prefix_cache.take();
                let hit = cache.as_ref().is_some_and(|c| c.tokens == prompt[..prefix]);
                if hit {
                    self.runtime
                        .restore_context(&self.ctx, &cache.as_ref().unwrap().snapshot)?;
                    self.last_reused_tokens = prefix;
                } else {
                    self.runtime.reset(&self.ctx)?;
                    self.runtime.prefill_greedy(
                        &self.ctx,
                        &self.decoder,
                        &self.weights,
                        &prompt[..prefix],
                        options,
                    )?;
                    if cache.as_ref().is_none_or(|c| c.capacity < prefix) {
                        cache = Some(SystemPrefixCache {
                            tokens: Vec::new(),
                            image: None,
                            image_first_token: None,
                            capacity: prefix,
                            snapshot: self.runtime.allocate_context_snapshot(
                                &self.ctx,
                                &self.decoder,
                                &self.weights,
                                prefix,
                            )?,
                        });
                    }
                    let c = cache.as_mut().unwrap();
                    c.tokens.clear();
                    self.runtime.capture_context(&self.ctx, &mut c.snapshot)?;
                    c.tokens.extend_from_slice(&prompt[..prefix]);
                }
                self.system_prefix_cache = cache;
                start = prefix;
            }
        }
        if let Some(checkpoint) = stable_checkpoint(prompt)
            .map(|n| n.max(start))
            .filter(|&n| n > 0 && n < prompt.len())
        {
            if checkpoint > start {
                self.mtp_image_chunk(
                    prompt,
                    embeddings,
                    &identity.positions,
                    start,
                    checkpoint,
                    options,
                )?;
            }
            if !self.mtp_prompt_prefix_caches.iter().any(|c| {
                c.tokens == prompt[..checkpoint]
                    && identity.matches_prefix(c.image.as_ref(), checkpoint)
            }) {
                let mut cache = self.take_mtp_prompt_prefix_cache(checkpoint)?;
                self.runtime
                    .capture_context(&self.ctx, &mut cache.snapshot)?;
                cache.tokens.extend_from_slice(&prompt[..checkpoint]);
                cache.image = Some(identity.clone());
                self.mtp_prompt_prefix_caches.push_front(cache);
            }
            start = checkpoint;
        }
        let output = self.mtp_image_chunk(
            prompt,
            embeddings,
            &identity.positions,
            start,
            prompt.len(),
            options,
        )?;
        let mut cache = self.take_mtp_prompt_prefix_cache(prompt.len())?;
        self.runtime
            .capture_context(&self.ctx, &mut cache.snapshot)?;
        cache.tokens.extend_from_slice(prompt);
        cache.image = Some(identity.clone());
        cache.image_first_token = Some(output.first_token);
        self.mtp_prompt_prefix_caches.push_front(cache);
        eprintln!(
            "image MTP prefix reuse: reused {} of {} prompt tokens; evaluated {}",
            self.last_reused_tokens,
            prompt.len(),
            prompt.len() - self.last_reused_tokens
        );
        Ok(output.first_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ImageInput;

    fn check_mixed_feature_cache(
        model: &mut Qwen35Moe,
        tokens: &[u32],
        pixels: &Tensor,
        grids: &[[u32; 3]],
    ) -> Result<()> {
        model.configure_context_reuse(false)?;
        let input = LlmInput::with_image(tokens, ImageInput::new(pixels, grids));
        let expected = model.prepare_image_embeddings(input)?.0;
        let expected = transfers::to_cpu(&expected)?.to_f32_vec()?;
        // Optional isolated diagnostics capture only the representative fixture,
        // after native preparation, never the small boundary cases below.
        let profile = if pixels.shape().dims()[0] > 28 {
            std::env::var("APXINF_QWEN35_FEATURE_PROFILE").ok()
        } else {
            None
        };
        if profile.as_deref() == Some("full") {
            let range = feature_profile::CudaProfileRange::start(true)
                .map_err(|e| Error::Other(e.to_string()))?;
            model.prepare_image_embeddings(input)?;
            model.ctx.synchronize().map_err(Error::Cuda)?;
            range.finish().map_err(|e| Error::Other(e.to_string()))?;
        }
        model.configure_context_reuse(true)?;
        let patches = (grids[0][1] * grids[0][2]) as usize;
        let first_pixels = pack_image_pixels(&model.ctx, pixels, grids, &[0])?;
        let mut first_tokens = vec![248053];
        first_tokens.extend(std::iter::repeat_n(248056, patches / 4));
        first_tokens.push(248054);
        model.prepare_image_embeddings(LlmInput::with_image(
            &first_tokens,
            ImageInput::new(&first_pixels, &grids[..1]),
        ))?;
        assert_eq!(model.image_features.len(), 1);
        let host = if pixels.device() == Device::Cpu {
            pixels.clone()
        } else {
            transfers::to_cpu(pixels)?
        };
        let identity = ImageIdentity::new(tokens, &host, grids)?;
        assert_eq!(model.image_features[0].key, identity.keys[0]);
        let range = if profile.as_deref() == Some("partial") {
            let missing: Vec<_> = (1..grids.len()).collect();
            let packed = pack_image_pixels(&model.ctx, pixels, grids, &missing)?;
            model.run_image_encoder(&packed, &grids[1..])?;
            model.ctx.synchronize().map_err(Error::Cuda)?;
            Some(
                feature_profile::CudaProfileRange::start(true)
                    .map_err(|e| Error::Other(e.to_string()))?,
            )
        } else {
            None
        };
        let partial = model.prepare_image_embeddings(input)?.0;
        model.ctx.synchronize().map_err(Error::Cuda)?;
        if let Some(range) = range {
            range.finish().map_err(|e| Error::Other(e.to_string()))?;
        }
        assert_eq!(
            transfers::to_cpu(&partial)?.to_f32_vec()?,
            expected,
            "partial-hit feature assembly differs from original packed execution"
        );
        let cached = model.prepare_image_embeddings(input)?.0;
        assert_eq!(
            transfers::to_cpu(&cached)?.to_f32_vec()?,
            expected,
            "per-image cached features differ from original packed execution"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA checkpoint, processed fixture and prepared recipes"]
    fn image_snapshot_append_matches_uncached_chunk_execution() -> Result<()> {
        let checkpoint = PathBuf::from(std::env::var("APXINF_QWEN35_CHECKPOINT").unwrap());
        let fixture = std::fs::read(std::env::var("APXINF_QWEN35_IMAGE_FIXTURE").unwrap()).unwrap();
        let end = 4 + u32::from_le_bytes(fixture[..4].try_into().unwrap()) as usize;
        let metadata: serde_json::Value = serde_json::from_slice(&fixture[4..end]).unwrap();
        let tokens: Vec<u32> = serde_json::from_value(metadata["tokens"].clone()).unwrap();
        let grids: Vec<[u32; 3]> = serde_json::from_value(metadata["grids"].clone()).unwrap();
        let shape: Vec<usize> = serde_json::from_value(metadata["shape"].clone()).unwrap();
        let values: Vec<f32> = fixture[end..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let pixels = Tensor::from_f32(shape, &values)?;
        let kv: Qwen35KvCacheDType = std::env::var("APXINF_QWEN35_IMAGE_KV")
            .unwrap()
            .parse()
            .map_err(Error::Other)?;
        let scales = if kv == Qwen35KvCacheDType::Fp8 {
            Some(super::super::super::config::load_trt_fp8_kv_cache_scales(
                &checkpoint.join("hf_quant_config.json"),
            )?)
        } else {
            None
        };
        let mut model = Qwen35Moe::from_checkpoint_path_with_runtime_options(
            Qwen35MoeConfig::from_json_file(&checkpoint.join("config.json"))?,
            &checkpoint,
            Device::Cuda(0),
            Arc::new(apxinf_cpu::CpuBackend),
            tokens.len() + 128,
            Qwen35ExecutionOptions {
                moe_activation: super::super::super::execution::Qwen35MoeActivation::Nvfp4,
                kv_cache: kv,
            },
            std::env::var_os("APXINF_QWEN35_IMAGE_PREPARE").is_some(),
            Some(PathBuf::from(
                std::env::var("APXINF_QWEN35_IMAGE_RECIPES").unwrap(),
            )),
            scales,
        )?;
        check_mixed_feature_cache(&mut model, &tokens, &pixels, &grids)?;
        // Odd merged-row boundaries exercise row-tile alignment independently
        // of the serving processor's usual minimum image dimensions.
        let miniature: Vec<f32> = values
            .iter()
            .take(8 * 1536)
            .enumerate()
            .map(|(i, &v)| if i >= 4 * 1536 { v + 0.25 } else { v })
            .collect();
        let miniature = Tensor::from_f32(vec![8, 1536], &miniature)?;
        check_mixed_feature_cache(
            &mut model,
            &[248053, 248056, 248054, 248053, 248056, 248054],
            &miniature,
            &[[1, 2, 2], [1, 2, 2]],
        )?;
        // Two different misses, duplicate missing media, and odd merged-row
        // boundaries exercise selection, offset assembly and deduplication.
        let mixed_grids = [[1, 2, 2], [1, 2, 4], [1, 2, 4], [1, 4, 2]];
        let mixed_values: Vec<_> = [0usize, 1, 1, 2]
            .iter()
            .zip(&mixed_grids)
            .flat_map(|(&id, g)| {
                values
                    .iter()
                    .take((g[1] * g[2]) as usize * 1536)
                    .map(move |v| v + id as f32 * 0.25)
            })
            .collect();
        let mixed = Tensor::from_f32(vec![28, 1536], &mixed_values)?;
        let mut mixed_tokens = Vec::new();
        for g in mixed_grids {
            mixed_tokens.push(248053);
            mixed_tokens.extend(std::iter::repeat_n(248056, (g[1] * g[2] / 4) as usize));
            mixed_tokens.push(248054);
        }
        check_mixed_feature_cache(&mut model, &mixed_tokens, &mixed, &mixed_grids)?;
        let gpu_f32 = transfers::to_cuda(&mixed, model.ctx.device_id())?;
        check_mixed_feature_cache(&mut model, &mixed_tokens, &gpu_f32, &mixed_grids)?;
        let gpu_bf16 = super::super::super::vision::upload_pixels(&model.ctx, &mixed)?;
        check_mixed_feature_cache(&mut model, &mixed_tokens, &gpu_bf16, &mixed_grids)?;
        let cpu_bf16 = transfers::to_cpu(&gpu_bf16)?;
        check_mixed_feature_cache(&mut model, &mixed_tokens, &cpu_bf16, &mixed_grids)?;
        model.configure_context_reuse(false)?;
        let suffix = [15, 16, 17];
        let input = LlmInput::with_image(&tokens, ImageInput::new(&pixels, &grids));
        let original = model.prepare_image_embeddings(input)?.0;
        let original = transfers::to_cpu(&original)?.to_f32_vec()?;
        model.configure_context_reuse(true)?;
        model.prepare_image_embeddings(input)?;
        let changed = Tensor::from_f32(
            pixels.shape().dims().to_vec(),
            &values.iter().map(|v| v + 0.125).collect::<Vec<_>>(),
        )?;
        model.prepare_image_embeddings(LlmInput::with_image(
            &tokens,
            ImageInput::new(&changed, &grids),
        ))?;
        let cached = model.prepare_image_embeddings(input)?.0;
        assert_eq!(
            transfers::to_cpu(&cached)?.to_f32_vec()?,
            original,
            "cached image embeddings changed after the vision workspace was overwritten"
        );
        model.configure_context_reuse(false)?;
        if let Some(checkpoint) = stable_checkpoint(&tokens) {
            let (embeddings, identity) = model.prepare_image_embeddings(input)?;
            model
                .runtime
                .target_mut()
                .attention_mut()
                .set_image_positions(Some(identity.positions));
            model.scalar_image_chunk(&tokens, &embeddings, 0, checkpoint)?;
            model.scalar_image_chunk(&tokens, &embeddings, checkpoint, tokens.len())?;
        } else {
            model.prefill(input)?;
        }
        // This is the existing text append implementation, running against
        // an image context with the original position plan's continuation.
        let expected = model
            .runtime
            .target_mut()
            .append_prefill(&model.ctx, &model.decoder, &model.weights, &suffix)?
            .logits;
        let expected = transfers::to_cpu(&expected)?.to_f32_vec()?;
        model.reset();
        model.configure_context_reuse(true)?;
        model.prefill(input)?;
        model.forward(&[20], tokens.len() as u32)?;
        model.reset();
        let mut followup = tokens.clone();
        followup.extend(suffix);
        let actual = model.prefill(LlmInput::with_image(
            &followup,
            ImageInput::new(&pixels, &grids),
        ))?;
        assert_eq!(model.last_reused_tokens, tokens.len());
        let actual = transfers::to_cpu(&actual)?.to_f32_vec()?;
        assert!(actual.iter().all(|v| v.is_finite()));
        assert_eq!(
            actual, expected,
            "restored image state differs from uncached chunk execution"
        );
        Ok(())
    }
}
