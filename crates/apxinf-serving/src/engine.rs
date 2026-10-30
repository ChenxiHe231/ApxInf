use std::collections::BTreeSet;
use std::path::PathBuf;

use apxinf_core::Device;
use apxinf_model::{
    AutoModel, GenerationOptions, LlmInput, LoadOptions, LoadedModel, SamplingMode,
};
use apxinf_tokenizer::{ChatTemplateMessage, ChatToolCall, ChatToolCallFunction, Tokenizer};
use serde_json::{Map, Value};

use crate::error::ApiError;
use crate::protocol::{ChatCompletionMessage, ToolDefinition};
use crate::tool_calling::ToolChoice;

#[derive(Clone, Debug)]
pub struct EngineRequest {
    pub messages: Vec<ChatCompletionMessage>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub chat_template_kwargs: Map<String, Value>,
    pub enable_thinking: bool,
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_k: Option<i64>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct EngineOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub reused_prompt_tokens: usize,
    pub completion_tokens: usize,
    pub hit_length: bool,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub generation_tps: Option<f64>,
    pub total_latency_ms: Option<f64>,
}

pub trait ChatEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError>;

    /// Generate a completion while publishing decoded text as it becomes
    /// stable. Returning `false` from the callback means the HTTP consumer is
    /// gone; implementations should stop publishing and may cancel inference
    /// when their runtime supports cooperative cancellation.
    ///
    /// The default keeps third-party/test engines source compatible. It emits
    /// their complete result once, while native ApxInf engines override this
    /// method to forward model token callbacks during generation.
    fn complete_stream(
        &mut self,
        request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        let output = self.complete(request)?;
        if !output.text.is_empty() {
            let _ = on_text(&output.text);
        }
        Ok(output)
    }
}

pub struct ApxInfEngineConfig {
    pub model_path: PathBuf,
    pub device: Device,
    pub load_options: LoadOptions,
    pub default_max_tokens: usize,
    pub max_input_tokens: usize,
    pub enable_context_reuse: bool,
}

pub struct ApxInfChatEngine {
    tokenizer: Tokenizer,
    model: LoadedModel,
    default_max_tokens: usize,
    max_input_tokens: usize,
    enable_context_reuse: bool,
}

impl ApxInfChatEngine {
    pub fn load(mut config: ApxInfEngineConfig) -> Result<Self, ApiError> {
        let admitted_tokens = config
            .max_input_tokens
            .checked_add(config.default_max_tokens)
            .ok_or_else(|| ApiError::unavailable("configured sequence capacity overflows usize"))?;
        // Qwen3.5 MTP verifies a fixed four-row target window before applying
        // output-budget truncation. Three lookahead slots let the last round
        // remain speculative; other text runtimes simply retain a tiny unused
        // tail in their fixed cache.
        let required_capacity = admitted_tokens
            .checked_add(3)
            .ok_or_else(|| ApiError::unavailable("configured MTP lookahead overflows usize"))?;
        match config.load_options.text_kv_capacity {
            Some(capacity) if capacity < required_capacity => {
                return Err(ApiError::unavailable(format!(
                    "text KV capacity {capacity} is smaller than the serving admission plus MTP lookahead {required_capacity}"
                )));
            }
            Some(_) => {}
            None => config.load_options.text_kv_capacity = Some(required_capacity),
        }
        let tokenizer_path = config.model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            ApiError::unavailable(format!(
                "failed to load tokenizer {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        if !tokenizer.has_chat_template() {
            return Err(ApiError::unavailable(
                "the checkpoint does not provide a chat template",
            ));
        }
        let mut model =
            AutoModel::load_model(config.device, &config.model_path, &config.load_options)
                .map_err(|error| ApiError::unavailable(format!("failed to load model: {error}")))?;
        model.text_capabilities().map_err(|error| {
            ApiError::unavailable(format!("model is not a text model: {error}"))
        })?;
        model
            .set_context_reuse(config.enable_context_reuse)
            .map_err(|error| {
                ApiError::unavailable(format!("failed to configure context reuse: {error}"))
            })?;
        Ok(Self {
            tokenizer,
            model,
            default_max_tokens: config.default_max_tokens,
            max_input_tokens: config.max_input_tokens,
            enable_context_reuse: config.enable_context_reuse,
        })
    }
}

impl ChatEngine for ApxInfChatEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.complete_stream(request, &mut |_| true)
    }

    fn complete_stream(
        &mut self,
        request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        let mut messages = to_template_messages(&request.messages)?;
        apply_tool_choice_instruction(&mut messages, &request.tool_choice);
        let mut options = request.chat_template_kwargs;
        for reserved in [
            "messages",
            "bos_token",
            "eos_token",
            "add_generation_prompt",
        ] {
            if options.contains_key(reserved) {
                return Err(ApiError::invalid(
                    format!("reserved chat_template_kwargs key: {reserved}"),
                    Some("chat_template_kwargs".to_owned()),
                ));
            }
        }
        options.insert("enable_thinking".into(), request.enable_thinking.into());
        let rendered_tools = match &request.tool_choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Function(name) => request
                .tools
                .iter()
                .filter(|tool| tool.function.name == *name)
                .cloned()
                .collect(),
            ToolChoice::Auto | ToolChoice::Required => request.tools.clone(),
        };
        if !rendered_tools.is_empty() {
            options.insert(
                "tools".into(),
                serde_json::to_value(&rendered_tools)
                    .map_err(|error| ApiError::internal(error.to_string()))?,
            );
        }
        let prompt = self
            .tokenizer
            .apply_chat_template_messages_with_options(&messages, &options)
            .map_err(|error| ApiError::invalid(error.to_string(), Some("messages".to_owned())))?;
        let prompt_tokens = self
            .tokenizer
            .encode(&prompt)
            .map_err(|error| ApiError::invalid(error.to_string(), Some("messages".to_owned())))?;
        if prompt_tokens.is_empty() {
            return Err(ApiError::invalid(
                "chat template produced an empty prompt",
                Some("messages".to_owned()),
            ));
        }
        if prompt_tokens.len() > self.max_input_tokens {
            return Err(ApiError::invalid(
                format!(
                    "prompt has {} tokens, exceeding the configured limit of {}",
                    prompt_tokens.len(),
                    self.max_input_tokens
                ),
                Some("messages".to_owned()),
            ));
        }
        let reusable_prefix_tokens = if self.enable_context_reuse {
            system_prefix_tokens(&self.tokenizer, &messages, &options, &prompt_tokens)?
        } else {
            0
        };
        if reusable_prefix_tokens > 0 {
            eprintln!(
                "system prefix candidate: {reusable_prefix_tokens} of {} prompt tokens",
                prompt_tokens.len()
            );
        }

        let default_eos = self
            .model
            .generation_defaults()
            .map_err(|error| ApiError::internal(error.to_string()))?
            .eos_token_ids
            .clone();
        let eos_token_ids = default_eos
            .clone()
            .or_else(|| self.tokenizer.eos_token_id().map(|id| vec![id]));
        let max_tokens = request
            .max_tokens
            .max(1)
            .min(self.default_max_tokens.max(1));
        let generation = GenerationOptions {
            max_new_tokens: Some(max_tokens),
            eos_token_ids: eos_token_ids.clone(),
            sampling_mode: request.temperature.map(|temperature| {
                if temperature == 0.0 {
                    SamplingMode::Greedy
                } else {
                    SamplingMode::Random
                }
            }),
            temperature: request.temperature,
            top_k: request.top_k,
            top_p: request.top_p,
            repetition_penalty: None,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            seed: request.seed,
            return_logprob: Some(false),
        };
        let mut decoder = IncrementalTextDecoder::new(&self.tokenizer);
        let mut decode_error = None;
        let mut consumer_connected = true;
        let output = self
            .model
            .generate_streaming_with_options(
                LlmInput::text_with_reusable_prefix(&prompt_tokens, reusable_prefix_tokens),
                &generation,
                |token| {
                    if !consumer_connected
                        || decode_error.is_some()
                        || eos_token_ids
                            .as_ref()
                            .is_some_and(|eos| eos.contains(&token.token_id))
                    {
                        return;
                    }
                    match decoder.push(token.token_id) {
                        Ok(Some(delta)) => consumer_connected = on_text(&delta),
                        Ok(None) => {}
                        Err(error) => decode_error = Some(error),
                    }
                },
            )
            .map_err(|error| ApiError::internal(format!("generation failed: {error}")))?;
        if let Some(error) = decode_error {
            return Err(ApiError::internal(format!(
                "incremental decode failed: {error}"
            )));
        }
        let mut output_ids = output.token_ids();
        let stopped_on_eos = output_ids.last().is_some_and(|token| {
            eos_token_ids
                .as_ref()
                .is_some_and(|eos| eos.contains(token))
        });
        if stopped_on_eos {
            output_ids.pop();
        }
        let text = self
            .tokenizer
            .decode_with_options(&output_ids, false)
            .map_err(|error| ApiError::internal(format!("decode failed: {error}")))?;
        if consumer_connected {
            if let Some(delta) = decoder.finish(&text).map_err(ApiError::internal)? {
                let _ = on_text(&delta);
            }
        }
        let completion_tokens = output.profile.output_tokens();
        eprintln!(
            "request profile: prompt_tokens={} reused_prompt_tokens={} completion_tokens={} ttft_ms={:?} tpot_ms={:?} generation_tps={:?} total_latency_ms={:?}",
            output.profile.input_tokens(),
            output.profile.reused_input_tokens(),
            completion_tokens,
            output.profile.ttft_ms(),
            output.profile.tpot_ms(),
            output.profile.generation_tps(),
            output.profile.total_latency_ms(),
        );
        Ok(EngineOutput {
            text,
            prompt_tokens: output.profile.input_tokens(),
            reused_prompt_tokens: output.profile.reused_input_tokens(),
            completion_tokens,
            hit_length: !stopped_on_eos && completion_tokens >= max_tokens,
            ttft_ms: output.profile.ttft_ms(),
            tpot_ms: output.profile.tpot_ms(),
            generation_tps: output.profile.generation_tps(),
            total_latency_ms: output.profile.total_latency_ms(),
        })
    }
}

/// Find the token span emitted before conversation content. Rendering a
/// system/tool-only probe with a dummy user message keeps this independent of
/// checkpoint-specific chat-template delimiters. Two distinct dummy contents
/// prevent their content tokens from being mistaken for the reusable prefix.
fn system_prefix_tokens(
    tokenizer: &Tokenizer,
    messages: &[ChatTemplateMessage],
    options: &Map<String, Value>,
    prompt_tokens: &[u32],
) -> Result<usize, ApiError> {
    let system_messages = messages
        .iter()
        .take_while(|message| message.role == "system")
        .cloned()
        .collect::<Vec<_>>();
    let has_tools = options
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if system_messages.is_empty() && !has_tools {
        return Ok(0);
    }

    let render_probe = |content: &str| -> Result<Vec<u32>, ApiError> {
        let mut probe = system_messages.clone();
        probe.push(ChatTemplateMessage::text("user", content));
        let rendered = tokenizer
            .apply_chat_template_messages_with_options(&probe, options)
            .map_err(|error| ApiError::invalid(error.to_string(), Some("messages".to_owned())))?;
        tokenizer
            .encode(&rendered)
            .map_err(|error| ApiError::invalid(error.to_string(), Some("messages".to_owned())))
    };
    let first = render_probe("APXINF_SYSTEM_PREFIX_BOUNDARY_A")?;
    let second = render_probe("\u{2603}APXINF_SYSTEM_PREFIX_BOUNDARY_B")?;
    let probe_prefix = common_token_prefix(&first, &second);
    Ok(probe_prefix.min(common_token_prefix(&first, prompt_tokens)))
}

fn common_token_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Cumulative decoding is required because byte-fallback tokens may not be
/// valid UTF-8 in isolation. We publish only the prefix that was unchanged by
/// the following token, then flush the final suffix when generation ends. This
/// gives token-granular output with a one-token stabilization delay and never
/// exposes a replacement character that a later token would repair.
struct IncrementalTextDecoder<'a> {
    tokenizer: &'a Tokenizer,
    token_ids: Vec<u32>,
    previous: String,
    emitted_bytes: usize,
}

impl<'a> IncrementalTextDecoder<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            token_ids: Vec::new(),
            previous: String::new(),
            emitted_bytes: 0,
        }
    }

    fn push(&mut self, token_id: u32) -> Result<Option<String>, String> {
        self.token_ids.push(token_id);
        let current = self
            .tokenizer
            .decode_with_options(&self.token_ids, false)
            .map_err(|error| error.to_string())?;
        let stable_end = common_prefix_bytes(&self.previous, &current);
        if stable_end < self.emitted_bytes {
            return Err("tokenizer changed text that was already streamed".into());
        }
        let delta = self.previous[self.emitted_bytes..stable_end].to_owned();
        self.emitted_bytes = stable_end;
        self.previous = current;
        Ok((!delta.is_empty()).then_some(delta))
    }

    fn finish(&mut self, text: &str) -> Result<Option<String>, String> {
        if !text.is_char_boundary(self.emitted_bytes)
            || !text
                .get(..self.emitted_bytes)
                .is_some_and(|prefix| self.previous.starts_with(prefix))
        {
            return Err("final decode changed text that was already streamed".into());
        }
        let delta = text[self.emitted_bytes..].to_owned();
        self.emitted_bytes = text.len();
        Ok((!delta.is_empty()).then_some(delta))
    }
}

fn common_prefix_bytes(left: &str, right: &str) -> usize {
    left.char_indices()
        .zip(right.chars())
        .take_while(|((_, left), right)| left == right)
        .last()
        .map_or(0, |((offset, character), _)| offset + character.len_utf8())
}

fn to_template_messages(
    messages: &[ChatCompletionMessage],
) -> Result<Vec<ChatTemplateMessage>, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid(
            "messages must not be empty",
            Some("messages".to_owned()),
        ));
    }
    let mut seen_calls = BTreeSet::new();
    let mut output = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let param = || Some(format!("messages[{index}]"));
        match message.role.as_str() {
            "system" | "user" => {
                if message.tool_calls.is_some() || message.tool_call_id.is_some() {
                    return Err(ApiError::invalid(
                        "system/user messages cannot carry tool call fields",
                        param(),
                    ));
                }
                output.push(ChatTemplateMessage::text(
                    &message.role,
                    text_content(message.content.as_ref(), false, index)?,
                ));
            }
            "assistant" => {
                if message.tool_call_id.is_some() {
                    return Err(ApiError::invalid(
                        "assistant messages cannot carry tool_call_id",
                        param(),
                    ));
                }
                let content = optional_text_content(message.content.as_ref(), index)?;
                let mut calls = Vec::new();
                for call in message.tool_calls.as_deref().unwrap_or_default() {
                    if call.kind != "function" || call.id.is_empty() {
                        return Err(ApiError::invalid(
                            "assistant tool calls require a nonempty id and type 'function'",
                            param(),
                        ));
                    }
                    let arguments: Value =
                        serde_json::from_str(&call.function.arguments).map_err(|error| {
                            ApiError::invalid(
                                format!("assistant tool arguments are not JSON: {error}"),
                                param(),
                            )
                        })?;
                    let arguments = arguments.as_object().cloned().ok_or_else(|| {
                        ApiError::invalid(
                            "assistant tool arguments must encode a JSON object",
                            param(),
                        )
                    })?;
                    if !seen_calls.insert(call.id.clone()) {
                        return Err(ApiError::invalid(
                            format!("duplicate tool call id '{}'", call.id),
                            param(),
                        ));
                    }
                    calls.push(ChatToolCall {
                        id: Some(call.id.clone()),
                        kind: Some("function".into()),
                        function: ChatToolCallFunction {
                            name: call.function.name.clone(),
                            arguments,
                        },
                    });
                }
                output.push(if calls.is_empty() {
                    ChatTemplateMessage {
                        role: "assistant".into(),
                        content,
                        reasoning_content: message.reasoning_content.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                    }
                } else {
                    ChatTemplateMessage::assistant_tool_calls(
                        content,
                        message.reasoning_content.clone(),
                        calls,
                    )
                });
            }
            "tool" => {
                let id = message.tool_call_id.as_deref().ok_or_else(|| {
                    ApiError::invalid("tool messages require tool_call_id", param())
                })?;
                if !seen_calls.contains(id) {
                    return Err(ApiError::invalid(
                        format!("dangling tool_call_id '{id}'"),
                        param(),
                    ));
                }
                if message.tool_calls.is_some() {
                    return Err(ApiError::invalid(
                        "tool messages cannot carry tool_calls",
                        param(),
                    ));
                }
                output.push(ChatTemplateMessage::tool(
                    text_content(message.content.as_ref(), true, index)?,
                    id,
                ));
            }
            other => {
                return Err(ApiError::invalid(
                    format!("unsupported message role '{other}'"),
                    Some(format!("messages[{index}].role")),
                ));
            }
        }
    }
    Ok(output)
}

fn apply_tool_choice_instruction(messages: &mut Vec<ChatTemplateMessage>, choice: &ToolChoice) {
    let instruction = match choice {
        ToolChoice::Required => Some(
            "You must call at least one of the provided tools. Do not answer without a tool call."
                .to_owned(),
        ),
        ToolChoice::Function(name) => Some(format!(
                "You must call the function '{name}'. Do not call another function and do not answer without that tool call."
            )),
        ToolChoice::None | ToolChoice::Auto => None,
    };
    let Some(instruction) = instruction else {
        return;
    };
    if let Some(system) = messages
        .first_mut()
        .filter(|message| message.role == "system")
    {
        let content = system.content.get_or_insert_with(String::new);
        if !content.is_empty() {
            content.push_str("\n\n");
        }
        content.push_str(&instruction);
    } else {
        messages.insert(0, ChatTemplateMessage::text("system", instruction));
    }
}

fn optional_text_content(
    content: Option<&Value>,
    index: usize,
) -> Result<Option<String>, ApiError> {
    match content {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(ApiError::invalid(
            "only text content is supported by this text-only server",
            Some(format!("messages[{index}].content")),
        )),
    }
}

fn text_content(
    content: Option<&Value>,
    allow_json: bool,
    index: usize,
) -> Result<String, ApiError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(value @ (Value::Object(_) | Value::Array(_))) if allow_json => {
            serde_json::to_string(value).map_err(|error| ApiError::internal(error.to_string()))
        }
        _ => Err(ApiError::invalid(
            "message content must be text",
            Some(format!("messages[{index}].content")),
        )),
    }
}
