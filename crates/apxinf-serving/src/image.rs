//! Container decoding and HF chat/image processing at the serving boundary.
//! The native model receives only token IDs and canonical patch tensors.
use crate::{error::ApiError, protocol::ChatCompletionMessage};
use apxinf_core::Tensor;
use apxinf_tokenizer::ChatTemplateMessage;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub(crate) struct ImageMessage {
    messages: Vec<(usize, Vec<Value>)>,
}
pub(crate) struct PreparedImage {
    pub tokens: Vec<u32>,
    pub pixels: Tensor,
    pub grids: Vec<[u32; 3]>,
}

pub(crate) fn parse_messages(
    messages: &[ChatCompletionMessage],
) -> Result<(Vec<ChatCompletionMessage>, Option<ImageMessage>), ApiError> {
    let mut text = messages.to_vec();
    let mut image_messages = Vec::new();
    let mut image_count = 0usize;
    for (index, message) in messages.iter().enumerate() {
        let Some(parts) = message.content.as_ref().and_then(Value::as_array) else {
            continue;
        };
        // Tool results may be arbitrary JSON arrays, not multimodal content parts.
        if message.role == "tool" {
            continue;
        }
        if message.role != "user" {
            return Err(invalid(
                "content arrays are supported only for user messages",
            ));
        }
        let mut strings = Vec::new();
        let mut has_image = false;
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("text") => strings.push(
                    part.get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("text parts require a text string"))?,
                ),
                Some("image_url") => {
                    let url = part
                        .get("image_url")
                        .and_then(|v| v.get("url"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("image_url requires an object with a url string"))?;
                    if !(url.starts_with("data:image/") && url.contains(";base64,"))
                        && !url.starts_with("https://")
                    {
                        return Err(invalid(
                            "image_url must be a base64 image data URL or a public HTTPS URL",
                        ));
                    }
                    image_count += 1;
                    if image_count > 16 {
                        return Err(invalid("at most 16 images are supported per request"));
                    }
                    has_image = true;
                }
                _ => return Err(invalid("content parts must be text or image_url")),
            }
        }
        text[index].content = Some(Value::String(strings.join("")));
        if has_image {
            image_messages.push((index, parts.clone()));
        }
    }
    let image = (!image_messages.is_empty()).then_some(ImageMessage {
        messages: image_messages,
    });
    Ok((text, image))
}

pub(crate) fn preprocess(
    model: &Path,
    messages: &[ChatTemplateMessage],
    original_count: usize,
    image: ImageMessage,
    options: &Map<String, Value>,
    processor: &mut Option<ImageProcessor>,
) -> Result<PreparedImage, ApiError> {
    let mut messages = serde_json::to_value(messages).map_err(|e| invalid(e.to_string()))?;
    let array = messages
        .as_array_mut()
        .ok_or_else(|| invalid("invalid processor messages"))?;
    let offset = array
        .len()
        .checked_sub(original_count)
        .ok_or_else(|| invalid("invalid message mapping"))?;
    for (index, content) in image.messages {
        array[index + offset]["content"] = Value::Array(content);
    }
    let payload = serde_json::to_vec(&serde_json::json!({"messages":messages,"options":options}))
        .map_err(|e| invalid(e.to_string()))?;
    if processor.is_none() {
        *processor = Some(ImageProcessor::new(model)?);
    }
    let result = processor.as_mut().unwrap().process(&payload);
    if processor.as_ref().is_some_and(|worker| worker.broken) {
        processor.take();
    }
    result
}

/// One lazy CPU processor per serial engine. No request or pixel history is retained.
pub(crate) struct ImageProcessor {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    broken: bool,
}
impl ImageProcessor {
    fn new(model: &Path) -> Result<Self, ApiError> {
        let python = std::env::var("APXINF_PYTHON").unwrap_or_else(|_| "python3".into());
        let mut child = Command::new(&python)
            .arg("-c")
            .arg(include_str!("image_processor.py"))
            .arg(model)
            .arg("--worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| ApiError::unavailable(format!("image processor {python}: {error}")))?;
        let input = child.stdin.take().unwrap();
        let output = BufReader::new(child.stdout.take().unwrap());
        let mut worker = Self {
            child,
            input,
            output,
            broken: false,
        };
        let header = worker.header()?;
        let ready: Value = serde_json::from_slice(&header).map_err(|error| {
            ApiError::unavailable(format!("invalid processor startup response: {error}"))
        })?;
        worker.check_error(&ready)?;
        if ready.get("ready") != Some(&Value::Bool(true)) {
            return Err(ApiError::unavailable(
                "image processor failed to become ready",
            ));
        }
        Ok(worker)
    }
    fn io_error(&mut self, error: std::io::Error) -> ApiError {
        self.broken = true;
        ApiError::unavailable(format!(
            "image processor stopped: {error}; check APXINF_PYTHON and processor dependencies"
        ))
    }
    fn header(&mut self) -> Result<Vec<u8>, ApiError> {
        let mut length = [0u8; 4];
        if let Err(error) = self.output.read_exact(&mut length) {
            return Err(self.io_error(error));
        }
        let length = u32::from_le_bytes(length) as usize;
        if length > 16 * 1024 * 1024 {
            self.broken = true;
            return Err(ApiError::unavailable(
                "oversized image processor response header",
            ));
        }
        let mut header = vec![0; length];
        if let Err(error) = self.output.read_exact(&mut header) {
            return Err(self.io_error(error));
        }
        Ok(header)
    }
    fn check_error(&mut self, header: &Value) -> Result<(), ApiError> {
        if let Some(message) = header.get("error").and_then(Value::as_str) {
            if header.get("unavailable").and_then(Value::as_bool) == Some(true) {
                self.broken = true;
                return Err(ApiError::unavailable(message));
            }
            return Err(invalid(format!("image preprocessing failed: {message}")));
        }
        Ok(())
    }
    fn process(&mut self, payload: &[u8]) -> Result<PreparedImage, ApiError> {
        if payload.len() > 32 * 1024 * 1024 {
            return Err(invalid("oversized image processor request"));
        }
        if let Err(error) = self
            .input
            .write_all(&(payload.len() as u32).to_le_bytes())
            .and_then(|_| self.input.write_all(payload))
            .and_then(|_| self.input.flush())
        {
            return Err(self.io_error(error));
        }
        let header = self.header()?;
        let meta: Value = serde_json::from_slice(&header).map_err(|error| {
            self.broken = true;
            ApiError::unavailable(format!("invalid image processor metadata: {error}"))
        })?;
        self.check_error(&meta)?;
        let shape = meta.get("shape").and_then(Value::as_array);
        let bytes = shape.filter(|v| v.len() == 2).and_then(|v| {
            let rows = usize::try_from(v[0].as_u64()?).ok()?;
            let columns = usize::try_from(v[1].as_u64()?).ok()?;
            if rows == 0 || rows > 4096 || columns == 0 || columns > 8192 {
                return None;
            }
            rows.checked_mul(columns)?.checked_mul(4)
        });
        let Some(bytes) = bytes else {
            self.broken = true;
            return Err(ApiError::unavailable(
                "invalid image processor tensor shape",
            ));
        };
        let mut data = Vec::with_capacity(4 + header.len() + bytes);
        data.extend_from_slice(&(header.len() as u32).to_le_bytes());
        data.extend_from_slice(&header);
        data.resize(data.len() + bytes, 0);
        if let Err(error) = self.output.read_exact(&mut data[4 + header.len()..]) {
            return Err(self.io_error(error));
        }
        let result = decode_output(&data);
        if result.is_err() {
            self.broken = true;
        }
        result
    }
}
impl Drop for ImageProcessor {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn decode_output(data: &[u8]) -> Result<PreparedImage, ApiError> {
    #[derive(Deserialize)]
    struct Header {
        tokens: Vec<u32>,
        shape: [usize; 2],
        grids: Vec<[u32; 3]>,
    }
    if data.len() < 4 {
        return Err(invalid("truncated image processor output"));
    }
    let header_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    let end = 4usize
        .checked_add(header_len)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| invalid("invalid image processor header length"))?;
    let header: Header = serde_json::from_slice(&data[4..end])
        .map_err(|e| invalid(format!("invalid image processor metadata: {e}")))?;
    let expected = header.shape[0]
        .checked_mul(header.shape[1])
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| invalid("image tensor size overflow"))?;
    if data.len() - end != expected
        || header.shape[1] == 0
        || header.shape[1] > 8192
        || header.shape[0] == 0
        || header.shape[0] > 4096
        || header.grids.is_empty()
        || header.grids.len() > 16
    {
        return Err(invalid("invalid image processor tensor shape or length"));
    }
    let values: Vec<f32> = data[end..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("image pixels must be finite"));
    }
    let pixels =
        Tensor::from_f32(header.shape.to_vec(), &values).map_err(|e| invalid(e.to_string()))?;
    Ok(PreparedImage {
        tokens: header.tokens,
        pixels,
        grids: header.grids,
    })
}
fn invalid(message: impl Into<String>) -> ApiError {
    ApiError::invalid(message, Some("messages".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn messages(content: Value) -> Vec<ChatCompletionMessage> {
        serde_json::from_value(serde_json::json!([{"role":"user","content":content}])).unwrap()
    }
    #[test]
    fn accepts_text_and_one_image_without_losing_order() {
        let content = serde_json::json!([{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"text","text":"after"}]);
        let (text, image) = parse_messages(&messages(content.clone())).unwrap();
        assert_eq!(text[0].content, Some(Value::String("beforeafter".into())));
        assert_eq!(
            image.unwrap().messages[0].1,
            content.as_array().unwrap().clone()
        );
    }
    #[test]
    fn multiple_images_across_messages_preserve_order() {
        let parts = serde_json::json!([
            {"type":"image_url","image_url":{"url":"https://example.com/first.png"}},
            {"type":"text","text":"then"},
            {"type":"image_url","image_url":{"url":"https://example.com/second.png"}}
        ]);
        let mut input = messages(parts.clone());
        input.push(input[0].clone());
        let (_, image) = parse_messages(&input).unwrap();
        let image = image.unwrap();
        assert_eq!(image.messages.len(), 2);
        for (i, (index, content)) in image.messages.iter().enumerate() {
            assert_eq!(*index, i);
            assert_eq!(content, parts.as_array().unwrap());
        }
    }

    #[test]
    fn preserves_existing_json_array_tool_results() {
        let messages: Vec<ChatCompletionMessage> = serde_json::from_value(serde_json::json!([
            {"role":"tool","tool_call_id":"call_1","content":[{"temperature":20},{"temperature":21}]}
        ])).unwrap();
        let (text, image) = parse_messages(&messages).unwrap();
        assert!(image.is_none());
        assert_eq!(text[0].content, messages[0].content);
    }

    #[test]
    fn rejects_excess_images_unknown_parts_and_local_urls() {
        let image =
            serde_json::json!({"type":"image_url","image_url":{"url":"https://example.com/a.png"}});
        for parts in [
            Value::Array(vec![image; 17]),
            serde_json::json!([{"type":"video"}]),
            serde_json::json!([{"type":"image_url","image_url":{"url":"file:///etc/passwd"}}]),
            serde_json::json!([{"type":"text","text":7}]),
        ] {
            assert!(parse_messages(&messages(parts)).is_err());
        }
    }
    #[test]
    fn rejects_truncated_or_nonfinite_processor_output() {
        for bytes in [vec![], vec![255, 255, 255, 255]] {
            assert!(decode_output(&bytes).is_err());
        }
        let header = serde_json::to_vec(
            &serde_json::json!({"tokens":[1],"shape":[1,1536],"grids":[[1,2,2]]}),
        )
        .unwrap();
        let mut data = (header.len() as u32).to_le_bytes().to_vec();
        data.extend(header);
        assert!(decode_output(&data).is_err());
        data.extend(
            std::iter::repeat(f32::NAN.to_le_bytes())
                .take(1536)
                .flatten(),
        );
        assert!(decode_output(&data).is_err());
    }
}
