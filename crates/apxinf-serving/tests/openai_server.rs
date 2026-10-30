use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::time::Duration;

use apxinf_serving::{protocol::ChatCompletionMessage, tool_calling::ToolChoice};
use apxinf_serving::{
    router, ApiConfig, ApiError, ChatEngine, EngineOutput, EngineRequest, WorkerClient,
};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use serde_json::Value;
use tokio_stream::StreamExt;
use tower::ServiceExt;

struct FakeEngine {
    calls: Arc<AtomicUsize>,
}

async fn wait_until_ready(worker: &WorkerClient) {
    for _ in 0..1_000 {
        if worker.ready() {
            return;
        }
        // Worker construction happens on a native thread. A bounded sleep is
        // required here: repeatedly yielding only the Tokio task does not
        // guarantee that the OS schedules that thread on slower ARM hosts.
        std::thread::sleep(Duration::from_millis(1));
        tokio::task::yield_now().await;
    }
    panic!(
        "worker did not become ready: {} {:?}",
        worker.status_message(),
        worker.startup_error()
    );
}

impl ChatEngine for FakeEngine {
    fn complete(&mut self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if request
            .messages
            .iter()
            .any(|message| message.content.as_ref().and_then(Value::as_str) == Some("fail"))
        {
            return Err(ApiError::internal("fake generation failure"));
        }
        let text = if request.tools.is_empty() {
            "你好，我是离线 fake engine。".to_owned()
        } else {
            "<tool_call>\n<function=weather.get>\n<parameter=city>\n上海\n</parameter>\n<parameter=date>\n2026-10-22\n</parameter>\n</function>\n</tool_call>".to_owned()
        };
        Ok(EngineOutput {
            text,
            prompt_tokens: 11,
            reused_prompt_tokens: 3,
            completion_tokens: 7,
            hit_length: false,
            ttft_ms: Some(10.0),
            tpot_ms: Some(5.0),
            generation_tps: Some(200.0),
            total_latency_ms: Some(40.0),
        })
    }
}

async fn app() -> (axum::Router, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let worker_calls = calls.clone();
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(FakeEngine {
            calls: worker_calls,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    (
        router(
            ApiConfig {
                served_model_name: "fake-model".into(),
                max_output_tokens: 128,
            },
            worker,
        ),
        calls,
    )
}

async fn json_response(app: axum::Router, body: Value) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn sse_response(app: axum::Router, body: Value) -> (StatusCode, Vec<Value>, bool) {
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert_eq!(
        response.headers()["content-type"],
        "text/event-stream",
        "unexpected response headers"
    );
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let mut values = Vec::new();
    let mut done = false;
    for event in text.split("\n\n").filter(|event| !event.is_empty()) {
        let data = event.strip_prefix("data: ").expect("SSE data field");
        if data == "[DONE]" {
            done = true;
        } else {
            values.push(serde_json::from_str(data).unwrap());
        }
    }
    (status, values, done)
}

#[tokio::test]
async fn serves_health_models_and_plain_chat() {
    let (app, calls) = app().await;
    let health = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let models = app
        .clone()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(models.status(), StatusCode::OK);

    let (status, response) = json_response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "messages":[{"role":"user","content":"你好"}],
            "temperature":0,
            "top_k":20,
            "max_tokens":32
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["choices"][0]["message"]["role"], "assistant");
    assert_eq!(response["choices"][0]["finish_reason"], "stop");
    assert_eq!(response["usage"]["total_tokens"], 18);
    assert_eq!(
        response["usage"]["prompt_tokens_details"]["cached_tokens"],
        3
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn converts_qwen_xml_to_openai_tool_calls() {
    let (app, _) = app().await;
    let (status, response) = json_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"明天上海天气怎么样"}],
            "tools":[{"type":"function","function":{
                "name":"weather.get",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"},"date":{"type":"string"}},
                    "required":["city","date"],
                    "additionalProperties":false
                }
            }}],
            "tool_choice":"required",
            "parallel_tool_calls":false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
    let call = &response["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "weather.get");
    let arguments: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(
        arguments,
        serde_json::json!({"city":"上海","date":"2026-10-22"})
    );
}

#[tokio::test]
async fn rejects_invalid_stream_options_and_bad_tool_schema_before_inference() {
    let (app, calls) = app().await;
    let (status, _) = json_response(
        app.clone(),
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "stream_options":{"include_usage":true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, response) = json_response(
        app.clone(),
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "temperature":1.0,
            "top_k":-1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(response["error"]["param"], "top_k");

    let (status, _) = json_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"x"}],
            "tools":[{"type":"function","function":{"name":"bad name","parameters":{}}}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn streams_text_finish_usage_and_done_in_openai_sse_shape() {
    let (app, calls) = app().await;
    let (status, chunks, done) = sse_response(
        app,
        serde_json::json!({
            "model":"fake-model",
            "messages":[{"role":"user","content":"你好"}],
            "stream":true,
            "stream_options":{"include_usage":true}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"].get(0))
        .filter_map(|choice| choice["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "你好，我是离线 fake engine。");
    assert!(chunks
        .iter()
        .any(|chunk| { chunk["choices"][0]["finish_reason"].as_str() == Some("stop") }));
    let usage = chunks
        .iter()
        .find(|chunk| chunk["choices"].as_array().is_some_and(Vec::is_empty))
        .expect("terminal usage chunk");
    assert_eq!(usage["usage"]["total_tokens"], 18);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

struct IncrementalEngine {
    release: Receiver<()>,
}

impl ChatEngine for IncrementalEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("streaming requests must use complete_stream")
    }

    fn complete_stream(
        &mut self,
        _request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        if !on_text("first") {
            return Err(ApiError::internal("stream consumer closed"));
        }
        self.release.recv().unwrap();
        let _ = on_text(" second");
        Ok(EngineOutput {
            text: "first second".into(),
            prompt_tokens: 1,
            reused_prompt_tokens: 0,
            completion_tokens: 2,
            hit_length: false,
            ttft_ms: Some(1.0),
            tpot_ms: Some(2.0),
            generation_tps: Some(500.0),
            total_latency_ms: Some(3.0),
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delivers_content_before_generation_completes() {
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(IncrementalEngine {
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let app = router(
        ApiConfig {
            served_model_name: "incremental-model".into(),
            max_output_tokens: 16,
        },
        worker,
    );
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model":"incremental-model",
                        "messages":[{"role":"user","content":"x"}],
                        "stream":true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body().into_data_stream();
    let mut before_completion = String::new();
    while !before_completion.contains(r#""content":"first""#) {
        let bytes = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .expect("first token delta was not delivered before completion")
            .expect("SSE body ended before the first token delta")
            .expect("SSE body error");
        before_completion.push_str(std::str::from_utf8(&bytes).unwrap());
    }

    // The engine is still blocked here. Releasing it only after observing the
    // first delta proves the response was not synthesized from the final text.
    release_sender.send(()).unwrap();
    let mut after_completion = String::new();
    while let Some(bytes) = body.next().await {
        after_completion.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
    }
    assert!(after_completion.contains(r#""content":" second""#));
    assert!(after_completion.contains("data: [DONE]"));
}

#[tokio::test]
async fn streams_tool_call_delta_and_tool_calls_finish_reason() {
    let (app, _) = app().await;
    let (status, chunks, done) = sse_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"明天上海天气怎么样"}],
            "tools":[{"type":"function","function":{
                "name":"weather.get",
                "parameters":{
                    "type":"object",
                    "properties":{"city":{"type":"string"},"date":{"type":"string"}},
                    "required":["city","date"],
                    "additionalProperties":false
                }
            }}],
            "tool_choice":"required",
            "parallel_tool_calls":false,
            "stream":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    let tool_delta = chunks
        .iter()
        .find_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
        .expect("tool call delta");
    assert_eq!(tool_delta[0]["type"], "function");
    assert_eq!(tool_delta[0]["function"]["name"], "weather.get");
    let arguments: Value =
        serde_json::from_str(tool_delta[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(
        arguments,
        serde_json::json!({"city":"上海","date":"2026-10-22"})
    );
    assert!(chunks
        .iter()
        .any(|chunk| { chunk["choices"][0]["finish_reason"].as_str() == Some("tool_calls") }));
}

struct SplitToolEngine {
    before_close: SyncSender<()>,
    release_close: Receiver<()>,
}

impl ChatEngine for SplitToolEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("streaming requests must use complete_stream")
    }

    fn complete_stream(
        &mut self,
        _request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        let fragments = [
            "<to",
            "ol_call>\n<function=weather.get>\n<parameter=city>\n上",
            "海\n</parameter>\n<parameter=date>\n2026-",
            "10-22\n</parameter>\n</function>\n",
        ];
        for fragment in fragments {
            if !on_text(fragment) {
                return Err(ApiError::internal("stream consumer closed"));
            }
        }
        self.before_close.send(()).unwrap();
        self.release_close.recv().unwrap();
        let closing = "</tool_call>";
        let _ = on_text(closing);
        Ok(EngineOutput {
            text: format!("{}{}", fragments.concat(), closing),
            prompt_tokens: 9,
            reused_prompt_tokens: 0,
            completion_tokens: 10,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn buffers_split_tool_xml_until_it_can_emit_valid_openai_json() {
    let (before_close_sender, before_close_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(SplitToolEngine {
            before_close: before_close_sender,
            release_close: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let app = router(
        ApiConfig {
            served_model_name: "split-tool-model".into(),
            max_output_tokens: 32,
        },
        worker,
    );
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model":"split-tool-model",
                        "messages":[{"role":"user","content":"天气"}],
                        "tools":[{"type":"function","function":{
                            "name":"weather.get",
                            "parameters":{
                                "type":"object",
                                "properties":{
                                    "city":{"type":"string"},
                                    "date":{"type":"string"}
                                },
                                "required":["city","date"],
                                "additionalProperties":false
                            }
                        }}],
                        "tool_choice":"required",
                        "parallel_tool_calls":false,
                        "stream":true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let mut body = response.into_body().into_data_stream();
    before_close_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("all incomplete XML fragments reached the callback");
    assert!(
        tokio::time::timeout(Duration::from_millis(50), body.next())
            .await
            .is_err(),
        "incomplete tool XML must not leak into an SSE delta"
    );

    release_sender.send(()).unwrap();
    let mut wire = String::new();
    while let Some(bytes) = body.next().await {
        wire.push_str(std::str::from_utf8(&bytes.unwrap()).unwrap());
    }
    assert!(!wire.contains("<tool_call>"));
    assert!(!wire.contains("<parameter="));
    let tool_delta = wire
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .find(|chunk| chunk["choices"][0]["delta"]["tool_calls"].is_array())
        .expect("validated tool-call delta");
    let arguments = tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(arguments).unwrap(),
        serde_json::json!({"city":"上海","date":"2026-10-22"})
    );
}

#[tokio::test]
async fn reports_post_start_generation_errors_as_sse_error_then_done() {
    let (app, _) = app().await;
    let (status, chunks, done) = sse_response(
        app,
        serde_json::json!({
            "messages":[{"role":"user","content":"fail"}],
            "stream":true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(done);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0]["error"]["type"], "server_error");
    assert!(chunks[0]["error"]["message"]
        .as_str()
        .unwrap()
        .contains("fake generation failure"));
}

struct BlockingEngine {
    calls: Arc<AtomicUsize>,
    started: SyncSender<()>,
    release: Receiver<()>,
}

impl ChatEngine for BlockingEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let _ = self.started.send(());
        self.release.recv().unwrap();
        Ok(EngineOutput {
            text: "done".into(),
            prompt_tokens: 1,
            reused_prompt_tokens: 0,
            completion_tokens: 1,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

fn empty_engine_request() -> EngineRequest {
    EngineRequest {
        messages: vec![ChatCompletionMessage {
            role: "user".into(),
            content: Some(Value::String("x".into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: ToolChoice::None,
        chat_template_kwargs: serde_json::Map::new(),
        enable_thinking: false,
        max_tokens: 1,
        temperature: None,
        top_k: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        seed: None,
        disable_spec_decode: false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canceled_queued_request_is_skipped_by_the_single_worker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine_calls = calls.clone();
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(BlockingEngine {
            calls: engine_calls,
            started: started_sender,
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let first = worker.submit(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("first request started");
    let canceled = worker.submit(empty_engine_request()).unwrap();
    drop(canceled);
    release_sender.send(()).unwrap();
    first.await.unwrap().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn canceled_queued_stream_is_skipped_by_the_single_worker() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine_calls = calls.clone();
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(2, move || {
        Ok(Box::new(BlockingEngine {
            calls: engine_calls,
            started: started_sender,
            release: release_receiver,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let first = worker.submit(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("first request started");
    let canceled = worker.submit_stream(empty_engine_request()).unwrap();
    drop(canceled);
    release_sender.send(()).unwrap();
    first.await.unwrap().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

struct CancellationAwareEngine {
    started: SyncSender<()>,
    release: Receiver<()>,
    callback_result: SyncSender<bool>,
}

impl ChatEngine for CancellationAwareEngine {
    fn complete(&mut self, _request: EngineRequest) -> Result<EngineOutput, ApiError> {
        panic!("streaming requests must use complete_stream")
    }

    fn complete_stream(
        &mut self,
        _request: EngineRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<EngineOutput, ApiError> {
        self.started.send(()).unwrap();
        self.release.recv().unwrap();
        self.callback_result.send(on_text("x")).unwrap();
        Ok(EngineOutput {
            text: "x".into(),
            prompt_tokens: 1,
            reused_prompt_tokens: 0,
            completion_tokens: 1,
            hit_length: false,
            ttft_ms: None,
            tpot_ms: None,
            generation_tps: None,
            total_latency_ms: None,
        })
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_stream_disconnect_reaches_the_engine_callback() {
    let (started_sender, started_receiver) = sync_channel(1);
    let (release_sender, release_receiver) = sync_channel(1);
    let (result_sender, result_receiver) = sync_channel(1);
    let worker = WorkerClient::spawn(1, move || {
        Ok(Box::new(CancellationAwareEngine {
            started: started_sender,
            release: release_receiver,
            callback_result: result_sender,
        }) as Box<dyn ChatEngine>)
    });
    wait_until_ready(&worker).await;
    let stream = worker.submit_stream(empty_engine_request()).unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("stream request started");
    drop(stream);
    release_sender.send(()).unwrap();
    assert!(!result_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("engine observed callback result"));
}

#[tokio::test]
async fn wraps_json_extraction_errors_in_an_openai_error_envelope() {
    let (app, calls) = app().await;
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages": ["#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["param"], "body");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}
