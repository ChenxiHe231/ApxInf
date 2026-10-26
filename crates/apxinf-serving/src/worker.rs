use std::sync::{
    atomic::{AtomicU8, Ordering},
    mpsc::{sync_channel, SyncSender as JobSender, TrySendError},
    Arc, Mutex,
};

use tokio::sync::{mpsc, oneshot};

use crate::engine::{ChatEngine, EngineOutput, EngineRequest};
use crate::error::ApiError;

const LOADING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;

enum JobResponse {
    Complete(oneshot::Sender<Result<EngineOutput, ApiError>>),
    Stream(mpsc::Sender<Result<EngineStreamEvent, ApiError>>),
}

impl JobResponse {
    fn is_closed(&self) -> bool {
        match self {
            Self::Complete(sender) => sender.is_closed(),
            Self::Stream(sender) => sender.is_closed(),
        }
    }
}

struct Job {
    request: EngineRequest,
    response: JobResponse,
}

#[derive(Clone, Debug)]
pub enum EngineStreamEvent {
    TextDelta(String),
    Complete(EngineOutput),
}

struct WorkerStatus {
    state: AtomicU8,
    error: Mutex<Option<String>>,
}

#[derive(Clone)]
pub struct WorkerClient {
    sender: JobSender<Job>,
    status: Arc<WorkerStatus>,
}

impl WorkerClient {
    /// Start a batch-one worker. `factory` runs inside the worker thread so a
    /// thread-bound CUDA/model object is never constructed and then moved.
    pub fn spawn<F>(queue_capacity: usize, factory: F) -> Self
    where
        F: FnOnce() -> Result<Box<dyn ChatEngine>, ApiError> + Send + 'static,
    {
        let (sender, receiver) = sync_channel::<Job>(queue_capacity.max(1));
        let status = Arc::new(WorkerStatus {
            state: AtomicU8::new(LOADING),
            error: Mutex::new(None),
        });
        let thread_status = status.clone();
        std::thread::Builder::new()
            .name("apxinf-inference".into())
            .spawn(move || {
                let mut engine = match factory() {
                    Ok(engine) => {
                        thread_status.state.store(READY, Ordering::Release);
                        engine
                    }
                    Err(error) => {
                        *thread_status.error.lock().expect("worker error mutex") =
                            Some(error.message);
                        thread_status.state.store(FAILED, Ordering::Release);
                        return;
                    }
                };
                while let Ok(job) = receiver.recv() {
                    // A queued HTTP request may have been canceled before it
                    // reached the batch-one worker. Do not spend an inference
                    // slot on work whose receiver no longer exists.
                    if job.response.is_closed() {
                        continue;
                    }
                    match job.response {
                        JobResponse::Complete(response) => {
                            let result = engine.complete(job.request);
                            let _ = response.send(result);
                        }
                        JobResponse::Stream(response) => {
                            let mut connected = true;
                            let result = engine.complete_stream(job.request, &mut |delta| {
                                if !connected {
                                    return false;
                                }
                                connected = response
                                    .blocking_send(Ok(EngineStreamEvent::TextDelta(
                                        delta.to_owned(),
                                    )))
                                    .is_ok();
                                connected
                            });
                            if connected {
                                let terminal = result.map(EngineStreamEvent::Complete);
                                let _ = response.blocking_send(terminal);
                            }
                        }
                    }
                }
            })
            .expect("create ApxInf inference worker thread");
        Self { sender, status }
    }

    pub fn ready(&self) -> bool {
        self.status.state.load(Ordering::Acquire) == READY
    }

    pub fn status_message(&self) -> &'static str {
        match self.status.state.load(Ordering::Acquire) {
            LOADING => "loading",
            READY => "ready",
            FAILED => "failed",
            _ => "unknown",
        }
    }

    pub fn startup_error(&self) -> Option<String> {
        self.status
            .error
            .lock()
            .ok()
            .and_then(|error| error.clone())
    }

    pub fn submit(
        &self,
        request: EngineRequest,
    ) -> Result<oneshot::Receiver<Result<EngineOutput, ApiError>>, ApiError> {
        match self.status.state.load(Ordering::Acquire) {
            LOADING => return Err(ApiError::unavailable("model is still loading")),
            FAILED => {
                return Err(ApiError::unavailable(
                    self.startup_error()
                        .unwrap_or_else(|| "model worker failed to start".into()),
                ))
            }
            READY => {}
            _ => return Err(ApiError::unavailable("model worker is unavailable")),
        }
        let (response, receiver) = oneshot::channel();
        match self.sender.try_send(Job {
            request,
            response: JobResponse::Complete(response),
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(ApiError::queue_full()),
            Err(TrySendError::Disconnected(_)) => {
                return Err(ApiError::unavailable("model worker stopped"))
            }
        }
        Ok(receiver)
    }

    pub fn submit_stream(
        &self,
        request: EngineRequest,
    ) -> Result<mpsc::Receiver<Result<EngineStreamEvent, ApiError>>, ApiError> {
        match self.status.state.load(Ordering::Acquire) {
            LOADING => return Err(ApiError::unavailable("model is still loading")),
            FAILED => {
                return Err(ApiError::unavailable(
                    self.startup_error()
                        .unwrap_or_else(|| "model worker failed to start".into()),
                ))
            }
            READY => {}
            _ => return Err(ApiError::unavailable("model worker is unavailable")),
        }
        // This is separate from the inference queue capacity. It bounds
        // decoded text waiting for the HTTP task and applies backpressure to
        // the generation callback when a client reads slowly.
        let (response, receiver) = mpsc::channel(8);
        match self.sender.try_send(Job {
            request,
            response: JobResponse::Stream(response),
        }) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => return Err(ApiError::queue_full()),
            Err(TrySendError::Disconnected(_)) => {
                return Err(ApiError::unavailable("model worker stopped"))
            }
        }
        Ok(receiver)
    }

    pub async fn complete(&self, request: EngineRequest) -> Result<EngineOutput, ApiError> {
        let receiver = self.submit(request)?;
        receiver
            .await
            .map_err(|_| ApiError::unavailable("model worker stopped before responding"))?
    }
}
