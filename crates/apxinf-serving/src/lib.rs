//! OpenAI-compatible, batch-one serving for ApxInf text models.
//!
//! The HTTP runtime never owns a model. A bounded queue feeds one dedicated
//! worker thread, and that thread constructs and retains the mutable model and
//! device context for their complete lifetime.

mod api;
mod engine;
mod error;
pub mod protocol;
pub mod tool_calling;
mod worker;

pub use api::{router, serve, serve_blocking, ApiConfig, ServeConfig};
pub use engine::{ApxInfEngineConfig, ChatEngine, EngineOutput, EngineRequest};
pub use error::ApiError;
pub use worker::{EngineStreamEvent, WorkerClient};
