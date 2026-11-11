#![cfg(feature = "qwen35-cuda")]
//! Real-checkpoint GPU protocol gates, not raw private-bank storage inspection.
//! Explicitly ignored; weights are loaded once per selected test. No tolerance
//! or repeat waiver. Run serially under the shared /tmp/apxinf-gpu.lock.

use std::{cell::RefCell, path::PathBuf, rc::Rc};
use apxinf_core::{Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext};
use apxinf_model::GenerationOptions;
use apxinf_model::qwen35moe::{
    Qwen35ExecutionOptions, Qwen35Fp8AttentionBackend, Qwen35KvCacheDType,
    Qwen35MoeActivation, Qwen35MoeConfig, Qwen35MtpCoordinator, Qwen35MtpRuntime,
    Qwen35MtpState, MtpPositionContract, PreparedMtpCommit, PreparedMtpPrefill,
};
use apxinf_model::qwen35moe::decoder::Qwen35Decoder;
use apxinf_model::qwen35moe::device_weights::{
    load_qwen35_device_weights_from_path_with_options, Qwen35DeviceWeights,
};
use apxinf_model::qwen35moe::mtp_schedule::MtpCommit;
use apxinf_model::qwen35moe::session::{MtpBackend, MtpDraft, Qwen35Session};

type Target = Qwen35Session<Qwen35Fp8AttentionBackend>;
type GateResult = std::result::Result<(), Box<dyn std::error::Error>>;
const REPEATS: usize = 20;

fn bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    let cpu = transfers::to_cpu(tensor)?;
    if cpu.dtype() == apxinf_core::DType::BF16 {
        assert!(cpu.as_bf16()?.iter().all(|x| x.to_f32().is_finite()), "nonfinite BF16 state observation");
    }
    Ok(cpu.storage().as_cpu().expect("to_cpu storage").to_vec())
}

fn digest(data: &[u8]) -> String {
    format!("{:016x}", data.iter().fold(0xcbf29ce484222325u64,
        |h, b| (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)))
}

fn exact(label: &str, actual: &[u8], expected: &[u8]) {
    assert_eq!(actual.len(), expected.len(), "{label}: byte length");
    if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
        panic!("{label}: exact mismatch first_byte={index} actual={} expected={} actual_digest={} expected_digest={}",
            actual[index], expected[index], digest(actual), digest(expected));
    }
}

fn parse_protocol_graphs(value: Option<&std::ffi::OsStr>) -> std::result::Result<bool, &'static str> {
    match value {
        None => Ok(false),
        Some(value) => match value.to_str() {
            Some("0") => Ok(false),
            Some("1") => Ok(true),
            _ => Err("APXINF_QWEN35_PROTOCOL_GRAPHS must be unset, 0 or 1"),
        },
    }
}

struct Fixture {
    ctx: CudaContext,
    config: Qwen35MoeConfig,
    decoder: Qwen35Decoder,
    weights: Qwen35DeviceWeights,
    prompt: Vec<u32>,
    capacity: usize,
}

impl Fixture {
    fn load() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        // These process-global controls are deliberate test admission, not
        // production defaults. Never run this test binary concurrently.
        let protocol_graphs_env = std::env::var_os("APXINF_QWEN35_PROTOCOL_GRAPHS");
        let graphs = parse_protocol_graphs(protocol_graphs_env.as_deref())?;
        std::env::set_var("APXINF_REQUIRE_RECIPES", "1");
        std::env::set_var("APXINF_QWEN35_CUDA_GRAPHS", if graphs { "1" } else { "0" });
        std::env::set_var("APXINF_QWEN35_VERIFY_GRAPHS", "0");
        let checkpoint = PathBuf::from(std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .expect("set absolute APXINF_QWEN35_CHECKPOINT"));
        let tuning = PathBuf::from(std::env::var_os("APXINF_QWEN35_PARITY_TUNING_DIR")
            .expect("set absolute pre-tuned APXINF_QWEN35_PARITY_TUNING_DIR"));
        assert!(checkpoint.is_absolute() && tuning.is_absolute());
        assert!(std::fs::read_dir(&tuning)?.filter_map(std::result::Result::ok)
            .any(|x| x.path().extension().is_some_and(|ext| ext == "recipe")), "autotune separately first");
        let device = std::env::var("APXINF_CUDA_DEVICE").ok().map(|x| x.parse::<usize>().unwrap()).unwrap_or(0);
        let prompt_len = std::env::var("APXINF_QWEN35_PROTOCOL_PROMPT_TOKENS").ok()
            .map(|x| x.parse::<usize>().unwrap()).unwrap_or(16);
        assert!(prompt_len >= 2 && prompt_len <= 49152);
        let pattern = [151644, 8948, 198, 9707, 16357, 220, 19687, 25,
                       198, 151645, 198, 151644, 77091, 198, 35064, 16933];
        let prompt = pattern.into_iter().cycle().take(prompt_len).collect();
        let capacity = prompt_len.checked_add(64).unwrap();
        let ctx = CudaContext::new_with_options(device, false, Some(&tuning))?;
        let config = Qwen35MoeConfig::from_json_file(&checkpoint.join("config.json"))?;
        let execution = Qwen35ExecutionOptions { moe_activation: Qwen35MoeActivation::Nvfp4, kv_cache: Qwen35KvCacheDType::Bf16 };
        let weights = load_qwen35_device_weights_from_path_with_options(&ctx, &config, &checkpoint, execution)?;
        let decoder = Qwen35Decoder::new(&config)?;
        ctx.synchronize()?;
        println!("{}", serde_json::json!({"schema":"apxinf.qwen35.gpu_protocol.config.v1", "checkpoint":checkpoint,
            "tuning_dir":tuning,"autotune":false,"strict_recipe_required":true,"target_kv":"bf16",
            "draft_kv":"fp8","draft_scale":1.0,"moe":"nvfp4","ordered_reduce":true,"graphs":graphs,"graph_gdn":std::env::var("APXINF_QWEN35_GRAPH_GDN").ok(),
            "prompt_tokens":prompt_len,"capacity":capacity,"depths":[1,2,3],"repeats":REPEATS,
            "state_observation":"public logical counts and GPU output/hidden observations; private raw KV/GDN banks not exposed",
            "performance_accepted":false}));
        Ok(Self { ctx, config, decoder, weights, prompt, capacity })
    }

    fn target(&self) -> Result<Target> {
        let attention = Qwen35Fp8AttentionBackend::with_kv_cache(&self.config, Qwen35KvCacheDType::Bf16, None)?;
        Qwen35Session::new(&self.ctx, &self.decoder, &self.weights, attention, self.capacity)
    }

    fn ids(&self, values: &[u32]) -> Result<Tensor> {
        let cpu: Vec<i32> = values.iter().map(|x| i32::try_from(*x).unwrap()).collect();
        transfers::to_cuda_on_stream(&self.ctx, Tensor::from_i32(vec![values.len()], &cpu)?)
    }

    fn scalar(&self, target: &mut Target, prompt: &[u32], count: usize) -> Result<Vec<u32>> {
        target.reset(&self.ctx)?;
        let first = target.prefill(&self.ctx, &self.decoder, &self.weights, prompt)?;
        let mut tokens = first.greedy_token_ids(&self.ctx)?;
        assert_eq!(tokens.len(), 1);
        while tokens.len() < count {
            let output = target.decode(&self.ctx, &self.decoder, &self.weights, *tokens.last().unwrap())?;
            tokens.extend(output.greedy_token_ids(&self.ctx)?);
        }
        assert_eq!(tokens.len(), count);
        Ok(tokens)
    }
}

#[test]
#[ignore = "full real checkpoint GPU gate; fixed20 repeats, pre-tuned strict recipes and shared GPU lock"]
fn real_target_verify_rollback_short_snapshot_protocol() -> GateResult {
    let f = Fixture::load()?;
    let mut candidate = f.target()?;
    let mut scalar = f.target()?;
    let mut snapshot = candidate.allocate_snapshot(&f.ctx, &f.decoder, &f.weights, f.prompt.len())?;
    let verified_ids = [42u32, 1729, 19, 777];
    let different_suffix = [1234u32, 5678];
    for depth in 1..=3usize {
        let device_ids = f.ids(&verified_ids[..depth+1])?;
        for keep in 1..=depth+1 {
            let mut first_signature: Option<(Vec<u32>, Vec<u8>)> = None;
            for repeat in 0..REPEATS {
                candidate.reset(&f.ctx)?; scalar.reset(&f.ctx)?;
                candidate.prefill(&f.ctx, &f.decoder, &f.weights, &f.prompt)?;
                scalar.prefill(&f.ctx, &f.decoder, &f.weights, &f.prompt)?;
                candidate.capture_snapshot(&f.ctx, &mut snapshot)?;
                let start = candidate.position(); let epoch = candidate.epoch();
                // Dropping real GPU verification must not publish its shadow.
                let aborted = candidate.verify_target_chunk_device(&f.ctx, &f.decoder, &f.weights, &device_ids)?;
                f.ctx.synchronize()?; drop(aborted);
                assert_eq!((candidate.position(), candidate.epoch()), (start, epoch));
                let a = candidate.decode(&f.ctx, &f.decoder, &f.weights, 991)?;
                let b = scalar.decode(&f.ctx, &f.decoder, &f.weights, 991)?;
                exact("aborted verify -> same-input scalar logits", &bytes(&a.logits)?, &bytes(&b.logits)?);
                exact("aborted verify -> same-input scalar hidden", &bytes(&a.hidden_states)?, &bytes(&b.hidden_states)?);
                // Restore a shorter prefix after a longer committed decode.
                candidate.restore_snapshot(&f.ctx, &snapshot)?;
                assert_eq!(candidate.position(), f.prompt.len());
                scalar.reset(&f.ctx)?; scalar.prefill(&f.ctx, &f.decoder, &f.weights, &f.prompt)?;
                let stale = candidate.verify_target_chunk_device(&f.ctx, &f.decoder, &f.weights, &device_ids)?;
                let latest = candidate.verify_target_chunk_device(&f.ctx, &f.decoder, &f.weights, &device_ids)?;
                let epoch = candidate.epoch();
                assert!(candidate.prepare_target_commit(&f.ctx, &f.decoder, &f.weights, stale, keep).is_err(), "stale GPU verifier must be rejected");
                drop(latest);
                assert_eq!((candidate.position(), candidate.epoch()), (start, epoch));
                let verified = candidate.verify_target_chunk_device(&f.ctx, &f.decoder, &f.weights, &device_ids)?;
                let prepared = candidate.prepare_target_commit(&f.ctx, &f.decoder, &f.weights, verified, keep)?;
                assert_eq!((candidate.position(), candidate.epoch()), (start, epoch), "prepare must not publish");
                let committed = candidate.publish_target_commit(prepared)?;
                assert_eq!((candidate.position(), candidate.epoch()), (start+keep, epoch+1));
                assert_eq!((committed.start_position, committed.end_position), (start+keep-1, start+keep));
                let mut reference = None;
                for token in &verified_ids[..keep] {
                    reference = Some(scalar.decode(&f.ctx, &f.decoder, &f.weights, *token)?);
                }
                // Across batched-vs-scalar arithmetic, the exact gate is token
                // equality. Byte equality is separately used for same-path
                // replay and abort/restore observations, not conflated here.
                let mut signature = committed.greedy_token_ids(&f.ctx)?;
                assert_eq!(signature, reference.unwrap().greedy_token_ids(&f.ctx)?, "depth={depth} keep={keep} repeat={repeat}");
                for token in [998u32, 654] {
                    let a = candidate.decode(&f.ctx, &f.decoder, &f.weights, token)?;
                    let b = scalar.decode(&f.ctx, &f.decoder, &f.weights, token)?;
                    let prediction = a.greedy_token_ids(&f.ctx)?;
                    assert_eq!(prediction, b.greedy_token_ids(&f.ctx)?, "discarded KV/GDN rows must remain unreachable");
                    signature.extend(prediction);
                }
                candidate.restore_snapshot(&f.ctx, &snapshot)?;
                assert_eq!(candidate.position(), f.prompt.len());
                scalar.reset(&f.ctx)?; scalar.prefill(&f.ctx, &f.decoder, &f.weights, &f.prompt)?;
                let a = candidate.append_prefill(&f.ctx, &f.decoder, &f.weights, &different_suffix)?;
                let b = scalar.append_prefill(&f.ctx, &f.decoder, &f.weights, &different_suffix)?;
                assert_eq!(candidate.position(), f.prompt.len()+different_suffix.len());
                exact("short restore + different suffix logits", &bytes(&a.logits)?, &bytes(&b.logits)?);
                let hidden = bytes(&a.hidden_states)?;
                exact("short restore + different suffix hidden", &hidden, &bytes(&b.hidden_states)?);
                if let Some((tokens, raw)) = &first_signature {
                    assert_eq!(&signature, tokens, "same protocol repeated token drift");
                    exact("same protocol repeated hidden", &hidden, raw);
                } else { first_signature = Some((signature.clone(), hidden.clone())); }
                println!("{}", serde_json::json!({"schema":"apxinf.qwen35.target_protocol.v1", "depth":depth,
                    "verified_rows":depth+1,"committed_rows":keep,"discarded_rows":depth+1-keep,"repeat":repeat,
                    "restored_position":f.prompt.len(),"different_suffix":different_suffix,"tokens":signature,
                    "restored_suffix_hidden_digest":digest(&hidden),"private_raw_bank_equality_checked":false,"performance_accepted":false}));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Audit {
    draft_hidden: Vec<u8>,
    append_previous_hidden: Vec<u8>,
    committed: Option<MtpPositionContract>,
}

// Test-local transparent delegation observes already-public MtpBackend inputs.
// No fake proposals, weights, acceptance or cache mutation are injected.
struct ObservedMtp { inner: Qwen35MtpRuntime, audit: Rc<RefCell<Audit>> }
impl MtpBackend for ObservedMtp {
    type State = Qwen35MtpState;
    type PreparedPrefill = PreparedMtpPrefill;
    type PreparedCommit = PreparedMtpCommit;
    fn allocate_state(&mut self, ctx: &CudaContext, capacity: usize) -> Result<Self::State> { self.inner.allocate_state(ctx, capacity) }
    fn reset_state(&mut self, ctx: &CudaContext, state: &mut Self::State) -> Result<()> { self.inner.reset_state(ctx, state) }
    fn copy_state(&mut self, ctx: &CudaContext, source: &Self::State, destination: &mut Self::State) -> Result<()> {
        self.inner.copy_state(ctx, source, destination)?;
        assert_eq!(source.position(), destination.position(), "draft snapshot copy logical/physical positions");
        Ok(())
    }
    fn prepare_prefill(&mut self, ctx: &CudaContext, weights: &Qwen35DeviceWeights, shadow: &mut Self::State, prompt_tokens: &[u32], target_hidden: &Tensor) -> Result<Self::PreparedPrefill> {
        self.inner.prepare_prefill(ctx, weights, shadow, prompt_tokens, target_hidden)
    }
    fn prepare_append_prefill(&mut self, ctx: &CudaContext, weights: &Qwen35DeviceWeights, committed: &Self::State, shadow: &mut Self::State, chunk_tokens: &[u32], previous_target_hidden: &Tensor, target_hidden: &Tensor) -> Result<Self::PreparedPrefill> {
        self.audit.borrow_mut().append_previous_hidden = bytes(previous_target_hidden)?;
        self.inner.prepare_append_prefill(ctx, weights, committed, shadow, chunk_tokens, previous_target_hidden, target_hidden)
    }
    fn publish_prefill(&mut self, committed: &mut Self::State, shadow: &mut Self::State, prepared: Self::PreparedPrefill) {
        self.inner.publish_prefill(committed, shadow, prepared); self.audit.borrow_mut().committed = committed.position();
    }
    fn draft(&mut self, ctx: &CudaContext, weights: &Qwen35DeviceWeights, committed: &Self::State, shadow: &mut Self::State, pending: u32, start_position: usize, target_hidden: &Tensor, draft_steps: usize) -> Result<MtpDraft> {
        let position = committed.position().expect("MTP prompt committed");
        assert_eq!(position.logical_position, start_position); assert_eq!(position.cache_rows+1, start_position);
        self.audit.borrow_mut().draft_hidden = bytes(target_hidden)?;
        self.inner.draft(ctx, weights, committed, shadow, pending, start_position, target_hidden, draft_steps)
    }
    fn prepare_commit(&mut self, ctx: &CudaContext, weights: &Qwen35DeviceWeights, shadow: &mut Self::State, contract: MtpCommit) -> Result<Self::PreparedCommit> {
        self.inner.prepare_commit(ctx, weights, shadow, contract)
    }
    fn publish_commit(&mut self, committed: &mut Self::State, shadow: &mut Self::State, prepared: Self::PreparedCommit) {
        self.inner.publish_commit(committed, shadow, prepared); self.audit.borrow_mut().committed = committed.position();
    }
}

#[test]
#[ignore = "real target+draft snapshot gate; captures after an MTP round to test hidden-scratch alias risk without assuming it is a bug"]
fn real_mtp_short_snapshot_different_continuation_protocol() -> GateResult {
    let f = Fixture::load()?;
    let audit = Rc::new(RefCell::new(Audit::default()));
    let mtp = ObservedMtp { inner: Qwen35MtpRuntime::new(&f.config)?, audit: Rc::clone(&audit) };
    let mut coordinator = Qwen35MtpCoordinator::new(&f.ctx, f.target()?, mtp)?;
    let mut scalar = f.target()?;
    let mut snapshot = coordinator.allocate_context_snapshot(&f.ctx, &f.decoder, &f.weights, f.capacity)?;
    let options = GenerationOptions::greedy(32, None);
    for depth in 1..=3usize {
        for repeat in 0..REPEATS {
            coordinator.reset(&f.ctx)?;
            let prefill = coordinator.prefill_greedy(&f.ctx, &f.decoder, &f.weights, &f.prompt, &options)?;
            let first = prefill.first_token;
            let round = coordinator.run_greedy_round(&f.ctx, &f.decoder, &f.weights, depth)?.outcome;
            let kept = round.contract.target.kv_range.committed_rows();
            assert_eq!(round.contract.target.kv_range.verified_end-round.contract.target.kv_range.start, depth+1);
            assert!((1..=depth+1).contains(&kept));
            assert_eq!(round.contract.target.gdn_range, round.contract.target.kv_range);
            assert_eq!(round.contract.target.kv_range.start, f.prompt.len());
            assert_eq!(round.contract.target.kv_range.committed_end, f.prompt.len()+kept);
            // CommitContract ranges are logical; only MtpPositionContract
            // cache_rows is physical P-1. Do not conflate those counters.
            assert_eq!(round.contract.mtp.cache_range.start, f.prompt.len());
            assert_eq!(round.contract.mtp.cache_range.committed_end, f.prompt.len()+kept);
            assert_eq!(round.emitted.len(), kept);
            let stats = coordinator.schedule().unwrap().stats();
            assert_eq!(stats.rounds, 1); assert_eq!(stats.proposed_tokens, depth);
            assert_eq!(stats.target_verified_rows, depth+1);
            assert_eq!(coordinator.target().position(), f.prompt.len()+kept);
            let mtp_position = audit.borrow().committed.unwrap();
            assert_eq!(mtp_position.logical_position, coordinator.target().position());
            assert_eq!(mtp_position.cache_rows+1, mtp_position.logical_position);
            let mut prefix = f.prompt.clone(); prefix.push(first);
            prefix.extend_from_slice(&round.emitted[..round.emitted.len()-1]);
            assert_eq!(prefix.len(), coordinator.target().position());
            coordinator.capture_context(&f.ctx, &mut snapshot)?;
            // The next draft receives capture-time target_hidden. Save CPU
            // bytes before this round reuses its verifier scratch profile.
            coordinator.run_greedy_round(&f.ctx, &f.decoder, &f.weights, depth)?;
            let saved_hidden = audit.borrow().draft_hidden.clone();
            assert!(!saved_hidden.is_empty());
            let mut branch_a = None;
            for suffix in [[1234u32, 5678], [3456u32, 7890], [1234u32, 5678]] {
                coordinator.restore_context(&f.ctx, &snapshot)?;
                assert_eq!(coordinator.target().position(), prefix.len());
                assert!(coordinator.schedule().is_none(), "restore clears old schedule/pending");
                let start = coordinator.append_prefill_greedy(&f.ctx, &f.decoder, &f.weights, &suffix, &options)?;
                exact("snapshot target_hidden must survive later verifier reuse", &audit.borrow().append_previous_hidden, &saved_hidden);
                let outcome = coordinator.run_greedy_round(&f.ctx, &f.decoder, &f.weights, depth)?.outcome;
                let mut tokens = vec![start.first_token]; tokens.extend_from_slice(&outcome.emitted);
                let mut full_prompt = prefix.clone(); full_prompt.extend_from_slice(&suffix);
                assert_eq!(tokens, f.scalar(&mut scalar, &full_prompt, tokens.len())?, "depth={depth} repeat={repeat} suffix={suffix:?}");
                let schedule = coordinator.schedule().unwrap();
                assert_eq!(schedule.stats().rounds, 1); assert_eq!(schedule.stats().proposed_tokens, depth);
                assert_eq!(schedule.stats().target_verified_rows, depth+1);
                let position = schedule.positions();
                assert_eq!(position.base, position.mtp); assert_eq!(position.base, coordinator.target().position());
                assert_eq!(outcome.contract.mtp.cache_range.committed_end, position.base);
                let draft = audit.borrow().committed.unwrap();
                assert_eq!(draft.logical_position, position.base); assert_eq!(draft.cache_rows+1, position.base);
                if suffix == [1234, 5678] {
                    if let Some(previous) = &branch_a { assert_eq!(&tokens, previous, "restored branch A must replay exactly"); }
                    else { branch_a = Some(tokens.clone()); }
                }
                println!("{}", serde_json::json!({"schema":"apxinf.qwen35.mtp_snapshot_protocol.v1", "depth":depth,
                    "repeat":repeat,"snapshot_position":prefix.len(),"suffix":suffix,"tokens":tokens,
                    "target_position":position.base,"draft_logical_position":draft.logical_position,"draft_cache_rows":draft.cache_rows,
                    "saved_hidden_digest":digest(&saved_hidden),"restored_hidden_digest":digest(&audit.borrow().append_previous_hidden),
                    "accepted_proposals":outcome.accepted_proposals,"private_raw_bank_equality_checked":false,"performance_accepted":false}));
            }
        }
    }
    Ok(())
}
