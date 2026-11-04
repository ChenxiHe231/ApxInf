//! Host callback boundaries for one diagnostic generate invocation.
//! Pure std: `rustc --edition 2021 --test this-file.rs` needs no CUDA.
//! The benchmark constructs this ONLY for the measured profiling iteration.

pub const GENERATE: &str = "qwen35.bench.generate_with_prompt_prefill";
pub const PREFILL: &str = "qwen35.bench.prefill_entry_to_first_token_callback";
pub const DECODE: &str = "qwen35.bench.decode_first_to_last_token_callback";
pub const FINALIZE: &str = "qwen35.bench.finalize_last_token_callback_to_return";

pub trait RangeSink {
    type Guard;
    fn push(&mut self, label: &'static str) -> Self::Guard;
}

pub struct GenerationRanges<S: RangeSink> {
    sink: S,
    phase: Option<S::Guard>,
    generation: Option<S::Guard>,
    budget: usize,
    callbacks: usize,
    overflow: bool,
}

impl<S: RangeSink> GenerationRanges<S> {
    pub fn new(mut sink: S, budget: usize) -> Result<Self, &'static str> {
        if budget == 0 {
            return Err("profile callback budget must be positive");
        }
        let generation = Some(sink.push(GENERATE));
        let phase = Some(sink.push(PREFILL));
        Ok(Self { sink, phase, generation, budget, callbacks: 0, overflow: false })
    }

    pub fn on_token(&mut self) {
        let Some(next) = self.callbacks.checked_add(1) else {
            self.overflow = true;
            return;
        };
        self.callbacks = next;
        if self.callbacks > self.budget {
            self.overflow = true;
            return;
        }
        if self.callbacks == 1 {
            drop(self.phase.take());
            self.phase = Some(self.sink.push(if self.budget == 1 { FINALIZE } else { DECODE }));
        } else if self.callbacks == self.budget {
            drop(self.phase.take());
            self.phase = Some(self.sink.push(FINALIZE));
        }
    }

    pub fn finish(mut self, returned_tokens: usize) -> Result<(), String> {
        // End host ranges immediately after generate returns, before validation.
        drop(self.phase.take());
        drop(self.generation.take());
        if self.overflow || self.callbacks != self.budget || returned_tokens != self.callbacks {
            return Err(format!(
                "profile callback boundary mismatch: callbacks={}, expected={}, returned={}, overflow={}",
                self.callbacks, self.budget, returned_tokens, self.overflow,
            ));
        }
        Ok(())
    }
}

impl<S: RangeSink> Drop for GenerationRanges<S> {
    fn drop(&mut self) {
        // Also close in correct nesting order on generation error/early return.
        drop(self.phase.take());
        drop(self.generation.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    #[derive(Default)]
    struct State {
        stack: Vec<&'static str>,
        events: Vec<(bool, &'static str)>,
    }
    #[derive(Clone)]
    struct MockSink(Rc<RefCell<State>>);
    struct MockGuard(Rc<RefCell<State>>, &'static str);
    impl RangeSink for MockSink {
        type Guard = MockGuard;
        fn push(&mut self, name: &'static str) -> MockGuard {
            let mut state = self.0.borrow_mut();
            state.stack.push(name);
            state.events.push((true, name));
            MockGuard(Rc::clone(&self.0), name)
        }
    }
    impl Drop for MockGuard {
        fn drop(&mut self) {
            let mut state = self.0.borrow_mut();
            assert_eq!(state.stack.pop(), Some(self.1), "NVTX stack order changed");
            state.events.push((false, self.1));
        }
    }
    fn sink() -> (MockSink, Rc<RefCell<State>>) {
        let state = Rc::new(RefCell::new(State::default()));
        (MockSink(Rc::clone(&state)), state)
    }

    #[test]
    fn one_token_has_no_decode_range() {
        let (sink, state) = sink();
        let mut ranges = GenerationRanges::new(sink, 1).unwrap();
        ranges.on_token();
        ranges.finish(1).unwrap();
        assert!(state.borrow().stack.is_empty());
        assert_eq!(state.borrow().events, vec![(true, GENERATE),(true, PREFILL),
            (false, PREFILL),(true, FINALIZE),(false, FINALIZE),(false, GENERATE)]);
    }

    #[test]
    fn two_through_1024_tokens_change_phase_only_at_endpoints() {
        for count in [2,3,4,1024] {
            let (sink, state) = sink();
            let mut ranges = GenerationRanges::new(sink, count).unwrap();
            for index in 0..count {
                ranges.on_token();
                let current = *state.borrow().stack.last().unwrap();
                assert_eq!(current, if index+1 == count { FINALIZE } else { DECODE });
            }
            ranges.finish(count).unwrap();
            assert!(state.borrow().stack.is_empty());
            assert_eq!(state.borrow().events, vec![(true, GENERATE),(true, PREFILL),
                (false, PREFILL),(true, DECODE),(false, DECODE),(true, FINALIZE),
                (false, FINALIZE),(false, GENERATE)]);
        }
    }

    #[test]
    fn mtp_batched_callbacks_do_not_create_per_token_ranges() {
        let (sink, state) = sink();
        let mut internal = sink.clone();
        let mut ranges = GenerationRanges::new(sink, 8).unwrap();
        ranges.on_token();
        for batch in [3,2,2] {
            { let _draft = internal.push("existing_mtp_draft_verify_commit"); }
            for _ in 0..batch { ranges.on_token(); }
        }
        ranges.finish(8).unwrap();
        assert!(state.borrow().stack.is_empty());
        assert_eq!(state.borrow().events.iter().filter(|&&(push,name)| push && name==DECODE).count(),1);
        assert_eq!(state.borrow().events.iter().filter(|&&(push,_)|push).count(),7);
    }

    #[test]
    fn early_error_before_or_after_first_callback_closes_stack() {
        for callbacks in [0,1,2] {
            let (sink, state) = sink();
            { let mut ranges = GenerationRanges::new(sink,4).unwrap();
                for _ in 0..callbacks { ranges.on_token(); }
            }
            assert!(state.borrow().stack.is_empty());
        }
    }

    #[test]
    fn incomplete_overflow_or_returned_count_mismatch_fails_closed() {
        for (budget, callbacks, returned) in [(4,3,3),(4,5,5),(4,4,3),(1,0,0)] {
            let (sink, state) = sink();
            let mut ranges = GenerationRanges::new(sink,budget).unwrap();
            for _ in 0..callbacks { ranges.on_token(); }
            assert!(ranges.finish(returned).is_err());
            assert!(state.borrow().stack.is_empty());
        }
    }

    #[test]
    fn zero_budget_does_not_push_any_range() {
        let (sink, state) = sink();
        assert!(GenerationRanges::new(sink,0).is_err());
        assert!(state.borrow().events.is_empty());
    }
}
