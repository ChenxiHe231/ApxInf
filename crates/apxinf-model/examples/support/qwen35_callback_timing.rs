//! One monotonic host clock for TTFT, all callback intervals, and post-token tail.
//! No per-token timestamps: entry, first callback, final callback, return only.
//! Pure std; independent CPU contract tests require neither Cargo nor CUDA.
use std::time::{Duration, Instant};

pub const METHOD: &str = "host-callback-first-last-v2";
pub const CLOCK: &str = "std::time::Instant";
pub const ORIGIN: &str = "benchmark-generate-entry";

pub trait Clock {
    type Mark: Copy;
    fn now(&mut self) -> Self::Mark;
    fn elapsed(start: Self::Mark, end: Self::Mark) -> Option<Duration>;
}
pub struct HostClock;
impl Clock for HostClock {
    type Mark = Instant;
    fn now(&mut self) -> Instant { Instant::now() }
    fn elapsed(start: Instant, end: Instant) -> Option<Duration> { end.checked_duration_since(start) }
}

#[derive(Debug)]
pub struct Measurement {
    pub callback_count: usize,
    pub intervals: usize,
    pub clock_reads: usize,
    pub first_callback: Duration,
    pub last_callback: Duration,
    pub generate_return: Duration,
    pub interval: Duration,
    pub post_last_callback: Duration,
}
impl Measurement {
    pub fn decode_tps(&self) -> Option<f64> {
        (self.intervals > 0).then(|| self.intervals as f64 / self.interval.as_secs_f64())
    }
    pub fn decode_ms(&self) -> Option<f64> {
        (self.intervals > 0).then(|| ms(self.interval))
    }
}
pub fn ms(duration: Duration) -> f64 { duration.as_secs_f64() * 1000.0 }

pub struct CallbackTimer<C: Clock = HostClock> {
    clock: C,
    budget: usize,
    callbacks: usize,
    overflow: bool,
    entry: C::Mark,
    first: Option<C::Mark>,
    last: Option<C::Mark>,
}
impl CallbackTimer<HostClock> {
    pub fn start(budget: usize) -> Result<Self, &'static str> { Self::with_clock(budget, HostClock) }
}
impl<C: Clock> CallbackTimer<C> {
    pub fn with_clock(budget: usize, mut clock: C) -> Result<Self, &'static str> {
        if budget == 0 { return Err("callback timing requires a positive output budget"); }
        let entry = clock.now();
        Ok(Self { clock,budget,callbacks:0,overflow:false,entry,first:None,last:None })
    }
    pub fn on_token(&mut self) {
        let Some(next) = self.callbacks.checked_add(1) else { self.overflow=true; return; };
        self.callbacks=next;
        if next > self.budget { self.overflow=true; return; }
        if next == 1 {
            let first=self.clock.now();
            self.first=Some(first);
            if self.budget == 1 { self.last=Some(first); }
        } else if next == self.budget {
            self.last=Some(self.clock.now());
        }
    }
    pub fn finish_after_generate(mut self, returned_tokens: usize) -> Result<Measurement,String> {
        // First action after generate: take return endpoint BEFORE validation.
        let returned=self.clock.now();
        if self.overflow || self.callbacks != self.budget || returned_tokens != self.callbacks {
            return Err(format!("callback timing output mismatch: callbacks={}, budget={}, returned={}, overflow={}",
                self.callbacks,self.budget,returned_tokens,self.overflow));
        }
        let first=self.first.ok_or("first callback clock missing")?;
        let last=self.last.ok_or("last callback clock missing")?;
        let first_callback=C::elapsed(self.entry,first).ok_or("first callback before entry")?;
        let last_callback=C::elapsed(self.entry,last).ok_or("last callback before entry")?;
        let generate_return=C::elapsed(self.entry,returned).ok_or("generate return before entry")?;
        let interval=C::elapsed(first,last).ok_or("last callback before first")?;
        let post_last_callback=C::elapsed(last,returned).ok_or("generate return before last callback")?;
        let intervals=self.callbacks-1;
        if intervals > 0 && interval.is_zero() { return Err("positive interval count has zero elapsed time".into()); }
        if first_callback.checked_add(interval).and_then(|d|d.checked_add(post_last_callback)) != Some(generate_return) {
            return Err("callback clock decomposition overflow/mismatch".into());
        }
        Ok(Measurement { callback_count:self.callbacks, intervals,
            clock_reads:if self.budget>1 {4} else {3}, first_callback,last_callback,
            generate_return,interval,post_last_callback })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell,collections::VecDeque,rc::Rc};
    struct FakeClock(Rc<RefCell<(VecDeque<u64>,usize)>>);
    impl Clock for FakeClock {
        type Mark=u64;
        fn now(&mut self)->u64 { let mut x=self.0.borrow_mut(); x.1+=1; x.0.pop_front().expect("unexpected clock read") }
        fn elapsed(start:u64,end:u64)->Option<Duration> { end.checked_sub(start).map(Duration::from_nanos) }
    }
    fn timer(budget:usize, times:&[u64])->(CallbackTimer<FakeClock>,Rc<RefCell<(VecDeque<u64>,usize)>>) {
        let state=Rc::new(RefCell::new((times.iter().copied().collect(),0)));
        (CallbackTimer::with_clock(budget,FakeClock(Rc::clone(&state))).unwrap(),state)
    }
    #[test]
    fn exact_1023_intervals_one_clock_and_tail_decompose_total() {
        let (mut t,state)=timer(1024,&[100,200,2_000_200,2_000_300]);
        for _ in 0..1024 { t.on_token(); }
        let m=t.finish_after_generate(1024).unwrap();
        assert_eq!(m.callback_count,1024);assert_eq!(m.intervals,1023);
        assert_eq!(m.first_callback,Duration::from_nanos(100));
        assert_eq!(m.interval,Duration::from_nanos(2_000_000));
        assert_eq!(m.post_last_callback,Duration::from_nanos(100));
        assert_eq!(m.generate_return,Duration::from_nanos(2_000_200));
        assert_eq!(m.decode_tps(),Some(511_500.0));assert_eq!(m.clock_reads,4);
        assert_eq!(state.borrow().1,4);
    }
    #[test]
    fn no_clock_read_on_intermediate_callbacks_even_when_mtp_emits_batches() {
        let (mut t,state)=timer(1024,&[0,10,1_000_010,1_000_100]);
        t.on_token(); assert_eq!(state.borrow().1,2);
        // 341 complete three-token batches after the prefill token. No
        // individual-token timestamps or uniformly distributed intervals exist.
        for batch in 0..341 {
            for slot in 0..3 {
                t.on_token();
                assert_eq!(state.borrow().1,if batch==340 && slot==2 {3} else {2});
            }
        }
        let m=t.finish_after_generate(1024).unwrap();
        assert_eq!(m.interval,Duration::from_nanos(1_000_000));
        assert_eq!(state.borrow().1,4);assert!(state.borrow().0.is_empty());
    }
    #[test]
    fn one_token_has_no_decode_rate_and_only_three_clock_reads() {
        let (mut t,state)=timer(1,&[0,10,25]);t.on_token();
        let m=t.finish_after_generate(1).unwrap();assert_eq!(m.intervals,0);
        assert_eq!(m.decode_tps(),None);assert_eq!(m.decode_ms(),None);
        assert_eq!(m.interval,Duration::ZERO);assert_eq!(m.post_last_callback,Duration::from_nanos(15));
        assert_eq!(m.clock_reads,3);assert_eq!(state.borrow().1,3);
    }
    #[test]
    fn all_small_budgets_preserve_endpoint_counts() {
        for budget in [2,3,4] { let (mut t,state)=timer(budget,&[0,10,40,50]);
            for _ in 0..budget { t.on_token(); }
            let m=t.finish_after_generate(budget).unwrap();assert_eq!(m.intervals,budget-1);
            assert_eq!(state.borrow().1,4);
        }
    }
    #[test]
    fn missing_callback_or_eos_short_output_is_rejected() {
        let (mut t,_)=timer(1024,&[0,10,99]);
        for _ in 0..1023 {t.on_token();}
        assert!(t.finish_after_generate(1023).unwrap_err().contains("output mismatch"));
    }
    #[test]
    fn extra_callback_or_return_count_mismatch_is_rejected() {
        for (callbacks,returned) in [(5,5),(4,3)] {
            let (mut t,_)=timer(4,&[0,10,20,30]);
            for _ in 0..callbacks {t.on_token();}
            assert!(t.finish_after_generate(returned).is_err());
        }
    }
    #[test]
    fn nonmonotonic_or_zero_interval_clocks_fail() {
        for times in [[20,10,30,40],[0,20,10,40],[0,10,30,20],[0,10,10,20]] {
            let (mut t,_)=timer(2,&times);t.on_token();t.on_token();
            assert!(t.finish_after_generate(2).is_err());
        }
    }
    #[test]
    fn zero_budget_is_rejected_without_reading_clock() {
        let state=Rc::new(RefCell::new((VecDeque::new(),0)));
        assert!(CallbackTimer::with_clock(0,FakeClock(Rc::clone(&state))).is_err());
        assert_eq!(state.borrow().1,0);
    }
    #[test]
    fn legacy_engine_minus_callback_interval_is_not_only_finalize() {
        // Independent literal host timeline: first ready=100; callbacks at
        // 110 and 900; finalize=1000. Engine excludes callback durations 15+20.
        let legacy_engine_decode=1000u64-100-35;
        let callback_interval=900u64-110;
        let callback_tail=1000u64-900;
        assert_eq!(legacy_engine_decode,865);assert_eq!(callback_interval,790);
        assert_eq!(legacy_engine_decode-callback_interval,75);
        assert_ne!(legacy_engine_decode-callback_interval,callback_tail);
    }
}
