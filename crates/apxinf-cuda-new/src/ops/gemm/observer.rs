use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{DType, Error, Result, Tensor};

use super::GemmArgs;

/// Optional thread-local instrumentation for calibration of BF16 GEMM inputs.
pub trait Bf16ActivationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()>;
}

thread_local! {
    static BF16_OBSERVER: RefCell<Option<Rc<dyn Bf16ActivationObserver>>> = RefCell::new(None);
}

pub struct Bf16ObserverGuard;

impl Drop for Bf16ObserverGuard {
    fn drop(&mut self) {
        BF16_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

pub fn install_bf16_observer(
    observer: Rc<dyn Bf16ActivationObserver>,
) -> Result<Bf16ObserverGuard> {
    BF16_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::Other(
                "a BF16 activation observer is already installed".into(),
            ));
        }
        *slot = Some(observer);
        Ok(Bf16ObserverGuard)
    })
}

pub(super) fn observe(args: &GemmArgs<'_>) -> Result<()> {
    if args.a.dtype() != DType::BF16 || args.b.dtype() != DType::BF16 {
        return Ok(());
    }
    BF16_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe(args.a, args.b)?;
        }
        Ok(())
    })
}
