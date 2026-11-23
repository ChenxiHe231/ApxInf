//! Scoped, same-capture DAG fork. Dependency handles never cross this module's ABI.
use crate::{workspace, CudaContext};
use apxinf_core::{Error, Result};
use std::{ffi::c_void, marker::PhantomData, rc::Rc};

use crate::ffi::abi::graph_capture::*;

fn checked(code: i32, stage: &str) -> Result<()> {
    if code == 0 { return Ok(()); }
    let message = format!("capture independent branches {stage}: CUDA status {code}");
    workspace::poison_capture(&message);
    Err(Error::Other(message))
}
struct Scope {
    raw: *mut c_void,
    joined: bool,
    _thread_confined: PhantomData<Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        if !self.joined { workspace::poison_capture("unfinished independent capture branch scope"); }
        let code = unsafe { apxinf_capture_frontier_destroy(self.raw) };
        if code != 0 {
            let message=format!("capture fork cleanup failed: CUDA status {code}");
            workspace::poison_capture(&message);
            eprintln!("{message}");
        }
    }
}

/// Capture two independent GPU branches in one graph, keeping host operation order.
///
/// Returns false without executing either closure outside capture or when disabled;
/// the caller then executes its original serial path. Returns true after a joined
/// capture. This keeps preparation/eager operator traversals exactly unchanged.
/// Caller must supply branches whose writable GPU buffers and provider workspaces
/// are disjoint. Readonly inputs may be shared. This changes execution topology,
/// not tensor arithmetic. Every dependent consumer must be enqueued after return.
/// Errors/panics poison the enclosing capture, even when caught by its caller.
/// The same mutable state argument is borrowed sequentially, avoiding overlapping
/// Rust closure captures. No raw graph/frontier token is public or can escape.
pub fn capture_independent<T>(
    ctx: &CudaContext,
    enabled: bool,
    state: &mut T,
    first: impl FnOnce(&mut T) -> Result<()>,
    second: impl FnOnce(&mut T) -> Result<()>,
) -> Result<bool> {
    if !enabled || !workspace::is_capturing() {
        return Ok(false);
    }
    if let Err(error) = workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize) {
        workspace::poison_capture(&error.to_string());
        return Err(error);
    }
    if let Err(error) = ctx.stream().set_current_device() {
        workspace::poison_capture(&error);
        return Err(Error::Cuda(error));
    }
    if let Some(error)=workspace::capture_failure(){return Err(Error::Other(error));}
    fault(Fault::Begin)?;
    #[cfg(test)]
    if take_fault(Fault::NonDefaultEdge){checked(unsafe{apxinf_capture_frontier_test_inject(std::ptr::null_mut(),0)},"inject edge")?;}
    let mut raw = std::ptr::null_mut();
    checked(unsafe { apxinf_capture_frontier_begin(ctx.stream().handle(), ctx.device_id() as i32, &mut raw) }, "begin")?;
    if raw.is_null() {
        workspace::poison_capture("null independent capture scope");
        return Err(Error::Other("null independent capture scope".into()));
    }
    let mut scope = Scope { raw, joined: false, _thread_confined: PhantomData };
    #[cfg(test)]
    for(stage,kind)in [(Fault::GraphId,1),(Fault::GraphHandle,2)]{if take_fault(stage){checked(unsafe{apxinf_capture_frontier_test_inject(scope.raw,kind)},"inject identity")?;}}
    first(state)?;
    fault(Fault::AfterFirst)?;
    fault(Fault::Rewind)?;
    checked(unsafe { apxinf_capture_frontier_rewind(scope.raw) }, "rewind")?;
    second(state)?;
    fault(Fault::AfterSecond)?;
    fault(Fault::Join)?;
    checked(unsafe { apxinf_capture_frontier_join(scope.raw) }, "join")?;
    scope.joined = true;
    drop(scope);
    // Cleanup failure is poison even if join succeeded.
    if let Some(error) = workspace::capture_failure() { return Err(Error::Other(error)); }
    Ok(true)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault { Begin, AfterFirst, Rewind, AfterSecond, Join, #[cfg(test)] GraphId, #[cfg(test)] GraphHandle, #[cfg(test)] NonDefaultEdge }
#[cfg(test)]
thread_local! { static INJECT: std::cell::Cell<Option<Fault>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
fn take_fault(stage:Fault)->bool {INJECT.with(|f| {if f.get()==Some(stage) { f.set(None); true } else {false}})}
fn fault(stage: Fault) -> Result<()> {
    #[cfg(test)]
    if take_fault(stage) {
        workspace::poison_capture("injected independent capture failure");
        return Err(Error::Other("injected independent capture failure".into()));
    }
    let _ = stage;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{capture, ops, transfers, CudaBuffer, ExecutionSession};
    use apxinf_core::{DType, Shape, Tensor};
    use half::bf16;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    fn read(t:&Tensor)->Vec<u8> {
        let b=CudaBuffer::from_tensor(t).unwrap();let mut bytes=vec![0;b.len()];b.copy_to_host(&mut bytes).unwrap();bytes
    }
    fn zeros(ctx:&CudaContext)->Tensor {
        CudaBuffer::alloc_zeros_async(128,ctx.device_id(),ctx.stream()).unwrap().as_tensor(Shape::new(vec![1,64]),DType::BF16).unwrap()
    }
    struct Bank {input:Tensor,weight:Tensor,a:Tensor,b:Tensor,out:Tensor,prefix:CudaBuffer}
    fn first(ctx:&CudaContext,b:&mut Bank)->Result<()> {ops::gemm(ctx,ops::GemmArgs::new(&b.input,&b.weight,&mut b.a))}
    fn second(ctx:&CudaContext,b:&mut Bank)->Result<()> {ops::gemm(ctx,ops::GemmArgs::new(&b.input,&b.weight,&mut b.b))}
    fn forward(ctx:&CudaContext,b:&mut Bank,fork:bool)->Result<()> {
        b.prefix.fill_async(0x12,ctx.stream()).map_err(Error::Cuda)?;
        if !capture_independent(ctx,fork,b,|s|first(ctx,s),|s|second(ctx,s))? {first(ctx,b)?;second(ctx,b)?;}
        ops::elementwise::add(ctx,&b.a,&b.b,&b.out)
    }
    #[test]
    fn same_session_fork_exact_topology_resources_and_replay() {
        let _lock=crate::CUDA_TEST_LOCK.lock().unwrap();
        let ctx=CudaContext::new_with_autotune(0,false).unwrap();
        let host=Tensor::from_bf16(vec![1,64],&(0..64).map(|i|bf16::from_f32((i as f32-32.)/64.)).collect::<Vec<_>>()).unwrap();
        let w=Tensor::from_bf16(vec![64,64],&(0..4096).map(|i|bf16::from_f32((i%13)as f32/16.)).collect::<Vec<_>>()).unwrap();
        let mut b=Bank{input:transfers::to_cuda(&host,0).unwrap(),weight:transfers::to_cuda(&w,0).unwrap(),a:zeros(&ctx),b:zeros(&ctx),out:zeros(&ctx),prefix:CudaBuffer::alloc(16,0).unwrap()};
        let session=ExecutionSession::with_capacity(32*1024*1024,0).unwrap();
        ops::prepare_with_session(&session,||forward(&ctx,&mut b,true)).unwrap();ctx.synchronize().unwrap();
        let ids=session.prepared_ids_for_test();assert_eq!(ids.len(),2);assert_ne!(ids[0],ids[1]);
        let weights=read(&b.weight);
        let flat=ops::with_session(&session,||capture(&ctx,||forward(&ctx,&mut b,false))).unwrap();
        let fork=ops::with_session(&session,||capture(&ctx,||forward(&ctx,&mut b,true))).unwrap();
        let mut flat_dag=[0;5];let mut fork_dag=[0;5];
        assert_eq!(unsafe{apxinf_capture_frontier_test_topology(flat.graph_for_test(),flat_dag.as_mut_ptr())},0);
        assert_eq!(unsafe{apxinf_capture_frontier_test_topology(fork.graph_for_test(),fork_dag.as_mut_ptr())},0);
        assert_eq!(flat_dag[0],fork_dag[0]);assert_eq!(flat_dag[2],1);assert_eq!(fork_dag[2],1);assert_eq!(fork_dag[3],1);
        assert_eq!(fork_dag[1],flat_dag[1]+1);assert!(fork_dag[4]>=2);
        let mut original=None;
        for pattern in [0,1,0] {
            let data=(0..64).flat_map(|i|bf16::from_f32(((i*17+pattern*31)%67)as f32/64.).to_bits().to_le_bytes()).collect::<Vec<_>>();
            CudaBuffer::from_tensor(&b.input).unwrap().copy_from_host_on(&data,ctx.stream()).unwrap();
            flat.replay().unwrap();ctx.synchronize().unwrap();let expected=read(&b.out);
            for t in [&b.a,&b.b,&b.out]{CudaBuffer::from_tensor(t).unwrap().fill_async(0xff,ctx.stream()).unwrap();}
            fork.replay().unwrap();ctx.synchronize().unwrap();assert_eq!(read(&b.out),expected);assert_eq!(read(&b.input),data);assert_eq!(read(&b.weight),weights);
            if pattern==0 {if let Some(old)=&original {assert_eq!(&expected,old)}else{original=Some(expected)}}
        }
        let memory_before=ctx.memory_info().unwrap();
        for _ in 0..32 {fork.replay().unwrap();}
        ctx.synchronize().unwrap();let memory_after=ctx.memory_info().unwrap();
        assert_eq!(session.prepared_ids_for_test(),ids);assert!(session.used()<=session.capacity());
        eprintln!("FFN_FORK_DAG_RESOURCE_SNAPSHOT flat={flat_dag:?} fork={fork_dag:?} prepared={ids:?} arena={}/{} mem_before={memory_before:?} mem_after={memory_after:?}",session.used(),session.capacity());
        fork.replay().unwrap();drop(fork);ctx.synchronize().unwrap();assert_eq!(read(&b.weight),weights);drop(flat);
    }
    fn recovered(ctx:&CudaContext,buffer:&CudaBuffer) {
        assert!(!workspace::is_capturing());assert!(workspace::capture_failure().is_none());
        let g=capture(ctx,||buffer.fill_async(0x5a,ctx.stream()).map_err(Error::Cuda)).unwrap();g.replay().unwrap();ctx.synchronize().unwrap();
        let mut data=[0;16];buffer.copy_to_host(&mut data).unwrap();assert_eq!(data,[0x5a;16]);
    }
    #[test]
    fn failed_ignored_error_and_panic_poison_then_next_capture_recovers() {
        let _lock=crate::CUDA_TEST_LOCK.lock().unwrap();let ctx=CudaContext::new_with_autotune(0,false).unwrap();let buffer=CudaBuffer::alloc(16,0).unwrap();let other=CudaBuffer::alloc(16,0).unwrap();
        for fault in [Fault::Begin,Fault::AfterFirst,Fault::Rewind,Fault::AfterSecond,Fault::Join,Fault::GraphId,Fault::GraphHandle,Fault::NonDefaultEdge] {
            INJECT.with(|f|f.set(Some(fault)));
            let result=capture(&ctx,||{
                buffer.fill_async(0,ctx.stream()).map_err(Error::Cuda)?;
                let _ignored=capture_independent(&ctx,true,&mut(),|_|buffer.fill_async(1,ctx.stream()).map_err(Error::Cuda),|_|other.fill_async(2,ctx.stream()).map_err(Error::Cuda));
                Ok(())
            });assert!(result.is_err());recovered(&ctx,&buffer);
        }
        // The backend trait's split begin/end entry point must fail closed too.
        crate::graph::begin(&ctx).unwrap();
        INJECT.with(|f|f.set(Some(Fault::AfterFirst)));
        let _ignored=capture_independent(&ctx,true,&mut(),|_|buffer.fill_async(1,ctx.stream()).map_err(Error::Cuda),|_|other.fill_async(2,ctx.stream()).map_err(Error::Cuda));
        assert!(crate::graph::end(&ctx).is_err());recovered(&ctx,&buffer);
        let result=capture(&ctx,||{
            let panic=catch_unwind(AssertUnwindSafe(||capture_independent(&ctx,true,&mut(),|_|buffer.fill_async(1,ctx.stream()).map_err(Error::Cuda),|_|panic!("injected shared panic"))));
            assert!(panic.is_err());Ok(())
        });assert!(result.is_err());recovered(&ctx,&buffer);
        let outer=catch_unwind(AssertUnwindSafe(||capture(&ctx,||{capture_independent(&ctx,true,&mut(),|_|Ok(()),|_|panic!("uncaught shared panic"))?;Ok(())})));assert!(outer.is_err());recovered(&ctx,&buffer);
        let mut edge_bytes=0;
        for offset in 0..64 {let zero=unsafe{apxinf_capture_frontier_test_edge(offset,0)};if zero<0{break}assert_eq!(zero,1);assert_eq!(unsafe{apxinf_capture_frontier_test_edge(offset,1)},0);edge_bytes+=1;}assert!(edge_bytes>0&&edge_bytes<64);
        let second=CudaContext::new_with_autotune(0,false).unwrap();
        assert!(capture(&ctx,||{let _ignored=capture_independent(&second,true,&mut(),|_|Ok(()),|_|Ok(()));Ok(())}).is_err());recovered(&ctx,&buffer);
        let mut calls=0;assert!(!capture_independent(&ctx,true,&mut calls,|s|{*s+=1;Ok(())},|s|{*s+=1;Ok(())}).unwrap());assert_eq!(calls,0);
    }
}
