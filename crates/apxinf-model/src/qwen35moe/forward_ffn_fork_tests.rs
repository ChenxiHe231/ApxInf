//! GPU gates for the same production scratch and SAME ExecutionSession.
//! Root runs this ignored unit test under its external device guard; no timing score.
use super::*;
use apxinf_cuda_new::{capture, ExecutionSession};
use half::bf16;
use sha2::{Digest,Sha256};
use std::{path::PathBuf,ffi::OsString};
struct Env(Vec<(&'static str,Option<OsString>)>);
impl Env {
    fn baseline()->Self {
        let values=[("APXINF_QWEN35_SHARED_SWIGLU_QUANT","0"),("APXINF_QWEN35_SHARED_GATE_UP","0")];
        let mut saved=values.iter().map(|(k,_)|(*k,std::env::var_os(k))).collect::<Vec<_>>();
        saved.push(("APXINF_QWEN35_MOE_WS_POISON",std::env::var_os("APXINF_QWEN35_MOE_WS_POISON")));
        std::env::remove_var("APXINF_QWEN35_MOE_WS_POISON");
        for(k,v)in values{std::env::set_var(k,v)}Self(saved)
    }
}
impl Drop for Env {fn drop(&mut self){for(k,v)in &self.0{if let Some(v)=v{std::env::set_var(k,v)}else{std::env::remove_var(k)}}}}
fn bytes(t:&Tensor)->Vec<u8>{let b=CudaBuffer::from_tensor(t).unwrap();let mut out=vec![0;b.len()];b.copy_to_host(&mut out).unwrap();out}
fn hash(t:&Tensor)->Vec<u8>{Sha256::digest(bytes(t)).to_vec()}
struct Guard{whole:CudaBuffer,size:usize}
impl Guard{
    fn payload(&self)->CudaBuffer{self.whole.view(256,self.size).unwrap()}
    fn check(&self,ctx:&CudaContext){let mut h=[0;256];let mut t=[0;256];self.whole.view(0,256).unwrap().copy_to_host_on(&mut h,ctx.stream()).unwrap();self.whole.view(256+self.size,256).unwrap().copy_to_host_on(&mut t,ctx.stream()).unwrap();assert!(h.iter().chain(t.iter()).all(|x|*x==0xa5),"FFN guard corrupt");}
}
fn guarded(ctx:&CudaContext,src:&CudaBuffer)->Guard{
    let g=Guard{whole:CudaBuffer::alloc(src.len()+512,ctx.device_id()).unwrap(),size:src.len()};g.whole.fill_async(0xa5,ctx.stream()).unwrap();g.payload().copy_from_device_async(src,src.len(),ctx.stream()).unwrap();ctx.synchronize().unwrap();g
}
fn guard_tensor(ctx:&CudaContext,t:&mut Tensor,guards:&mut Vec<Guard>){let g=guarded(ctx,&CudaBuffer::from_tensor(t).unwrap());*t=g.payload().as_tensor(t.shape().clone(),t.dtype()).unwrap();guards.push(g);}
fn guard_scratch(ctx:&CudaContext,s:&mut BaseMoeScratch)->Vec<Guard>{
    let mut g=Vec::new();
    for t in [&mut s.router_logits,&mut s.expert_ids,&mut s.expert_weights,&mut s.assignment_expert_ids,&mut s.assignment_token_ids,&mut s.assignment_slots,&mut s.token_assignment_indices,&mut s.assignment_weights,&mut s.expert_offsets,&mut s.route_intermediate,&mut s.shared_intermediate,&mut s.assignment_output,&mut s.shared_output,&mut s.shared_gate_logits,&mut s.output]{guard_tensor(ctx,t,&mut g)}
    for t in [&mut s.expert_fused,&mut s.act_packed,&mut s.act_scales,&mut s.inter_packed,&mut s.inter_scales]{if let Some(t)=t.as_mut(){guard_tensor(ctx,t,&mut g)}}
    if let Some(workspace)=s.edge_workspace.as_mut(){let x=guarded(ctx,workspace);*workspace=x.payload();g.push(x)}g
}
fn live_bytes(t:&Tensor,n:usize)->Vec<u8>{let b=CudaBuffer::from_tensor(t).unwrap().view(0,n).unwrap();let mut v=vec![0;n];b.copy_to_host(&mut v).unwrap();v}
fn quantized(s:&BaseMoeScratch)->Vec<Vec<u8>>{vec![live_bytes(s.act_packed.as_ref().unwrap(),s.tokens*s.hidden/2),bytes(s.act_scales.as_ref().unwrap()),live_bytes(s.inter_packed.as_ref().unwrap(),s.tokens*s.intermediate/2),bytes(s.inter_scales.as_ref().unwrap())]}
fn poison(ctx:&CudaContext,s:&BaseMoeScratch,workspace:u8){
    for t in [&s.router_logits,&s.expert_ids,&s.expert_weights,&s.assignment_output,&s.shared_intermediate,&s.shared_output,&s.shared_gate_logits,&s.output]{CudaBuffer::from_tensor(t).unwrap().fill_async(0xff,ctx.stream()).unwrap();}
    for t in [&s.expert_fused,&s.act_packed,&s.act_scales,&s.inter_packed,&s.inter_scales]{CudaBuffer::from_tensor(t.as_ref().unwrap()).unwrap().fill_async(0xff,ctx.stream()).unwrap();}
    s.edge_workspace.as_ref().unwrap().fill_async(workspace,ctx.stream()).unwrap();
}
fn state(s:&BaseMoeScratch)->(Vec<u8>,Vec<u8>,Vec<u8>,Vec<Vec<u8>>){(bytes(s.output()),bytes(&s.expert_ids),bytes(&s.expert_weights),quantized(s))}
fn snapshot_ranges(s:&BaseMoeScratch)->Vec<(&'static str,usize,usize)>{
    let mut out=Vec::new();
    for(group,t)in [("R",&s.expert_ids),("R",&s.expert_weights),("R",&s.assignment_output),("S",&s.shared_intermediate),("S",&s.shared_output),("S",&s.shared_gate_logits)]{let b=CudaBuffer::from_tensor(t).unwrap();out.push((group,t.storage().as_gpu().unwrap().ptr(),b.len()))}
    let b=s.edge_workspace.as_ref().unwrap();let t=b.as_tensor(Shape::new(vec![b.len()]),DType::I8).unwrap();out.push(("R",t.storage().as_gpu().unwrap().ptr(),b.len()));
    for t in [&s.expert_fused,&s.act_packed,&s.act_scales,&s.inter_packed,&s.inter_scales]{let b=CudaBuffer::from_tensor(t.as_ref().unwrap()).unwrap();out.push(("S",t.as_ref().unwrap().storage().as_gpu().unwrap().ptr(),b.len()))}out
}
fn weights(m:&DeviceNvfp4Moe)->Vec<&Tensor>{
    let mut out=vec![&m.router.weight,&m.shared_expert_gate.weight];
    for w in [&m.shared_gate,&m.shared_up,&m.shared_down]{out.extend([&w.packed_weight,&w.block_scales,&w.weight_scale_2]);out.push(w.gemm_block_scales.as_ref().unwrap())}
    let DeviceNvfp4RoutedExperts::Edge(e)=&m.routed_experts else{panic!("all256 Edge required")};
    assert_eq!((e.experts,e.hidden,e.intermediate),(256,2048,512));
    out.extend([&e.fc1_weight,&e.fc1_scales,&e.fc1_alpha,&e.input_global_scale,&e.fc2_weight,&e.fc2_scales,&e.fc2_alpha,&e.down_input_scale]);out
}
#[test]
#[ignore="root-owned real checkpoint CUDA13.2 exact/resource gate, no performance result"]
fn actual_layer_same_session_default_ffn_graph_exact_guards_and_serial_boundaries(){
    let _env=Env::baseline();
    let ctx=CudaContext::new_with_recipe_options(0,false,std::env::var("APXINF_RECIPE_CACHE_DIR").ok()).unwrap();
    let before_load=ctx.memory_info().unwrap();
    let path=PathBuf::from(std::env::var_os("APXINF_QWEN35_CHECKPOINT").expect("set real checkpoint"));
    let config=super::super::config::Qwen35MoeConfig::from_json_file(&path.join("config.json")).unwrap();
    let moe=super::super::device_weights::load_qwen35_moe_layer_from_path(&ctx,&config,&path,27).unwrap();ctx.synchronize().unwrap();
    let source_before=weights(&moe).into_iter().map(hash).collect::<Vec<_>>();
    for m in [1,4,2,3]{
        let mut s=BaseMoeScratch::new(&ctx,m,2048,256,8,512).unwrap();let eligible=matches!(m,1|4);
        let mut guards=guard_scratch(&ctx,&mut s);
        let mut input=zeros(&ctx,vec![m,2048],DType::BF16).unwrap();guard_tensor(&ctx,&mut input,&mut guards);
        let ranges=snapshot_ranges(&s);
        for(a,x,n)in &ranges{for(b,y,k)in &ranges{if a!=b{assert!(*x+*n<=*y||*y+*k<=*x,"cross-branch writable alias")}}}
        s.edge_workspace.as_ref().unwrap().fill_async(0,ctx.stream()).unwrap();
        let session=ExecutionSession::with_capacity(32*1024*1024,0).unwrap();
        ops::prepare_with_session(&session,||s.run(&ctx,&moe,&input)).unwrap();ctx.synchronize().unwrap();let arena_prepared=session.used();
        let graph=ops::with_session(&session,||capture(&ctx,||s.run(&ctx,&moe,&input))).unwrap();
        let bound_before=ctx.memory_info().unwrap();assert!(before_load.0.saturating_sub(bound_before.0)<2*1024*1024*1024,"exact gate exceeds2GiB device delta");
        let mut first=None;let mut distinct=None;
        for(pass,pattern)in [0usize,1,0,2].into_iter().enumerate(){
            let data=(0..m*2048).flat_map(|j|{
                let x=if pattern==2{0.}else{(((j*(17+pattern*16)+pattern*197)%257)as f32-128.)/128.};bf16::from_f32(x).to_bits().to_le_bytes()
            }).collect::<Vec<_>>();
            CudaBuffer::from_tensor(&input).unwrap().copy_from_host_on(&data,ctx.stream()).unwrap();
            poison(&ctx,&s,if pass%2==0{0}else{255});
            ops::with_session(&session,||s.run(&ctx,&moe,&input)).unwrap();ctx.synchronize().unwrap();let expected=state(&s);
            assert!(expected.0.chunks_exact(2).all(|x|bf16::from_bits(u16::from_le_bytes([x[0],x[1]])).to_f32().is_finite()));
            poison(&ctx,&s,0);
            graph.replay().unwrap();ctx.synchronize().unwrap();assert_eq!(state(&s),expected,"default graph/eager output/IDs/weights/packed/SF exact");
            assert_eq!(bytes(&input),data,"input readonly");for g in &guards{g.check(&ctx)}
            assert_eq!(weights(&moe).into_iter().map(hash).collect::<Vec<_>>(),source_before,"W/SF/scalars readonly");
            if pass==0{first=Some(expected.clone())}if pass==1{distinct=Some(expected.1.clone())}if pass==2{assert_eq!(first.as_ref().unwrap(),&expected,"0->1->0 restore");assert_ne!(distinct.as_ref().unwrap(),&expected.1,"route distribution must change")}
        }
        let bound_after=ctx.memory_info().unwrap();assert!(arena_prepared<=session.capacity());
        eprintln!("FFN_SAME_SESSION_GATE M={m} default_fork={eligible} ranges={ranges:?} guarded_payload={} arena={arena_prepared}/{} mem_before={bound_before:?} mem_after={bound_after:?}",guards.iter().map(|g|g.size).sum::<usize>(),session.capacity());
        // Graphs are destroyed before their explicit fixed-op scratch owners.
        graph.replay().unwrap();drop(graph);ctx.synchronize().unwrap();for g in &guards{g.check(&ctx)}
    }
}
