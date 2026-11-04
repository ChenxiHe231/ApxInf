//! Experimental FA2 packed-GQA candidate gates. All GPU entries are ignored.
//! Explicit prepare is separate from fresh strict quality runs; no timing scores.
use apxinf_core::{DType, Device, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext, ExecutionSession};
use half::bf16;
#[path="support/bf16_attention_reference.rs"] mod reference;
#[path="support/fa2_packed_gqa_uniform_reference.rs"] mod uniform;
const H:usize=16; const KH:usize=2; const D:usize=256; const GUARD:usize=256;
const MAX_ABS:f64=0.01;
const PREFIXES:[usize;10]=[127,128,129,4096,8192,16384,24576,32768,40960,49152];
const PREPARE_PREFIXES:[usize;11]=[65,127,128,129,4096,8192,16384,24576,32768,40960,49152];
fn raw(t:&Tensor)->Vec<u8>{transfers::to_cpu(t).unwrap().storage().as_cpu().unwrap().to_vec()}
fn fnv(b:&[u8])->String{format!("{:016x}",b.iter().fold(0xcbf29ce484222325u64,|h,b|(h^u64::from(*b)).wrapping_mul(0x100000001b3)))}
fn words(v:&[f32])->Vec<u8>{v.iter().flat_map(|&x|{let b=bf16::from_f32(x);assert_eq!(b.to_f32().to_bits(),x.to_bits());b.to_bits().to_le_bytes()}).collect()}
fn decode(b:&[u8])->Vec<f32>{b.chunks_exact(2).map(|x|bf16::from_bits(u16::from_le_bytes([x[0],x[1]])).to_f32()).collect()}
struct Guarded{whole:Tensor,t:Tensor,initial:Vec<u8>}
impl Guarded{
 fn new(ctx:&CudaContext,shape:Vec<usize>,dtype:DType,payload:Vec<u8>)->Self{
  assert_eq!(shape.iter().product::<usize>()*dtype.size_in_bytes(),payload.len());
  let mut bytes=vec![0xa5;payload.len()+2*GUARD];bytes[GUARD..GUARD+payload.len()].copy_from_slice(&payload);
  let host=Tensor::from_raw(Shape::new(vec![bytes.len()]),DType::I8,Device::Cpu,bytes.clone()).unwrap();
  let whole=transfers::to_cuda_on_stream(ctx,host).unwrap();
  let t=CudaBuffer::from_tensor(&whole).unwrap().view(GUARD,payload.len()).unwrap().as_tensor(Shape::new(shape),dtype).unwrap();
  let r=Self{whole,t,initial:bytes};assert_eq!(raw(&r.whole),r.initial);r
 }
 fn payload(&self)->Vec<u8>{let b=raw(&self.whole);b[GUARD..b.len()-GUARD].to_vec()}
 fn readonly(&self){assert_eq!(raw(&self.whole),self.initial,"read-only tensor/padding/outerguard changed");}
 fn guards(&self){let b=raw(&self.whole);assert_eq!(&b[..GUARD],&self.initial[..GUARD]);assert_eq!(&b[b.len()-GUARD..],&self.initial[self.initial.len()-GUARD..]);}
 fn poison(&self,ctx:&CudaContext){CudaBuffer::from_tensor(&self.t).unwrap().memset_async(0xff,ctx.stream()).unwrap();}
}
struct Fixture{p:usize,t:usize,cap:usize,q:Vec<f32>,k:Vec<f32>,v:Vec<f32>,expected:Vec<f32>,kind:&'static str}
impl Fixture{
 fn ordinary(p:usize,t:usize,shift:f32,reference_needed:bool)->Self{
  let valid=p+t;let cap=(valid+65).div_ceil(128)*128;
  let q=(0..t*H*D).map(|i|((i*13%61)as f32-30.)/32.).collect::<Vec<_>>();
  let mut k=vec![3.5;cap*KH*D];let mut v=vec![7.5;cap*KH*D];
  for pos in 0..valid{for h in 0..KH{for d in 0..D{let i=(pos*KH+h)*D+d;k[i]=(((pos*11+h*7+d*3)%59)as f32-29.)/64.;v[i]=(((pos*7+h*17+d*5)%31)as f32-15.)/32.+0.5+h as f32/4.+(d%8)as f32/32.+shift;}}}
  let g=reference::Geometry{queries:t,valid_keys:valid,capacity:cap,query_heads:H,kv_heads:KH,head_dim:D,query_start:p};
  let expected=if reference_needed{reference::causal_attention(g,&q,&k,&v,1./16.).unwrap()}else{Vec::new()};
  Self{p,t,cap,q,k,v,expected,kind:"nonzero-full-f64"}
 }
 // Additional typed arithmetic witness, NOT the ideal-F64 gate's domain.
 // S2 is forced by K66..69 =>2 K64 tiles. Split0 has64 zero-V scores-16;
 // split1 contains one score-.6875/V+-16 plus causal append scores0/V0.
 // FP32 partial O/LSE are combined before final BF16; no BF16 split-O step.
 fn probability_literal(t:usize,negative:bool)->Self{
  let p=65;let cap=(p+t+65).div_ceil(128)*128;
  let mut q=vec![0.;t*H*D];for r in 0..t{for h in 0..H{q[(r*H+h)*D+64]=1.;}}
  let mut k=vec![0.;cap*KH*D];let mut v=vec![16.;cap*KH*D];
  for pos in 0..p+t{for h in 0..KH{for d in 0..D{v[(pos*KH+h)*D+d]=0.;}if pos<64{k[(pos*KH+h)*D+64]=-256.;}else if pos==64{k[(pos*KH+h)*D+64]=-11.;}}}
  let sign=|h:usize,d:usize|if ((h+d)%2==0)^negative{1.}else{-1.};
  for h in 0..KH{for d in 0..D{v[(64*KH+h)*D+d]=sign(h,d)*16.;}}
  let literal=[5.375f32,3.21875,2.296875,1.7890625];let mut expected=vec![0.;t*H*D];
  for r in 0..t{for h in 0..H{for d in 0..D{expected[(r*H+h)*D+d]=sign(h/8,d)*literal[r];}}}
  Self{p,t,cap,q,k,v,expected,kind:"typed-BF16-P-FP32-denominator-split-FP32-finalBF16"}
 }

 // Nonzero one-hot prepared Q/K: every chronological row/head selects its
 // unique score8 prefix beacon. This distinguishes within-group permutations
 // that QKzero means cannot observe. All V codes are exact BF16 multiples1/16.
 fn head_beacon(p:usize,t:usize,shift:f32)->Self{
  assert!([40960,49152].contains(&p));let cap=(p+t+65).div_ceil(128)*128;
  let mut q=vec![0.;t*H*D];let mut k=vec![256.;cap*KH*D];let mut v=vec![16.;cap*KH*D];
  for r in 0..t{for h in 0..H{q[(r*H+h)*D+64+r*8+h%8]=1.;}}
  for pos in 0..p+t{for h in 0..KH{for d in 0..D{let sign=if h==0{1.}else{-1.};k[(pos*KH+h)*D+d]=0.;v[(pos*KH+h)*D+d]=sign*(-0.5+(d%4)as f32/16.)+shift;}}}
  for r in 0..t{for g in 0..8{let pos=p-t*8+r*8+g;for h in 0..KH{k[(pos*KH+h)*D+64+r*8+g]=128.;for d in 0..D{let sign=if h==0{1.}else{-1.};v[(pos*KH+h)*D+d]=sign*((r*8+g+1)as f32/4.+(d%4)as f32/16.)+shift;}}}}
  let mut sums=vec![0f64;KH*D];for pos in 0..p{for h in 0..KH{for d in 0..D{sums[h*D+d]+=f64::from(v[(pos*KH+h)*D+d]);}}}
  let weight=8f64.exp();let mut expected=vec![0.;t*H*D];
  for r in 0..t{for h in 0..H{let kv=h/8;let target=p-t*8+r*8+h%8;for d in 0..D{let sum=sums[kv*D+d]+(r+1)as f64*f64::from(v[(p*KH+kv)*D+d]);let target_value=f64::from(v[(target*KH+kv)*D+d]);expected[(r*H+h)*D+d]=((sum+(weight-1.)*target_value)/(weight+(p+r)as f64))as f32;}}}
  Self{p,t,cap,q,k,v,expected,kind:"nonzero-onehot-score8-independent-prefix-beacon"}
 }

 fn uniform(p:usize,t:usize,shift:f32)->Self{
  assert!(shift==0. || shift==0.125);
  let u=uniform::Fixture::new(p,t,shift==0.125);assert!(uniform::PREFIXES.contains(&p));
  let cap=u.capacity();let q=vec![0.;t*H*D];let k=vec![0.;cap*KH*D];let mut v=vec![0.;cap*KH*D];
  for pos in 0..cap{for h in 0..KH{for d in 0..D{v[(pos*KH+h)*D+d]=u.value_at(pos,h,d);}}}
  let expected=u.expected_output();Self{p,t,cap,q,k,v,expected,kind:"QKzero-independent-full-mean"}
 }
}
struct Bound{q:Guarded,k:Guarded,v:Guarded,out:Guarded,klive:Tensor,vlive:Tensor,p:usize,t:usize}
impl Bound{
 fn new(ctx:&CudaContext,f:&Fixture)->Self{
  let q=Guarded::new(ctx,vec![1,f.t,H,D],DType::BF16,words(&f.q));
  let k=Guarded::new(ctx,vec![1,f.cap,KH,D],DType::BF16,words(&f.k));let v=Guarded::new(ctx,vec![1,f.cap,KH,D],DType::BF16,words(&f.v));
  let live=|x:&Tensor|CudaBuffer::from_tensor(x).unwrap().view(0,(f.p+f.t)*KH*D*2).unwrap().as_tensor(Shape::new(vec![1,f.p+f.t,KH,D]),DType::BF16).unwrap();
  let klive=live(&k.t);let vlive=live(&v.t);
  let out=Guarded::new(ctx,vec![1,f.t,H,D],DType::BF16,vec![0xff;f.t*H*D*2]);Self{q,k,v,out,klive,vlive,p:f.p,t:f.t}
 }
 fn run(&self,ctx:&CudaContext,dir:&str,prepare:bool)->apxinf_core::Result<()>{
  let mut out=self.out.t.clone();let mut a=ops::KvCacheAttentionArgs::new(&self.q.t,&self.klive,&self.vlive,&mut out);
  a.valid_key_tokens=self.p+self.t;a.query_start=self.p;a.scale=1./16.;a.policy.online_tune=prepare;a.policy.allow_fallback=false;a.policy.graph_safe=true;a.policy.cache_dir=Some(dir.to_owned());ops::kv_cache_attention(ctx,a)
 }
 fn check(&self,f:&Fixture,label:&str)->Vec<u8>{
  self.q.readonly();self.k.readonly();self.v.readonly();self.out.guards();let b=self.out.payload();let got=decode(&b);assert_eq!(got.len(),f.expected.len());
  assert!(got.iter().all(|x|x.is_finite()));assert!(got.iter().any(|x|*x!=got[0]),"unexpected constant output");
  let m=reference::metrics(&got,&f.expected,D).unwrap();let rows=got.chunks_exact(D).zip(f.expected.chunks_exact(D)).map(|(a,b)|a.iter().zip(b).map(|(&x,&y)|(f64::from(x)-f64::from(y)).abs()).fold(0.,f64::max)).collect::<Vec<_>>();
  assert!(m.max_abs<MAX_ABS && m.max_head_row_abs<MAX_ABS,"frozen.01 failed {label}: {m:?}");
  println!("PACKED-GQA gate label={label} kind={} prefix={} queries={} full_output_elements={} all_live_keys={} output_fnv={} max_abs={} head_row_max_abs={rows:?} readonly=true guards=true performance_eligible=false",f.kind,f.p,f.t,got.len(),f.p+f.t,fnv(&b),m.max_abs);b
 }
}
fn config(prepare:bool)->String{
 assert!(std::env::args().any(|a|a=="--exact"),"run each ignored entry in a separate fresh --exact process");
 assert_eq!(std::env::var("APXINF_FA2_PACKED_GQA").as_deref(),Ok("1"));assert_eq!(std::env::var("APXINF_REQUIRE_RECIPES").as_deref(),Ok(if prepare{"0"}else{"1"}));
 let p=std::env::var("APXINF_FA2_PACKED_RECIPE_DIR").expect("explicit isolated recipe directory required");assert!(!p.is_empty());std::fs::create_dir_all(&p).unwrap();p
}
fn cases(defaults:&[usize])->Vec<(usize,usize)>{
 let ps=match std::env::var("APXINF_FA2_PACKED_TEST_PREFIX"){Ok(v)=>{let p=v.parse().unwrap();assert!(defaults.contains(&p),"prefix outside this declared entry scope");vec![p]},Err(_)=>defaults.to_vec()};
 let ts=match std::env::var("APXINF_FA2_PACKED_TEST_QUERIES"){Ok(v)=>{let t=v.parse().unwrap();assert!((1..=4).contains(&t));vec![t]},Err(_)=>(1..=4).collect()};
 let c=ps.into_iter().flat_map(|p|ts.iter().map(move |&t|(p,t))).collect::<Vec<_>>();println!("PACKED-GQA explicit_case_selection={c:?} numeric_threshold={MAX_ABS} performance_eligible=false");c
}
fn ab_a(ctx:&CudaContext,dir:&str,fa:&Fixture,fb:&Fixture){
 let a=Bound::new(ctx,fa);let b=Bound::new(ctx,fb);assert_ne!(a.q.t.storage().as_gpu().unwrap().ptr(),b.q.t.storage().as_gpu().unwrap().ptr());assert_ne!(a.k.t.storage().as_gpu().unwrap().ptr(),b.k.t.storage().as_gpu().unwrap().ptr());
 let sa=ExecutionSession::with_capacity(1<<20,ctx.device_id()).unwrap();let sb=ExecutionSession::with_capacity(1<<20,ctx.device_id()).unwrap();
 sa.prepare(||a.run(ctx,dir,false)).unwrap();ctx.synchronize().unwrap();let a0=a.check(fa,"A-first");
 sb.prepare(||b.run(ctx,dir,false)).unwrap();ctx.synchronize().unwrap();let b0=b.check(fb,"B-second-live");assert_ne!(a0,b0);assert_eq!(a.out.payload(),a0);
 // No fence between calls: two live prepared owners must not alias scratch.
 sa.run(||a.run(ctx,dir,false)).unwrap();sb.run(||b.run(ctx,dir,false)).unwrap();sa.run(||a.run(ctx,dir,false)).unwrap();ctx.synchronize().unwrap();
 assert_eq!(a.check(fa,"A-interleaved-repeat"),a0);assert_eq!(b.check(fb,"B-preserved-after-A"),b0);
 // Perturb all query0 future rows and physical padding, preserving its legal history.
 if fa.t>1{
  let mut future=Fixture{p:fa.p,t:fa.t,cap:fa.cap,q:fa.q.clone(),k:fa.k.clone(),v:fa.v.clone(),expected:Vec::new(),kind:"future-poison"};
  future.k[(fa.p+1)*KH*D..].fill(-6.5);future.v[(fa.p+1)*KH*D..].fill(9.5);
  let c=Bound::new(ctx,&future);c.run(ctx,dir,false).unwrap();ctx.synchronize().unwrap();let poisoned=c.out.payload();assert_eq!(&poisoned[..H*D*2],&a0[..H*D*2],"query0 leaked future/padding");c.q.readonly();c.k.readonly();c.v.readonly();c.out.guards();
 }
}
#[test]
#[ignore="GPU prepare ONLY; separate process REQUIRE0; never quality/performance acceptance"]
fn fa2_packed_gqa_recipe_prepare_only(){let dir=config(true);let ctx=CudaContext::new_with_autotune(0,true).unwrap();let c=cases(&PREPARE_PREFIXES);for (p,t) in &c{let f=Fixture::ordinary(*p,*t,0.,false);let a=Bound::new(&ctx,&f);a.run(&ctx,&dir,true).unwrap();ctx.synchronize().unwrap();a.q.readonly();a.k.readonly();a.v.readonly();a.out.guards();}println!("PACKED-GQA prepare_complete shapes={} quality_accepted=false performance_eligible=false",c.len());}
#[test]
#[ignore="fresh strict GPU; independent F64 ALL outputs/ALL live keys40K48K, two live bindings, .01"]
fn fa2_packed_gqa_full_output_oracle_gate(){let dir=config(false);let ctx=CudaContext::new_with_autotune(0,false).unwrap();for (p,t) in cases(&PREFIXES){let a=Fixture::ordinary(p,t,0.,true);let b=Fixture::ordinary(p,t,0.125,true);ab_a(&ctx,&dir,&a,&b);}}
#[test]
#[ignore="fresh strict GPU; independent full40K48K uniform means/time/KVgroup/future; not group-head or P-round proof"]
fn fa2_packed_gqa_long_analytic_gate(){let dir=config(false);let ctx=CudaContext::new_with_autotune(0,false).unwrap();for (p,t) in cases(&[40960,49152]){let a=Fixture::uniform(p,t,0.);let b=Fixture::uniform(p,t,0.125);ab_a(&ctx,&dir,&a,&b);}}
#[test]
#[ignore="fresh strict GPU; interleaved live Execution owners and graph-retained resources after session drop"]
fn fa2_packed_gqa_graph_owner_gate(){
 let dir=config(false);let ctx=CudaContext::new_with_autotune(0,false).unwrap();
 for (p,t) in cases(&[4096]){
  let fa=Fixture::ordinary(p,t,0.,true);let fb=Fixture::ordinary(p,t,0.125,true);let fc=Fixture::ordinary(p,t,-0.125,true);let a=Bound::new(&ctx,&fa);let b=Bound::new(&ctx,&fb);let c=Bound::new(&ctx,&fc);
  let sa=ExecutionSession::with_capacity(1<<20,0).unwrap();let sb=ExecutionSession::with_capacity(1<<20,0).unwrap();
  sa.prepare(||a.run(&ctx,&dir,false)).unwrap();sb.prepare(||b.run(&ctx,&dir,false)).unwrap();ctx.synchronize().unwrap();let a0=a.check(&fa,"graph-A-warm");let b0=b.check(&fb,"graph-B-warm");assert_ne!(a0,b0);
  let ga=apxinf_cuda_new::capture(&ctx,||sa.run(||a.run(&ctx,&dir,false))).unwrap();let gb=apxinf_cuda_new::capture(&ctx,||sb.run(||b.run(&ctx,&dir,false))).unwrap();drop(sa);drop(sb);
  let sc=ExecutionSession::with_capacity(1<<20,0).unwrap();sc.prepare(||c.run(&ctx,&dir,false)).unwrap();ctx.synchronize().unwrap();let c0=c.check(&fc,"third-owner-after-session-drop");
  for _ in 0..3{a.out.poison(&ctx);b.out.poison(&ctx);ga.replay().unwrap();gb.replay().unwrap();ga.replay().unwrap();ctx.synchronize().unwrap();assert_eq!(a.check(&fa,"graph-A-after-owner-drop"),a0);assert_eq!(b.check(&fb,"graph-B-after-owner-drop"),b0);assert_eq!(c.check(&fc,"third-owner-preserved"),c0);}
 }
}
#[test]
#[ignore="fresh strict GPU setup rejection; recipe dir must be empty; no online tune or fallback"]
fn fa2_packed_gqa_missing_recipe_is_error(){let dir=config(false);assert!(std::fs::read_dir(&dir).unwrap().next().is_none(),"use new empty recipe directory");let ctx=CudaContext::new_with_autotune(0,false).unwrap();let f=Fixture::ordinary(4096,4,0.,false);let a=Bound::new(&ctx,&f);let before=a.out.payload();let r=a.run(&ctx,&dir,false);assert!(r.is_err(),"strict miss silently tuned/fell back");ctx.synchronize().unwrap();a.q.readonly();a.k.readonly();a.v.readonly();a.out.guards();assert_eq!(a.out.payload(),before);assert!(std::fs::read_dir(&dir).unwrap().next().is_none());println!("PACKED-GQA strict_recipe_miss_rejected=true output_unchanged=true performance_eligible=false");}

#[test]
#[ignore="fresh strict GPU; additional typed probability BF16 witness; NOT idealF64 domain or relaxed original.01"]
fn fa2_packed_gqa_probability_rounding_literal(){let dir=config(false);let ctx=CudaContext::new_with_autotune(0,false).unwrap();for (_,t) in cases(&[65]){let a=Fixture::probability_literal(t,false);let b=Fixture::probability_literal(t,true);ab_a(&ctx,&dir,&a,&b);}}

#[test]
#[ignore="fresh strict GPU; nonzero Q/K full40K48K onehot beacons distinguish within-group head/time permutations"]
fn fa2_packed_gqa_nonzero_head_beacon_gate(){let dir=config(false);let ctx=CudaContext::new_with_autotune(0,false).unwrap();for (p,t) in cases(&[40960,49152]){let a=Fixture::head_beacon(p,t,0.);let b=Fixture::head_beacon(p,t,0.125);ab_a(&ctx,&dir,&a,&b);}}
