//! Ignored operator-only benchmark. Not model throughput or the30% target.
//! Prepare/tune and fresh strict measurement are separate processes.
use apxinf_core::{DType, Device, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext, ExecutionSession};
use half::bf16;
use std::{path::{Path,PathBuf},time::Instant};
#[path="support/bf16_attention_reference.rs"] mod reference;
const H:usize=16;const KH:usize=2;const D:usize=256;const GUARD:usize=256;
const WARMUPS:usize=2;const REPEATS:usize=5;const DEFAULT_ITERATIONS:usize=50;
const MAX_ABS:f64=0.01;
// Public CudaEvent disables timing and exposes no elapsed API. Host-wall
// batches include safe dispatch and one final stream synchronization.
mod contract{
 pub const PREFIXES:[usize;6]=[4096,8192,16384,24576,32768,49152];
 pub fn fnv(b:&[u8])->u64{b.iter().fold(0xcbf29ce484222325,|h,b|(h^u64::from(*b)).wrapping_mul(0x100000001b3))}
 #[derive(Debug,Clone,PartialEq,Eq)]pub struct RecipePin{pub provider:u32,pub implementation:u32,pub version:u32,pub configuration:i32}
 impl RecipePin{pub fn split(&self)->Option<i32>{if self.provider==2&&(self.implementation==3||self.implementation==4){Some(self.configuration)}else{None}}}
 pub fn parse_recipe(text:&str,packed:bool)->Result<(&str,RecipePin),String>{
  let mut lines=text.lines();let key=lines.next().ok_or("missing key")?;let encoded=lines.next().ok_or("missing tuple")?;
  if lines.any(|x|!x.trim().is_empty())||!key.starts_with("attention-recipe-v5|"){return Err("invalid exact recipe envelope".into());}
  if key.contains("|packed-gqa-q1to4-v1")!=packed{return Err("recipe mode mismatch".into());}
  let fields=encoded.split_whitespace().collect::<Vec<_>>();if fields.len()!=4{return Err("recipe tuple must have4 fields".into());}
  let pin=RecipePin{provider:fields[0].parse().map_err(|_|"provider")?,implementation:fields[1].parse().map_err(|_|"implementation")?,version:fields[2].parse().map_err(|_|"version")?,configuration:fields[3].parse().map_err(|_|"configuration")?};
  if pin.provider!=2||pin.version!=1{return Err("benchmark requires actual FA2 provider/version1".into());}
  if packed{if pin.implementation!=4||pin.configuration<2||pin.configuration>128{return Err("not packed split-KV tuple".into());}}
  else if !((pin.implementation==3&&(2..=128).contains(&pin.configuration))||(pin.implementation==1&&pin.configuration==0)){return Err("unsupported baseline FA2 tuple".into());}
  Ok((key,pin))
 }
 pub fn median(v:&[f64])->f64{assert!(!v.is_empty()&&v.iter().all(|x|x.is_finite()));let mut s=v.to_vec();s.sort_by(f64::total_cmp);let m=s.len()/2;if s.len()%2==1{s[m]}else{(s[m-1]+s[m])/2.}}
 #[test]fn cpu_contract(){
  assert_eq!(median(&[5.,1.,4.,3.,2.]),3.);assert_eq!(median(&[1.,3.,2.,4.]),2.5);
  let normal="attention-recipe-v5|fixture\n2 3 1 7\n";assert_eq!(parse_recipe(normal,false).unwrap().1.split(),Some(7));assert!(parse_recipe(normal,true).is_err());
  let packed="attention-recipe-v5|fixture|packed-gqa-q1to4-v1\n2 4 1 11\n";assert_eq!(parse_recipe(packed,true).unwrap().1.split(),Some(11));assert!(parse_recipe(packed,false).is_err());
  assert_eq!(parse_recipe("attention-recipe-v5|fixture\n2 1 1 0\n",false).unwrap().1.split(),None);
  for x in ["", "attention-recipe-v5|fixture\n2 3 1 0\n", "attention-recipe-v5|fixture\n2 3 1 7 extra\n", "wrong\n2 3 1 7\n", "attention-recipe-v5|fixture\n3 1 1 0\n"]{assert!(parse_recipe(x,false).is_err());}
  assert_eq!(PREFIXES.len()*2,12);assert_eq!(fnv(b""),0xcbf29ce484222325);
 }
}
fn raw(t:&Tensor)->Vec<u8>{transfers::to_cpu(t).unwrap().storage().as_cpu().unwrap().to_vec()}
fn words(v:&[f32])->Vec<u8>{v.iter().flat_map(|&x|{let b=bf16::from_f32(x);assert_eq!(b.to_f32().to_bits(),x.to_bits());b.to_bits().to_le_bytes()}).collect()}
struct Guarded{whole:Tensor,t:Tensor,initial:Vec<u8>}
impl Guarded{
 fn new(ctx:&CudaContext,shape:Vec<usize>,payload:Vec<u8>)->Self{
  assert_eq!(shape.iter().product::<usize>()*2,payload.len());let mut bytes=vec![0xa5;payload.len()+2*GUARD];bytes[GUARD..GUARD+payload.len()].copy_from_slice(&payload);
  let host=Tensor::from_raw(Shape::new(vec![bytes.len()]),DType::I8,Device::Cpu,bytes.clone()).unwrap();let whole=transfers::to_cuda_on_stream(ctx,host).unwrap();
  let t=CudaBuffer::from_tensor(&whole).unwrap().view(GUARD,payload.len()).unwrap().as_tensor(Shape::new(shape),DType::BF16).unwrap();let result=Self{whole,t,initial:bytes};assert_eq!(raw(&result.whole),result.initial);result
 }
 fn readonly(&self){assert_eq!(raw(&self.whole),self.initial,"input/padding/guard changed");}
 fn payload(&self)->Vec<u8>{let b=raw(&self.whole);assert!(b[..GUARD].iter().chain(&b[b.len()-GUARD..]).all(|x|*x==0xa5));b[GUARD..b.len()-GUARD].to_vec()}
}
struct Case{p:usize,t:usize,q:Guarded,k:Guarded,v:Guarded,klive:Tensor,vlive:Tensor,out:Guarded,expected:Vec<f32>,input_fnv:u64,reference_seconds:f64}
impl Case{
 fn new(ctx:&CudaContext,p:usize,t:usize,oracle:bool)->Self{
  let valid=p+t;let cap=(valid+65).div_ceil(128)*128;
  let q=(0..t*H*D).map(|i|((i*13%61)as f32-30.)/32.).collect::<Vec<_>>();let mut k=vec![3.5;cap*KH*D];let mut v=vec![7.5;cap*KH*D];
  for pos in 0..valid{for h in 0..KH{for d in 0..D{let i=(pos*KH+h)*D+d;k[i]=(((pos*11+h*7+d*3)%59)as f32-29.)/64.;v[i]=(((pos*7+h*17+d*5)%31)as f32-15.)/32.+0.5+h as f32/4.+(d%8)as f32/32.;}}}
  let start=Instant::now();let expected=if oracle{reference::causal_attention(reference::Geometry{queries:t,valid_keys:valid,capacity:cap,query_heads:H,kv_heads:KH,head_dim:D,query_start:p},&q,&k,&v,1./16.).unwrap()}else{Vec::new()};let reference_seconds=start.elapsed().as_secs_f64();
  let qb=words(&q);let kb=words(&k);let vb=words(&v);let input_fnv=qb.iter().chain(&kb).chain(&vb).fold(0xcbf29ce484222325u64,|h,b|(h^u64::from(*b)).wrapping_mul(0x100000001b3));
  let q=Guarded::new(ctx,vec![1,t,H,D],qb);let k=Guarded::new(ctx,vec![1,cap,KH,D],kb);let v=Guarded::new(ctx,vec![1,cap,KH,D],vb);
  let live=|x:&Tensor|CudaBuffer::from_tensor(x).unwrap().view(0,valid*KH*D*2).unwrap().as_tensor(Shape::new(vec![1,valid,KH,D]),DType::BF16).unwrap();let klive=live(&k.t);let vlive=live(&v.t);
  let out=Guarded::new(ctx,vec![1,t,H,D],vec![0xff;t*H*D*2]);Self{p,t,q,k,v,klive,vlive,out,expected,input_fnv,reference_seconds}
 }
 fn run(&self,ctx:&CudaContext,dir:&str,tune:bool)->apxinf_core::Result<()>{let mut out=self.out.t.clone();let mut a=ops::KvCacheAttentionArgs::new(&self.q.t,&self.klive,&self.vlive,&mut out);a.valid_key_tokens=self.p+self.t;a.query_start=self.p;a.scale=1./16.;a.policy.online_tune=tune;a.policy.allow_fallback=false;a.policy.graph_safe=true;a.policy.cache_dir=Some(dir.to_owned());ops::kv_cache_attention(ctx,a)}
 fn check(&self)->(f64,u64){self.q.readonly();self.k.readonly();self.v.readonly();let b=self.out.payload();let got=b.chunks_exact(2).map(|x|bf16::from_bits(u16::from_le_bytes([x[0],x[1]])).to_f32()).collect::<Vec<_>>();let m=reference::metrics(&got,&self.expected,D).unwrap();assert!(m.max_abs<MAX_ABS&&m.max_head_row_abs<MAX_ABS,"F64 admission/consistency failed {m:?}");(m.max_abs,contract::fnv(&b))}
}
fn flag()->bool{match std::env::var("APXINF_FA2_PACKED_GQA").as_deref(){Ok("0")=>false,Ok("1")=>true,_=>panic!("explicit APXINF_FA2_PACKED_GQA=0/1 required")}}
fn config(tune:bool)->(bool,PathBuf,String){
 assert!(std::env::args().any(|a|a=="--exact"),"fresh separate --exact process required");let packed=flag();assert_eq!(std::env::var("APXINF_REQUIRE_RECIPES").as_deref(),Ok(if tune{"0"}else{"1"}));assert_eq!(std::env::var("APXINF_ATTENTION_REPORT").as_deref(),Ok("1"));
 let root=PathBuf::from(std::env::var("APXINF_FA2_BENCH_RECIPE_ROOT").expect("explicit isolated per-flag/per-shape recipe root required"));assert!(!root.as_os_str().is_empty());
 let origin=std::env::var("APXINF_FA2_BENCH_RECIPE_ORIGIN").expect("recipe origin restored/retuned required");assert!(["restored","retuned"].contains(&origin.as_str()));(packed,root,origin)
}
fn cases()->Vec<(usize,usize)>{
 let ps=match std::env::var("APXINF_FA2_BENCH_PREFIX"){Ok(x)=>{let p=x.parse().unwrap();assert!(contract::PREFIXES.contains(&p));vec![p]},Err(_)=>contract::PREFIXES.to_vec()};let ts=match std::env::var("APXINF_FA2_BENCH_QUERIES"){Ok(x)=>{let t=x.parse().unwrap();assert!([1,4].contains(&t));vec![t]},Err(_)=>vec![1,4]};ps.into_iter().flat_map(|p|ts.iter().map(move|&t|(p,t))).collect()
}
fn directory(root:&Path,packed:bool,p:usize,t:usize)->PathBuf{root.join(if packed{"flag1"}else{"flag0"}).join(format!("p{p}-q{t}"))}
fn recipe(dir:&Path,packed:bool)->(contract::RecipePin,u64,u64,String){
 let paths=std::fs::read_dir(dir).unwrap().map(|x|x.unwrap().path()).collect::<Vec<_>>();assert_eq!(paths.len(),1,"one exact recipe and no other files per isolated case required");assert!(paths[0].symlink_metadata().unwrap().file_type().is_file());assert_eq!(paths[0].extension().and_then(|x|x.to_str()),Some("recipe"));
 let bytes=std::fs::read(&paths[0]).unwrap();let text=std::str::from_utf8(&bytes).unwrap();let (key,pin)=contract::parse_recipe(text,packed).unwrap();let key_fnv=contract::fnv(key.as_bytes());assert_eq!(paths[0].file_name().unwrap().to_str().unwrap(),format!("{key_fnv:x}.recipe"));
 for (name,actual) in [("APXINF_FA2_BENCH_EXPECT_PROVIDER",pin.provider as i64),("APXINF_FA2_BENCH_EXPECT_IMPLEMENTATION",pin.implementation as i64),("APXINF_FA2_BENCH_EXPECT_CONFIGURATION",i64::from(pin.configuration))]{if let Ok(v)=std::env::var(name){assert_eq!(actual,v.parse::<i64>().unwrap(),"existing recipe pin mismatch; never rewrite recipe");}}
 (pin,key_fnv,contract::fnv(&bytes),text.to_owned())
}
#[test]
#[ignore="GPU preparation only; never performance/quality score; root global lock required"]
fn fa2_packed_gqa_operator_prepare_only(){
 let (packed,root,origin)=config(true);let retune=origin=="retuned";let ctx=CudaContext::new_with_autotune(0,retune).unwrap();for (p,t) in cases(){let dir=directory(&root,packed,p,t);std::fs::create_dir_all(&dir).unwrap();let before=if retune{assert!(std::fs::read_dir(&dir).unwrap().next().is_none(),"retuned preparation requires fresh empty per-case directory");None}else{Some(recipe(&dir,packed))};let c=Case::new(&ctx,p,t,false);c.run(&ctx,dir.to_str().unwrap(),retune).unwrap();ctx.synchronize().unwrap();c.q.readonly();c.k.readonly();c.v.readonly();let (pin,key,bytes,text)=recipe(&dir,packed);if let Some(old)=before{assert_eq!((pin.clone(),key,bytes,text),old,"restored prepare changed recipe");}println!("{{\"schema\":\"apxinf.fa2.operator.prepare.v1\",\"packed\":{packed},\"prefix\":{p},\"queries\":{t},\"provider\":{},\"implementation\":{},\"configuration\":{},\"recipe_key_fnv\":\"{key:016x}\",\"recipe_bytes_fnv\":\"{bytes:016x}\",\"input_fnv\":\"{:016x}\",\"recipe_origin_declared\":\"{origin}\",\"quality_accepted\":false,\"performance_accepted\":false}}",pin.provider,pin.implementation,pin.configuration,c.input_fnv);}}
#[test]
#[ignore="fresh strict operator-only 2warm5; F64 full-output admission, no whole-model/30% claim"]
fn fa2_packed_gqa_operator_measure(){
 let (packed,root,origin)=config(false);let diagnostic=std::env::var("APXINF_FA2_BENCH_DIAGNOSTIC").as_deref()==Ok("1");let operator_eligible=!diagnostic;let iterations=std::env::var("APXINF_FA2_BENCH_ITERATIONS").map(|x|x.parse::<usize>().unwrap()).unwrap_or(DEFAULT_ITERATIONS);assert!((1..=100000).contains(&iterations));let ctx=CudaContext::new_with_autotune(0,false).unwrap();
 for (p,t) in cases(){
  let dir=directory(&root,packed,p,t);let before=recipe(&dir,packed);let c=Case::new(&ctx,p,t,true);let session=ExecutionSession::with_capacity(1<<20,ctx.device_id()).unwrap();
  let prep=Instant::now();session.prepare(||c.run(&ctx,dir.to_str().unwrap(),false)).unwrap();ctx.synchronize().unwrap();let preparation_seconds=prep.elapsed().as_secs_f64();let (admission_abs,expected_hash)=c.check();
  let split=before.0.split().map(|x|x.to_string()).unwrap_or("null".into());
  println!("{{\"schema\":\"apxinf.fa2.operator.config.v1\",\"packed\":{packed},\"prefix\":{p},\"queries\":{t},\"valid_keys\":{},\"iterations\":{iterations},\"warmups\":{WARMUPS},\"repeats\":{REPEATS},\"provider\":{},\"implementation\":{},\"version\":{},\"configuration\":{},\"split_count\":{split},\"same_s_forced\":false,\"matched_s_comparison\":\"pending-not-forced\",\"timing_method\":\"host-wall-batch-dispatch-plus-stream-sync\",\"device_event_timing_available\":false,\"recipe_origin_declared\":\"{origin}\",\"recipe_key_fnv\":\"{:016x}\",\"recipe_bytes_fnv\":\"{:016x}\",\"input_fnv\":\"{:016x}\",\"admission_max_abs\":{admission_abs},\"cpu_reference_seconds_excluded\":{},\"preparation_seconds_excluded\":{preparation_seconds},\"preparation_and_admission_operator_calls\":1,\"profiling_diagnostic\":{diagnostic},\"operator_measurement_eligible\":{operator_eligible},\"scope\":\"operator\",\"whole_model_eligible\":false}}",p+t,before.0.provider,before.0.implementation,before.0.version,before.0.configuration,before.1,before.2,c.input_fnv,c.reference_seconds);
  for _ in 0..WARMUPS{for _ in 0..iterations{session.run(||c.run(&ctx,dir.to_str().unwrap(),false)).unwrap();}ctx.synchronize().unwrap();assert_eq!(c.check().1,expected_hash);}
  let mut host=Vec::new();
  for sample in 0..REPEATS{
   ctx.synchronize().unwrap();let clock=Instant::now();for _ in 0..iterations{session.run(||c.run(&ctx,dir.to_str().unwrap(),false)).unwrap();}ctx.synchronize().unwrap();let host_ms=clock.elapsed().as_secs_f64()*1e3;
   let (max_abs,digest)=c.check();assert_eq!(digest,expected_hash);assert_eq!(recipe(&dir,packed),before,"strict run changed recipe bytes/config");host.push(host_ms/iterations as f64);
   println!("{{\"schema\":\"apxinf.fa2.operator.sample.v1\",\"packed\":{packed},\"prefix\":{p},\"queries\":{t},\"sample\":{sample},\"iterations\":{iterations},\"device_event_ms\":null,\"host_wall_ms\":{host_ms},\"host_ms_per_call\":{},\"max_abs\":{max_abs},\"output_fnv\":\"{digest:016x}\",\"validation_and_D2H_excluded\":true,\"profiling_diagnostic\":{diagnostic},\"operator_measurement_eligible\":{operator_eligible},\"whole_model_eligible\":false}}",host_ms/iterations as f64);
  }
  println!("{{\"schema\":\"apxinf.fa2.operator.summary.v1\",\"packed\":{packed},\"prefix\":{p},\"queries\":{t},\"repeats\":{REPEATS},\"device_event_median_ms_per_call\":null,\"host_median_ms_per_call\":{},\"operator_scope_only\":true,\"profiling_diagnostic\":{diagnostic},\"operator_measurement_eligible\":{operator_eligible},\"whole_model_eligible\":false}}",contract::median(&host));
 }
}
