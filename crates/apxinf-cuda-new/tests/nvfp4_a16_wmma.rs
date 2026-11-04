//! Ignored GPU oracle/guard tests for default-off W4A16 WMMA candidate.
//! Use the unchanged safe L3 op and C ABI. Model/precision/perf acceptance is
//! separate; this fixture is synthetic, including real vocab geometry only.
use apxinf_core::{DType,Device,Shape,Tensor};
use apxinf_cuda_new::{ops,transfers,CudaBuffer,CudaContext};
#[path="support/nvfp4_a16_independent_reference.rs"]mod reference;
const GUARD:usize=256;const MAX_ABS:f64=0.01;
fn bytes(v:&[f32])->Vec<u8>{v.iter().flat_map(|x|((reference::bf16(*x).to_bits()>>16)as u16).to_le_bytes()).collect()}
fn decode(b:&[u8])->Vec<f32>{b.chunks_exact(2).map(|x|f32::from_bits(u32::from(u16::from_le_bytes([x[0],x[1]]))<<16)).collect()}
fn fnv(b:&[u8])->String{format!("{:016x}",b.iter().fold(0xcbf29ce484222325u64,|h,b|(h^u64::from(*b)).wrapping_mul(0x100000001b3)))}
struct Guarded{whole:Tensor,view:Tensor,initial:Vec<u8>}
impl Guarded{
 fn new(ctx:&CudaContext,shape:Vec<usize>,dtype:DType,payload:Vec<u8>)->Self{
  assert_eq!(payload.len(),shape.iter().product::<usize>()*dtype.size_in_bytes());
  let mut data=vec![0xa5;payload.len()+2*GUARD];data[GUARD..GUARD+payload.len()].copy_from_slice(&payload);
  let host=Tensor::from_raw(Shape::new(vec![data.len()]),DType::I8,Device::Cpu,data.clone()).unwrap();
  let whole=transfers::to_cuda_on_stream(ctx,host).unwrap();
  let view=CudaBuffer::from_tensor(&whole).unwrap().view(GUARD,payload.len()).unwrap().as_tensor(Shape::new(shape),dtype).unwrap();
  let out=Self{whole,view,initial:data};assert_eq!(out.raw(),out.initial);out
 }
 fn raw(&self)->Vec<u8>{transfers::to_cpu(&self.whole).unwrap().storage().as_cpu().unwrap().to_vec()}
 fn payload(&self)->Vec<u8>{let b=self.raw();assert!(b[..GUARD].iter().chain(&b[b.len()-GUARD..]).all(|x|*x==0xa5));b[GUARD..b.len()-GUARD].to_vec()}
 fn readonly(&self){assert_eq!(self.raw(),self.initial);}
 fn poison(&self,ctx:&CudaContext){CudaBuffer::from_tensor(&self.view).unwrap().memset_async(0xff,ctx.stream()).unwrap();}
}
struct Weight{n:usize,k:usize,q:Guarded,s:Guarded,g:Guarded}
impl Weight{
 fn new(ctx:&CudaContext,n:usize,k:usize,q:&[u8],s:&[u8],global:f32)->Self{
  Self{n,k,q:Guarded::new(ctx,vec![n/128,k/64,128,32],DType::E2M1Pair,q.to_vec()),s:Guarded::new(ctx,vec![n/128,k/64,128,4],DType::F8E4M3,s.to_vec()),g:Guarded::new(ctx,vec![1],DType::F32,global.to_le_bytes().to_vec())}
 }
 fn run(&self,ctx:&CudaContext,a:&Guarded,out:&Guarded)->apxinf_core::Result<()>{ops::nvfp4_a16_blackwell_gemv(ctx,&a.view,&self.q.view,&self.s.view,&self.g.view,&out.view)}
 fn readonly(&self){self.q.readonly();self.s.readonly();self.g.readonly();}
}
fn policy()->(bool,usize){
 let enabled=match std::env::var("APXINF_NVFP4_A16_WMMA").as_deref(){Err(_)|Ok("0")=>false,Ok("1")=>true,_=>panic!("WMMA flag must be0/1")};
 let tile=if enabled{match std::env::var("APXINF_NVFP4_A16_WMMA_TILE_N").as_deref(){Err(_)|Ok("64")=>64,Ok("32")=>32,Ok("128")=>128,_=>panic!("TileN must32/64/128")}}else{0};
 (enabled,tile)
}
fn check(w:&Weight,a:&Guarded,o:&Guarded,expected:&[f32],rows:usize,label:&str,constant:bool)->Vec<u8>{
 w.readonly();a.readonly();let raw=o.payload();let got=decode(&raw);assert_eq!(got.len(),rows*w.n);assert_eq!(got.len(),expected.len());
 let mut worst=0.0f64;let mut row_max=vec![0.0f64;rows];let mut argmaxes=Vec::new();
 for row in 0..rows{
  let range=row*w.n..(row+1)*w.n;let gr=&got[range.clone()];let er=&expected[range];
  for (&x,&y)in gr.iter().zip(er){assert!(x.is_finite()&&y.is_finite());let e=(f64::from(x)-f64::from(y)).abs();worst=worst.max(e);row_max[row]=row_max[row].max(e);}
  if !constant{assert!(gr.iter().any(|x|*x!=gr[0]),"fixture output unexpectedly constant");}
  let g=reference::argmax(gr);let e=reference::argmax(er);assert_eq!(g,e,"argmax differs in row{row}");argmaxes.push(g);
 }
 assert!(worst<MAX_ABS,"{label} max_abs{worst} must be strictly <{MAX_ABS}");
 let (requested,tile)=policy();println!("A16-GATE label={label} N={} K={} M={rows} requested_wmma={requested} tileN={tile} max_abs={worst} row_max_abs={row_max:?} argmax={argmaxes:?} output_fnv={} readonly_full=true guards=true BF16_A=true global_scale_after_dot=true performance_eligible=false",w.n,w.k,fnv(&raw));raw
}
fn ab_a(ctx:&CudaContext,w:&Weight,rows:usize,a:&[f32],expected:&[f32],label:&str){
 let input=Guarded::new(ctx,vec![rows,w.k],DType::BF16,bytes(a));let neg=a.iter().map(|x|-*x).collect::<Vec<_>>();let other=Guarded::new(ctx,vec![rows,w.k],DType::BF16,bytes(&neg));
 let out=Guarded::new(ctx,vec![rows,w.n],DType::BF16,vec![0xff;rows*w.n*2]);let bout=Guarded::new(ctx,vec![rows,w.n],DType::BF16,vec![0xff;rows*w.n*2]);
 assert_ne!(input.view.storage().as_gpu().unwrap().ptr(),other.view.storage().as_gpu().unwrap().ptr());
 assert_ne!(out.view.storage().as_gpu().unwrap().ptr(),bout.view.storage().as_gpu().unwrap().ptr());
 w.run(ctx,&input,&out).unwrap();ctx.synchronize().unwrap();let first=check(w,&input,&out,expected,rows,&format!("{label}-A"),false);
 w.run(ctx,&other,&bout).unwrap();ctx.synchronize().unwrap();let b_expected=expected.iter().map(|x|-*x).collect::<Vec<_>>();let b=check(w,&other,&bout,&b_expected,rows,&format!("{label}-B-neg"),false);assert_eq!(out.payload(),first);
 out.poison(ctx);w.run(ctx,&input,&out).unwrap();ctx.synchronize().unwrap();assert_eq!(check(w,&input,&out,expected,rows,&format!("{label}-A-repeat"),false),first);assert_eq!(bout.payload(),b);other.readonly();
}
#[test]
#[ignore="GPU exclusive lock; M1..4, N128/256, K64/128/192/1984/2048/2112; fresh process per tile"]
fn nvfp4_a16_wmma_packing_and_dense_matrix(){
 let _=policy();let ctx=CudaContext::new_with_autotune(0,false).unwrap();let mut cases=0;
 for n in [128,256]{for k in [64,128,192,1984,2048,2112]{
  let(q,s)=reference::packed_fixture(n,k);let w=Weight::new(&ctx,n,k,&q,&s,1.0/128.0);
  let dense=reference::activation(4,k);let gold=reference::evaluate(n,k,4,&dense,&q,&s,1.0/128.0).unwrap();
  println!("A16-REF N={n} K={k} max_abs_dot={} max_l1_dot={} weight_q_fnv={} weight_scale_fnv={}",gold.max_abs_dot,gold.max_l1_dot,fnv(&q),fnv(&s));
  let mut basis=vec![0.0;4*k];for row in 0..4{basis[row*k+[0,15,16,k-1][row]]=1.0;}
  let basis_gold=reference::evaluate(n,k,4,&basis,&q,&s,1.0/128.0).unwrap();
  for rows in 1..=4{ab_a(&ctx,&w,rows,&dense[..rows*k],&gold.output[..rows*n],"dense");ab_a(&ctx,&w,rows,&basis[..rows*k],&basis_gold.output[..rows*n],"basis");cases+=2;}
 }}
 println!("A16-GATE small suite PASS cases={cases} executions={} threshold_strict=0.01 mathematical_reference=independent_f64 performance_eligible=false",cases*3);
}
#[test]
#[ignore="GPU exclusive lock; actual Qwen vocab248320/K2048 geometry, synthetic immutable weights; expensive CPU F64 reference"]
fn nvfp4_a16_wmma_real_vocab_geometry(){
 let _=policy();let ctx=CudaContext::new_with_autotune(0,false).unwrap();let(n,k)=(248320,2048);
 let(q,s)=reference::packed_fixture(n,k);let a=reference::activation(4,k);let gold=reference::evaluate(n,k,4,&a,&q,&s,1.0/128.0).unwrap();
 println!("A16-REF real-vocab synthetic N={n} K={k} max_abs_dot={} max_l1_dot={} q_fnv={} sf_fnv={}",gold.max_abs_dot,gold.max_l1_dot,fnv(&q),fnv(&s));
 let w=Weight::new(&ctx,n,k,&q,&s,1.0/128.0);for rows in 1..=4{ab_a(&ctx,&w,rows,&a[..rows*k],&gold.output[..rows*n],"real-vocab");}
 println!("A16-GATE real-vocab PASS shapes=4 executions=12 all_output_elements_checked=true argmax_checked=true performance_eligible=false");
}
#[test]
#[ignore="GPU exclusive lock; exhaustive finite weight product oracle and global F32 stage literal"]
fn nvfp4_a16_wmma_all_finite_codes_and_global_scale(){
 let _=policy();let ctx=CudaContext::new_with_autotune(0,false).unwrap();let(n,k)=(4096,64);let mut q=vec![0;n*k/2];let mut s=vec![0x38;n*k/16];let finite=(0u8..=255).filter(|c|c&0x7f!=0x7f).collect::<Vec<_>>();
 for col in 0..4064{let tile=(col/128)*128+col%128;q[tile*32]=((col%16)as u8)&15;s[tile*4]=finite[col/16];}
 let mut a=vec![0.;k];a[0]=1.;let gold=reference::evaluate(n,k,1,&a,&q,&s,1.0).unwrap();let w=Weight::new(&ctx,n,k,&q,&s,1.0);let input=Guarded::new(&ctx,vec![1,k],DType::BF16,bytes(&a));let out=Guarded::new(&ctx,vec![1,n],DType::BF16,vec![0xff;n*2]);
 w.run(&ctx,&input,&out).unwrap();ctx.synchronize().unwrap();let b=check(&w,&input,&out,&gold.output,1,"all4064-finite-products",false);assert_eq!(b,bytes(&gold.output));
 let(n,k)=(256,192);let mut q=vec![0;n*k/2];let mut sf=vec![0;n*k/16];q[513*32]=0x50;sf[513*4]=0x30;
 let mut a=vec![0.;4*k];for row in 0..4{a[row*k+[0,63,64,65][row]]=1.;}let mut expected=vec![0.;4*n];expected[3*n+129]=1.5;
 let w=Weight::new(&ctx,n,k,&q,&sf,1.);let input=Guarded::new(&ctx,vec![4,k],DType::BF16,bytes(&a));let out=Guarded::new(&ctx,vec![4,n],DType::BF16,vec![0xff;4*n*2]);
 w.run(&ctx,&input,&out).unwrap();ctx.synchronize().unwrap();let got=check(&w,&input,&out,&expected,4,"N129-K65-high-nibble-beacon",true);assert_eq!(got,bytes(&expected));
 let(n,k)=(128,64);let mut q=vec![0;n*k/2];let s=vec![0x38;n*k/16];for col in 0..n{for pair in 0..k/2{q[col*32+pair]=if pair%2==0{0xa5}else{0xaa};}}
 let w=Weight::new(&ctx,n,k,&q,&s,1.1);for rows in 1..=4{let a=vec![1.0;rows*k];let input=Guarded::new(&ctx,vec![rows,k],DType::BF16,bytes(&a));let out=Guarded::new(&ctx,vec![rows,n],DType::BF16,vec![0xff;rows*n*2]);
  w.run(&ctx,&input,&out).unwrap();ctx.synchronize().unwrap();let b=check(&w,&input,&out,&vec![0.0;rows*n],rows,"global-after-dot-literal",true);assert!(b.iter().all(|x|*x==0));}
 println!("A16-GATE finite4064 and F32-global-after-dot PASS original=0 wrongly_folded_BF16=-0.125 performance_eligible=false");
}
#[test]
#[ignore="GPU exclusive lock; all small rows eager/graph/metadata failure gates"]
fn nvfp4_a16_wmma_capture_and_validation(){
 let _=policy();let ctx=CudaContext::new_with_autotune(0,false).unwrap();let(n,k)=(256,2048);let(q,s)=reference::packed_fixture(n,k);let a=reference::activation(4,k);let gold=reference::evaluate(n,k,4,&a,&q,&s,1.0/128.0).unwrap();let w=Weight::new(&ctx,n,k,&q,&s,1.0/128.0);
 for rows in 1..=4{let input=Guarded::new(&ctx,vec![rows,k],DType::BF16,bytes(&a[..rows*k]));let out=Guarded::new(&ctx,vec![rows,n],DType::BF16,vec![0xff;rows*n*2]);
  w.run(&ctx,&input,&out).unwrap();ctx.synchronize().unwrap();let b=check(&w,&input,&out,&gold.output[..rows*n],rows,"capture-warm",false);
  let graph=apxinf_cuda_new::capture(&ctx,||w.run(&ctx,&input,&out)).unwrap();for i in 0..4{out.poison(&ctx);if i%2==0{graph.replay().unwrap();}else{w.run(&ctx,&input,&out).unwrap();}ctx.synchronize().unwrap();assert_eq!(check(&w,&input,&out,&gold.output[..rows*n],rows,if i%2==0{"graph"}else{"eager"},false),b);}
 }
 let(n,k)=(128,128);let(q,s)=reference::packed_fixture(n,k);let w=Weight::new(&ctx,n,k,&q,&s,1.0);let input=Guarded::new(&ctx,vec![1,k],DType::BF16,bytes(&vec![1.;k]));let before=input.raw();
 assert!(ops::nvfp4_a16_blackwell_gemv(&ctx,&input.view,&w.q.view,&w.s.view,&w.g.view,&input.view).is_err());ctx.synchronize().unwrap();assert_eq!(input.raw(),before);w.readonly();
 println!("A16-GATE capture PASS shapes=4 graph_replays=8 eager_replays=8 warmups=4 alias_reject_before_write=true performance_eligible=false");
}
