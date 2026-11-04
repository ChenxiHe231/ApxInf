//! Artifact for crates/apxinf-cuda-new/tests/qwen35_gdn_independent_math.rs.
//! Copies only parser/metric support; math reference is an offline CPU asset.
#[path = "support/qwen35_gdn_math_asset.rs"]
mod asset;
use asset::*;
use std::path::PathBuf;
use apxinf_core::{DType,Device,Shape,Tensor};
use apxinf_cuda_new::{ops,transfers,CudaBuffer,CudaContext,CudaStream};

fn bytes(t:&Tensor)->Vec<u8>{transfers::to_cpu(t).unwrap().storage().as_cpu().unwrap().to_vec()}
fn values(t:&Tensor)->Vec<f32>{
    let raw=bytes(t);
    if t.dtype()==DType::BF16 {raw.chunks_exact(2).map(|b|f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16)).collect()}
    else {assert_eq!(t.dtype(),DType::F32);raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()}
}
fn upload(ctx:&CudaContext,raw:Vec<u8>,shape:Vec<usize>,dtype:DType)->Tensor{
    let host=Tensor::from_raw(Shape::new(shape.clone()),dtype,Device::Cpu,raw).unwrap();
    let out=CudaBuffer::alloc(host.storage().len(),ctx.device_id()).unwrap().as_tensor(Shape::new(shape),dtype).unwrap();
    ctx.synchronize().unwrap();
    let status=transfers::copy_cpu_to_cuda(&host,&out)
        .map_err(|e|e.to_string()).and_then(|()|CudaStream::default_stream().synchronize());
    if let Err(e)=status {std::mem::forget((host,out));panic!("fixture upload: {e}");}out
}
fn zero(ctx:&CudaContext,shape:Vec<usize>,dtype:DType)->Tensor{
    let n=shape.iter().product::<usize>()*dtype.size_in_bytes();
    CudaBuffer::alloc_zeros_async(n,ctx.device_id(),ctx.stream()).unwrap().as_tensor(Shape::new(shape),dtype).unwrap()
}
fn view(t:&Tensor,offset:usize,shape:Vec<usize>,dtype:DType)->Tensor{
    CudaBuffer::from_tensor(t).unwrap().view(offset*dtype.size_in_bytes(),shape.iter().product::<usize>()*dtype.size_in_bytes()).unwrap()
        .as_tensor(Shape::new(shape),dtype).unwrap()
}
struct Cake{q:Tensor,k:Tensor,v:Tensor,out:Tensor,alpha:Tensor,cu:Tensor,workspace:Tensor}
struct Fixture{
    qkv:Tensor,z:Tensor,a:Tensor,b:Tensor,conv_weight:Tensor,alog:Tensor,dt:Tensor,norm:Tensor,
    conv:Tensor,state:Tensor,convolved:Tensor,decay:Tensor,beta:Tensor,core:Tensor,gated:Tensor,cake:Option<Cake>,
}
impl Fixture{
    fn new(ctx:&CudaContext,b:&Bundle,with_cake:bool)->Self{
        let rows=b.steps+1;
        let inp=|id:usize,shape:Vec<usize>,dtype:DType|upload(ctx,b.arrays[id].clone(),shape,dtype);
        let cake=with_cake.then(||{
            assert!(ctx.caps().sm>=100,"Cake gate explicitly requires Blackwell/Thor; no silent skip");
            Cake{
                q:zero(ctx,vec![rows,KH,D],DType::F16),k:zero(ctx,vec![rows,KH,D],DType::F16),
                v:zero(ctx,vec![rows,VH,D],DType::F16),out:zero(ctx,vec![rows,VH,D],DType::F16),
                alpha:zero(ctx,vec![rows,VH],DType::F32),cu:zero(ctx,vec![2],DType::I32),
                workspace:zero(ctx,vec![ops::gdn_prefill_workspace_bytes(VH,1).div_ceil(4)],DType::F32),
            }
        });
        let f=Self{
            qkv:inp(0,vec![rows,QKV],DType::BF16),z:inp(1,vec![rows,VW],DType::BF16),
            a:inp(2,vec![rows,VH],DType::BF16),b:inp(3,vec![rows,VH],DType::BF16),
            conv_weight:inp(4,vec![QKV,1,CW],DType::BF16),alog:inp(5,vec![VH],DType::BF16),
            dt:inp(6,vec![VH],DType::BF16),norm:inp(7,vec![D],DType::BF16),
            conv:inp(8,vec![QKV,CW],DType::F32),state:inp(9,vec![VH,D,D],DType::F32),
            convolved:zero(ctx,vec![rows,QKV],DType::BF16),decay:zero(ctx,vec![rows,VH],DType::F32),
            beta:zero(ctx,vec![rows,VH],DType::F32),core:zero(ctx,vec![rows,VW],DType::BF16),
            gated:zero(ctx,vec![rows,VW],DType::BF16),cake,
        };
        for t in [&f.gated,&f.convolved,&f.core] {CudaBuffer::from_tensor(t).unwrap().memset_async(0xff,ctx.stream()).unwrap();}
        f
    }
    fn weights(&self)->ops::Qwen35GdnCoreWeights<'_>{
        ops::Qwen35GdnCoreWeights{conv:&self.conv_weight,a_log:&self.alog,dt_bias:&self.dt,norm:&self.norm}
    }
    fn prefill(&self,ctx:&CudaContext,start:usize,rows:usize){
        let cake=self.cake.as_ref().map(|c|Cake{
            q:view(&c.q,0,vec![rows,KH,D],DType::F16),k:view(&c.k,0,vec![rows,KH,D],DType::F16),
            v:view(&c.v,0,vec![rows,VH,D],DType::F16),out:view(&c.out,0,vec![rows,VH,D],DType::F16),
            alpha:view(&c.alpha,0,vec![rows,VH],DType::F32),cu:c.cu.clone(),workspace:c.workspace.clone(),
        });
        ops::qwen35_gdn_prefill(ctx,rows,&ops::Qwen35GdnPrefillArgs{
            projected_qkv:&view(&self.qkv,start*QKV,vec![rows,QKV],DType::BF16),
            z:&view(&self.z,start*VW,vec![rows,VW],DType::BF16),
            a:&view(&self.a,start*VH,vec![rows,VH],DType::BF16),b:&view(&self.b,start*VH,vec![rows,VH],DType::BF16),
            weights:self.weights(),state:ops::Qwen35GdnState{conv:&self.conv,recurrent:&self.state},checkpoints:None,
            scratch:ops::Qwen35GdnPrefillScratch{
                convolved_qkv:&view(&self.convolved,start*QKV,vec![rows,QKV],DType::BF16),
                decay:&view(&self.decay,start*VH,vec![rows,VH],DType::F32),beta:&view(&self.beta,start*VH,vec![rows,VH],DType::F32),
                core_output:&view(&self.core,start*VW,vec![rows,VW],DType::BF16),
                gated_output:&view(&self.gated,start*VW,vec![rows,VW],DType::BF16),
            },cake:cake.as_ref().map(|c|ops::Qwen35GdnCakeScratch{
                q16:&c.q,k16:&c.k,v16:&c.v,out16:&c.out,alpha:&c.alpha,cu_seqlens:&c.cu,workspace:&c.workspace,
            }),
        }).unwrap();
    }
    fn decode(&self,ctx:&CudaContext,row:usize){
        ops::qwen35_gdn_decode(ctx,&ops::Qwen35GdnDecodeArgs{
            projected_qkv:&view(&self.qkv,row*QKV,vec![QKV],DType::BF16),z:&view(&self.z,row*VW,vec![VW],DType::BF16),
            a:&view(&self.a,row*VH,vec![VH],DType::BF16),b:&view(&self.b,row*VH,vec![VH],DType::BF16),
            weights:self.weights(),state:ops::Qwen35GdnState{conv:&self.conv,recurrent:&self.state},
            scratch:ops::Qwen35GdnDecodeScratch{
                convolved_qkv:&view(&self.convolved,row*QKV,vec![QKV],DType::BF16),
                decay:&view(&self.decay,row*VH,vec![VH],DType::F32),beta:&view(&self.beta,row*VH,vec![VH],DType::F32),
                core_output:&view(&self.core,row*VW,vec![VW],DType::BF16),gated_output:&view(&self.gated,row*VW,vec![VW],DType::BF16),
            },
        }).unwrap();
    }
    fn readonly_inputs(&self,b:&Bundle){
        for (id,t) in [&self.qkv,&self.z,&self.a,&self.b,&self.conv_weight,&self.alog,&self.dt,&self.norm].into_iter().enumerate(){
            assert_eq!(bytes(t),b.arrays[id],"input/weight section{id} changed");
        }
    }
}
fn check(name:&str,got:&[f32],want:&[f32],width:usize,cos:f64,rel:f64,gate_rows:bool){
    let m=metrics(got,want,width);
    println!("GDN-INDEPENDENT {name} elements={} cosine={:.9} relL2={:.9} maxabs={:.9} worst_row={} worst_row_relL2={:.9} worst_row_cos={:.9}",
             got.len(),m.cosine,m.rel,m.maxabs,m.worst_row,m.worst_row_rel,m.worst_row_cos);
    assert!(m.cosine>=cos&&m.rel<=rel,"{name}: {m:?}");
    if gate_rows{assert!(m.worst_row_rel<=rel&&m.worst_row_cos>=cos,"{name} worst token: {m:?}");}
}
fn load(steps:usize)->Bundle{
    let root=PathBuf::from(std::env::var_os("APXINF_GDN_MATH_ASSET_DIR").expect("set CPU-generated/frozen APXINF_GDN_MATH_ASSET_DIR"));
    assert!(root.is_absolute());let path=root.join(format!("gdn-math-{steps}.bin"));
    let b=Bundle::read(&path).unwrap();assert_eq!(b.steps,steps);
    println!("GDN-INDEPENDENT asset={} steps={steps} rows={} stage_fingerprints={:?} independent_math=true CPU_contract=F32/BF16-RNE SHA256_must_be_bound_by_owner_receipt",
             path.display(),steps+1,b.arrays.iter().map(|a|format!("{:016x}",fnv(a))).collect::<Vec<_>>());b
}
fn boundary(f:&Fixture,b:&Bundle,rows:usize,after_decode:bool,label:&str){
    let(ci,si)=if after_decode{(17,18)}else{(15,16)};
    assert_eq!(bytes(&f.conv),b.arrays[ci],"{label} conv state not bit exact");
    check(&format!("{label}/recurrent"),&values(&f.state),&b.f32_values(si),D,0.9998,0.025,false);
    let got=values(&view(&f.gated,0,vec![rows,VW],DType::BF16));
    let reference=b.f32_values(14);
    check(&format!("{label}/all-gated"),&got,&reference[..rows*VW],VW,0.999,0.05,true);
    let first=rows.min(64);
    check(&format!("{label}/first64"),&got[..first*VW],&reference[..first*VW],VW,0.999,0.05,true);
}
fn run_case(steps:usize,with_cake:bool,split:bool){
    let b=load(steps);let ctx=CudaContext::new_with_autotune(0,false).unwrap();
    assert_eq!((ops::QWEN35_GDN_KEY_HEADS,ops::QWEN35_GDN_VALUE_HEADS,ops::QWEN35_GDN_HEAD_DIM,ops::QWEN35_GDN_CONV_WIDTH),(KH,VH,D,CW));
    assert_eq!(ops::QWEN35_GDN_EPSILON.to_bits(),EPS.to_bits());
    let f=Fixture::new(&ctx,&b,with_cake);
    ctx.synchronize().unwrap();
    assert_eq!(bytes(&f.conv),b.arrays[8],"initial conv state upload mismatch");
    assert_eq!(bytes(&f.state),b.arrays[9],"initial recurrent state upload mismatch");
    let provider=if with_cake{"Cake-F16"}else{"scalar-BF16"};
    if split{
        assert_eq!(steps,128);let short=load(64);
        for id in 0..8{
            if id<4{let(count,kind)=count_kind(64,id);assert_eq!(&b.arrays[id][..count*kind],short.arrays[id].as_slice());}
            else{assert_eq!(b.arrays[id],short.arrays[id]);}
        }
        assert_eq!(b.arrays[8],short.arrays[8]);assert_eq!(b.arrays[9],short.arrays[9]);
        f.prefill(&ctx,0,64);ctx.synchronize().unwrap();
        boundary(&f,&short,64,false,&format!("{provider}/split64"));
        assert!(bytes(&view(&f.gated,64*VW,vec![65,VW],DType::BF16)).iter().all(|&x|x==0xff),"first segment overwrote future outputs");
        f.prefill(&ctx,64,64);
    }else{f.prefill(&ctx,0,steps);}
    ctx.synchronize().unwrap();
    boundary(&f,&b,steps,false,&format!("{provider}/prefill{steps}"));
    for (id,t) in [(11,&f.decay),(12,&f.beta)]{
        let got=values(&view(t,0,vec![steps,VH],DType::F32));let want=b.f32_values(id);
        let m=metrics(&got,&want[..steps*VH],VH);
        println!("GDN-INDEPENDENT {provider}/gate{id} {m:?}");assert!(m.maxabs<2.0e-5);
    }
    if !with_cake{
        check("scalar normalized-convolved",&values(&view(&f.convolved,0,vec![steps,QKV],DType::BF16)),
              &b.f32_values(10)[..steps*QKV],QKV,0.99999,0.01,true);
        let m=metrics(&values(&view(&f.core,0,vec![steps,VW],DType::BF16)),&b.f32_values(13)[..steps*VW],VW);
        println!("GDN-INDEPENDENT scalar/core diagnostic_only {m:?}");
    }else{println!("GDN-INDEPENDENT Cake convolved/core BF16 scratch not applicable; CPU reference is BF16 canonical math, Cake uses F16 intermediates");}
    let prefix=bytes(&view(&f.gated,0,vec![steps,VW],DType::BF16));
    assert!(bytes(&view(&f.gated,steps*VW,vec![VW],DType::BF16)).iter().all(|&x|x==0xff),"prefill overwrote reserved next-row canary");
    f.decode(&ctx,steps);ctx.synchronize().unwrap();
    boundary(&f,&b,steps+1,true,&format!("{provider}/next-decode"));
    assert_eq!(bytes(&view(&f.gated,0,vec![steps,VW],DType::BF16)),prefix,"decode rewrote previous outputs; mutation guard only, not math oracle");
    f.readonly_inputs(&b);
    println!("GDN-INDEPENDENT PASS steps={steps} provider={provider} split={split} full_output_and_final_state=true independent_math=true no_model_quality_claim=true");
}
#[test]
#[ignore="independent CPU asset + exclusive Thor GPU gate"]
fn gdn_independent_math_64(){for cake in [false,true]{run_case(64,cake,false);}}
#[test]
#[ignore="independent CPU asset + exclusive Thor GPU gate"]
fn gdn_independent_math_128(){for cake in [false,true]{run_case(128,cake,false);}}
#[test]
#[ignore="2048 complete rows, scalar+Cake versus offline independent CPU math"]
fn gdn_independent_math_2048(){for cake in [false,true]{run_case(2048,cake,false);}}
#[test]
#[ignore="nonzero carried64-state ->append64 against independent CPU references"]
fn gdn_independent_math_prefix64_append64(){run_case(128,true,true);}
