//! Standalone GDN causal-prologue gate. No checkpoint, scan, model, or recipe.
//! Public gdn_conv_prepare; exact full-array prepared literals, not CUDA math.
use apxinf_core::{DType, Device, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext, CudaStream};
const T: usize = 64;
const KH: usize = 16;
const VH: usize = 32;
const D: usize = 128;
const C: usize = (2*KH+VH)*D;
const GUARD: usize = 256;
const GUARD_BYTE: u8 = 0xa5;
const POISON: u8 = 0xcd;

fn u16s(bits: impl IntoIterator<Item=u16>) -> Vec<u8> {
    bits.into_iter().flat_map(u16::to_le_bytes).collect()
}
fn f32s(values: impl IntoIterator<Item=f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn with_guards(payload: &[u8]) -> Vec<u8> {
    let mut all = vec![GUARD_BYTE; GUARD+payload.len()+GUARD];
    all[GUARD..GUARD+payload.len()].copy_from_slice(payload); all
}
struct Guarded { whole: Tensor, view: Tensor, initial: Vec<u8> }
impl Guarded {
    fn new(ctx: &CudaContext, payload: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        assert_eq!(payload.len(), shape.iter().product::<usize>()*dtype.size_in_bytes());
        let initial=with_guards(&payload);
        let host=Tensor::from_raw(Shape::new(vec![initial.len()]), DType::I8, Device::Cpu, initial.clone()).unwrap();
        let buffer=CudaBuffer::alloc(initial.len(),ctx.device_id()).unwrap();
        let whole=buffer.as_tensor(Shape::new(vec![initial.len()]),DType::I8).unwrap();
        // Explicit old-API-safe dependency and host lifetime. Do not make
        // prologue correctness depend on the separately qualified H2D fix.
        ctx.synchronize().unwrap();
        let upload=transfers::copy_cpu_to_cuda(&host,&whole)
            .map_err(|e|e.to_string())
            .and_then(|()|CudaStream::default_stream().synchronize());
        if let Err(error)=upload {
            std::mem::forget((host,whole,buffer));
            panic!("fixture H2D drain failed, owners retained: {error}");
        }
        let view=buffer.view(GUARD,payload.len()).unwrap().as_tensor(Shape::new(shape),dtype).unwrap();
        Self { whole,view,initial }
    }
    fn raw(&self) -> Vec<u8> {
        transfers::to_cpu(&self.whole).unwrap().storage().as_cpu().unwrap().to_vec()
    }
    fn check(&self,label:&str,payload:Option<&[u8]>) -> bool {
        let expected=payload.map(with_guards).unwrap_or_else(||self.initial.clone());
        let actual=self.raw();
        assert_eq!(actual.len(),expected.len());
        let mut guards=0;let mut contents=0;let mut first=None;
        for (i,(&got,&want)) in actual.iter().zip(&expected).enumerate() {
            if got!=want {
                if i<GUARD || i>=actual.len()-GUARD {guards+=1;}else{contents+=1;}
                first.get_or_insert((i,got,want));
            }
        }
        println!("GDN-PROLOGUE label={label} payload_bytes={} payload_mismatches={contents} guard_mismatches={guards} first_difference={first:?}",actual.len()-2*GUARD);
        contents==0 && guards==0
    }
}
fn run(carry:bool) {
    let ctx=CudaContext::new_with_autotune(0,false).unwrap();
    let input=Guarded::new(&ctx,vec![0;T*C*2],vec![T,C],DType::BF16);
    let weight=Guarded::new(&ctx,u16s((0..C).flat_map(|_|[0x3f80,0x4000,0x4080,0x4100])),vec![C,4],DType::BF16);
    let history=if carry {[9.0,1.0,2.0,4.0]}else{[0.0;4]};
    let window=Guarded::new(&ctx,f32s((0..C).flat_map(|_|history)),vec![C,4],DType::F32);
    let decay=Guarded::new(&ctx,vec![0;T*VH*4],vec![T,VH],DType::F32);
    let q=Guarded::new(&ctx,vec![POISON;T*KH*D*2],vec![T,KH,D],DType::F16);
    let k=Guarded::new(&ctx,vec![POISON;T*KH*D*2],vec![T,KH,D],DType::F16);
    let v=Guarded::new(&ctx,vec![POISON;T*VH*D*2],vec![T,VH,D],DType::F16);
    let alpha=Guarded::new(&ctx,vec![POISON;T*VH*4],vec![T,VH],DType::F32);
    ctx.synchronize().unwrap();
    // Initial bytes are checked before the public op; neither actual output
    // nor a CUDA helper is used to manufacture any expected result.
    for (name,tensor) in [("initial/input",&input),("initial/weight",&weight),("initial/window",&window),("initial/decay",&decay)] {
        assert!(tensor.check(name,None));
    }
    ops::gdn_conv_prepare(&ctx,&input.view,&weight.view,&window.view,&q.view,&k.view,&v.view,
                         &decay.view,&alpha.view,T,KH,VH,1.0e-6).unwrap();
    ctx.synchronize().unwrap();
    // Frozen literals: incoming history is rotated once per zero input.
    // Raw sums21/10/4 -> BF16 SiLU21/10/3.921875 -> these FP16 bits.
    let qk_expected=u16s((0..T).flat_map(|row| std::iter::repeat_n(if carry && row<3 {0x2da8}else{0},KH*D)));
    let v_expected=u16s((0..T).flat_map(|row| {
        let bits=if carry {match row {0=>0x4d40,1=>0x4900,2=>0x43d8,_=>0}}else{0};
        std::iter::repeat_n(bits,VH*D)
    }));
    let alpha_expected=f32s(std::iter::repeat_n(1.0,T*VH));
    let window_expected=vec![0;C*4*4];
    let checks=[
        input.check("readonly/input",None),weight.check("readonly/weight",None),decay.check("readonly/decay",None),
        q.check("Q/all-elements",Some(&qk_expected)),k.check("K/all-elements",Some(&qk_expected)),
        v.check("V/all-elements",Some(&v_expected)),alpha.check("alpha/all-elements",Some(&alpha_expected)),
        window.check("final-window/all-elements",Some(&window_expected)),
    ];
    println!("GDN-PROLOGUE case={} tokens={T} key_heads={KH} value_heads={VH} dim={D} cold_control={} all_checks={} independent_literal=true model_scan_not_run=true",
             if carry {"nonzero-carry"}else{"cold-zero"},!carry,checks.iter().all(|&ok|ok));
    assert!(checks.into_iter().all(|ok|ok),"GDN causal-prologue exact-byte/readonly/guard gate failed; retain baseline evidence");
}
#[test]
#[ignore="exclusive GPU; standalone public-op nonzero carry boundary, no model/scan"]
fn gdn_conv_prepare_nonzero_causal_prologue_exact(){run(true);}
#[test]
#[ignore="exclusive GPU; cold-zero control for the same standalone public op"]
fn gdn_conv_prepare_cold_zero_window_exact(){run(false);}
