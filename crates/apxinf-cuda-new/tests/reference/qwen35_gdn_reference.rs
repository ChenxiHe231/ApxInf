//! Independent scalar CPU GDN math. No CUDA/half/model crate dependency.
//! F32 arithmetic in declared scalar order, BF16 RNE at existing decode boundaries.
#[path = "../support/qwen35_gdn_math_asset.rs"]
mod asset;
use asset::*;
use std::path::Path;

fn bf16(value:f32)->u16 {
    let b=value.to_bits();
    if b&0x7f800000==0x7f800000 && b&0x007fffff!=0 {return ((b>>16) as u16)|0x40;}
    (b.wrapping_add(0x7fff+((b>>16)&1))>>16) as u16
}
fn decode(bits:u16)->f32{f32::from_bits((bits as u32)<<16)}
fn rb(value:f32)->f32{decode(bf16(value))}
fn nz(n:i32)->i32{if n==0{1}else{n}}

struct Inputs {
    qkv:Vec<u16>,z:Vec<u16>,a:Vec<u16>,b:Vec<u16>,conv:Vec<u16>,alog:Vec<u16>,dt:Vec<u16>,norm:Vec<u16>,
    initial_conv:Vec<f32>,initial_recurrent:Vec<f32>,
}
fn inputs(rows:usize)->Inputs {
    Inputs {
        qkv:(0..rows*QKV).map(|i|bf16(nz(((i as u64*13+17)%71) as i32-35) as f32/64.0)).collect(),
        z:(0..rows*VW).map(|i|bf16(nz(((i as u64*7+11)%63) as i32-31) as f32/32.0)).collect(),
        a:(0..rows*VH).map(|i|bf16(nz(((i as u64*3+7)%35) as i32-17) as f32/32.0)).collect(),
        b:(0..rows*VH).map(|i|bf16(nz(((i as u64*5+19)%39) as i32-19) as f32/32.0)).collect(),
        conv:(0..QKV*CW).map(|i|{let tap=i%CW;let ch=i/CW;
            bf16(([16,-32,64,192][tap]+((ch+tap)%7) as i32-3) as f32/256.0)}).collect(),
        alog:(0..VH).map(|h|bf16((-192+(h%17) as i32-8) as f32/32.0)).collect(),
        dt:(0..VH).map(|h|bf16(nz((h%17) as i32-8) as f32/64.0)).collect(),
        norm:(0..D).map(|d|bf16((128+(d%13) as i32-6) as f32/128.0)).collect(),
        initial_conv:(0..QKV*CW).map(|i|nz((i*3%19) as i32-9) as f32/64.0).collect(),
        initial_recurrent:(0..VH*D*D).map(|i|nz((i*11%31) as i32-15) as f32/512.0).collect(),
    }
}
struct Step {convolved:Vec<u16>,decay:Vec<f32>,beta:Vec<f32>,core:Vec<u16>,gated:Vec<u16>}

// Independent rank-one delta-rule recurrence, value-major state[Dv,Dk].
// Decay is applied before prediction; beta correction precedes q readout.
fn recurrent(q:&[f32],k:&[f32],v:&[f32],decay:f32,beta:f32,state:&mut[f32])->Vec<u16>{
    let d=q.len();assert_eq!(k.len(),d);assert_eq!(v.len(),d);assert_eq!(state.len(),d*d);
    let mut out=vec![0u16;d];let scale=(d as f32).sqrt();
    for vd in 0..d {
        let s=&mut state[vd*d..(vd+1)*d];
        let mut predicted=0.0f32;
        for kd in 0..d {s[kd]*=decay;predicted+=s[kd]*k[kd];}
        let delta=(v[vd]-predicted)*beta;
        let mut value=0.0f32;
        for kd in 0..d {s[kd]+=delta*k[kd];value+=s[kd]*q[kd];}
        out[vd]=bf16(value/scale);
    }
    out
}
fn step(row:usize,i:&Inputs,conv_state:&mut[f32],state:&mut[f32])->Step{
    let mut convolved=vec![0u16;QKV];
    for channel in 0..QKV {
        let history=&mut conv_state[channel*CW..(channel+1)*CW];
        history.rotate_left(1);history[CW-1]=decode(i.qkv[row*QKV+channel]);
        let mut sum=0.0f32;
        for tap in 0..CW {sum+=history[tap]*decode(i.conv[channel*CW+tap]);}
        convolved[channel]=bf16(sum/(1.0+(-sum).exp()));
    }
    for offset in [0,QW] {for head in 0..KH {
        let begin=offset+head*D;let mut squared=0.0f32;
        for dim in 0..D {let value=decode(convolved[begin+dim]);squared+=value*value;}
        let inverse=1.0/(squared+EPS).sqrt();
        for dim in 0..D {convolved[begin+dim]=bf16(decode(convolved[begin+dim])*inverse);}
    }}
    let mut decay=vec![0.0f32;VH];let mut beta=vec![0.0f32;VH];let mut core=vec![0u16;VW];
    for head in 0..VH {
        let shifted=decode(i.a[row*VH+head])+decode(i.dt[head]);
        let softplus=if shifted>20.0{shifted}else{(1.0+shifted.exp()).ln()};
        decay[head]=-decode(i.alog[head]).exp()*softplus;
        beta[head]=1.0/(1.0+(-decode(i.b[row*VH+head])).exp());
        let kh=head/(VH/KH);
        let q:Vec<_>=convolved[kh*D..(kh+1)*D].iter().map(|&x|decode(x)).collect();
        let k:Vec<_>=convolved[QW+kh*D..QW+(kh+1)*D].iter().map(|&x|decode(x)).collect();
        let v:Vec<_>=convolved[2*QW+head*D..2*QW+(head+1)*D].iter().map(|&x|decode(x)).collect();
        let out=recurrent(&q,&k,&v,decay[head].exp(),beta[head],&mut state[head*D*D..(head+1)*D*D]);
        core[head*D..(head+1)*D].copy_from_slice(&out);
    }
    let mut gated=vec![0u16;VW];
    for head in 0..VH {
        let begin=head*D;let mut variance=0.0f32;
        for dim in 0..D {let value=decode(core[begin+dim]);variance+=value*value;}
        variance/=D as f32;let inverse=1.0/(variance+EPS).sqrt();
        for dim in 0..D {
            let normalized=rb(decode(core[begin+dim])*inverse)*decode(i.norm[dim]);
            let gate=decode(i.z[row*VW+begin+dim]);
            gated[begin+dim]=bf16(normalized*gate/(1.0+(-gate).exp()));
        }
    }
    assert!(state.iter().chain(conv_state.iter()).all(|v|v.is_finite()));
    assert!(gated.iter().any(|&v|v&0x7fff!=0),"degenerate all-zero readout");
    Step{convolved,decay,beta,core,gated}
}
fn generate(steps:usize)->Bundle {
    assert!([64,128,2048].contains(&steps));let rows=steps+1;let input=inputs(rows);
    let mut conv=input.initial_conv.clone();let mut state=input.initial_recurrent.clone();
    let(mut c,mut decay,mut beta,mut core,mut gated)=(Vec::new(),Vec::new(),Vec::new(),Vec::new(),Vec::new());
    let(mut pre_conv,mut pre_state)=(Vec::new(),Vec::new());
    for row in 0..rows {
        let s=step(row,&input,&mut conv,&mut state);
        c.extend(s.convolved);decay.extend(s.decay);beta.extend(s.beta);core.extend(s.core);gated.extend(s.gated);
        if row+1==steps {pre_conv=conv.clone();pre_state=state.clone();}
    }
    Bundle{steps,arrays:vec![
        u16_bytes(input.qkv),u16_bytes(input.z),u16_bytes(input.a),u16_bytes(input.b),
        u16_bytes(input.conv),u16_bytes(input.alog),u16_bytes(input.dt),u16_bytes(input.norm),
        f32_bytes(input.initial_conv),f32_bytes(input.initial_recurrent),
        u16_bytes(c),f32_bytes(decay),f32_bytes(beta),u16_bytes(core),u16_bytes(gated),
        f32_bytes(pre_conv),f32_bytes(pre_state),f32_bytes(conv),f32_bytes(state),
    ]}
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=5||args[1]!="--steps"||args[3]!="--out" {return Err("usage: oracle --steps64|128|2048 --out NEW-FILE".into());}
    let steps=args[2].parse::<usize>()?;let started=std::time::Instant::now();let bundle=generate(steps);
    bundle.write_new(Path::new(&args[4]))?;
    println!("GDN-CPU-REFERENCE steps={} rows={} geometry=16/32/128 conv=4 arithmetic=scalar-F32/BF16-RNE final_full_state=true all_output_rows=true generated=true gpu_run=false elapsed_s={:.3}",
             steps,steps+1,started.elapsed().as_secs_f64());
    Ok(())
}

#[test]
fn bf16_roundtrip_and_signed_midpoint_sanity(){
    for b in 0..=u16::MAX {if b&0x7f80!=0x7f80 {assert_eq!(bf16(decode(b)),b);}}
    for b in 0..0x7f7fu16 {
        let mid=((decode(b) as f64+decode(b+1) as f64)*0.5) as f32;
        let expected=if b&1==0{b}else{b+1};
        assert_eq!(bf16(mid),expected);assert_eq!(bf16(-mid),expected|0x8000);
    }
}
#[test]
fn rank_one_delta_rule_has_independent_two_dimensional_solution(){
    let mut s=[0.25,0.5,-0.5,0.75];
    let out=recurrent(&[1.0,0.0],&[1.0,0.0],&[2.0,3.0],1.0,0.5,&mut s);
    assert_eq!(s,[1.125,0.5,1.25,0.75]);
    assert_eq!(out,[bf16((1.125f64/2.0f64.sqrt()) as f32),bf16((1.25f64/2.0f64.sqrt()) as f32)]);
}
#[test]
fn zero_readout_recurrence_has_analytic_nonzero_state_half_life(){
    let mut s=vec![0.125f32;D*D];
    for n in 1..=4 {
        let out=recurrent(&vec![0.0;D],&vec![0.0;D],&vec![0.0;D],0.5,0.5,&mut s);
        assert!(out.iter().all(|&x|x==0));let expected=0.125/(1u32<<n) as f32;assert!(s.iter().all(|&x|x==expected));
    }
}
#[test]
fn convolution_history_after_four_steps_is_exact_prepared_input(){
    let i=inputs(4);let mut conv=i.initial_conv.clone();let mut state=i.initial_recurrent.clone();
    for row in 0..4 {let _=step(row,&i,&mut conv,&mut state);}
    for channel in 0..QKV {for tap in 0..4 {assert_eq!(conv[channel*4+tap].to_bits(),decode(i.qkv[tap*QKV+channel]).to_bits());}}
}
#[test]
fn all_declared_raw_inputs_are_finite_nonzero_and_prefix_stable(){
    let a=inputs(65);let b=inputs(129);
    assert_eq!(a.qkv.as_slice(),&b.qkv[..a.qkv.len()]);assert_eq!(a.z.as_slice(),&b.z[..a.z.len()]);
    for v in [&a.qkv,&a.z,&a.a,&a.b,&a.conv,&a.alog,&a.dt,&a.norm] {
        assert!(v.iter().all(|&x|x&0x7fff!=0&&x&0x7f80!=0x7f80));
    }
    assert!(a.initial_conv.iter().chain(&a.initial_recurrent).all(|x|x.is_finite()&&*x!=0.0));
}
