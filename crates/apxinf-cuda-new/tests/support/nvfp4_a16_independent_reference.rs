//! Independent scalar math/packing oracle; std-only, no native/CUDA decoder.
//! BF16 activations remain BF16. Weights use actual N128/K64 packing. The
//! F32 global multiplier follows the complete dot and is never folded into B.
pub fn fp4(code:u8)->f64 {
    let v=[0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0][(code&7)as usize];
    if code&8==0 {v}else{-v}
}
pub fn e4m3(code:u8)->Result<f64,String> {
    let e=(code>>3)&15;let m=code&7;
    if e==15&&m==7{return Err("E4M3 NaN is outside finite-weight reference domain".into());}
    let v=if e==0 {f64::from(m)/512.0}else{f64::from(8+m)*2.0f64.powi(i32::from(e)-10)};
    Ok(if code&128==0{v}else{-v})
}
pub fn bf16(x:f32)->f32 {
    let u=x.to_bits();let r=if u&0x7f800000==0x7f800000 {u}else{u.wrapping_add(0x7fff+((u>>16)&1))};
    f32::from_bits(r&0xffff0000)
}
pub fn mix(mut x:u64)->u64 {
    x=x.wrapping_add(0x9e3779b97f4a7c15);x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^(x>>31)
}
pub fn activation(rows:usize,k:usize)->Vec<f32>{
    (0..rows*k).map(|i|bf16((mix((i as u64)^0x1635a16)%127)as f32/128.0-63.0/128.0)).collect()
}
pub fn packed_fixture(n:usize,k:usize)->(Vec<u8>,Vec<u8>){
    assert!(n>0&&n%128==0&&k>0&&k%64==0);
    let mut q=vec![0;n*k/2];let mut s=vec![0;n*k/16];
    const SF:[u8;8]=[0x28,0x2f,0x30,0x35,0x38,0x3b,0x3e,0x40];
    for nb in 0..n/128 {for kb in 0..k/64 {for col in 0..128 {
        let row=nb*128+col;let tile=(nb*(k/64)+kb)*128+col;
        for pair in 0..32 {let x=(row as u64).wrapping_mul(0x100000001b3)^(kb*64+pair*2)as u64;
            let lo=(mix(x^0xe2a1)&15)as u8;let hi=(mix((x+1)^0xe2a1)&15)as u8;q[tile*32+pair]=lo|(hi<<4);
        }
        for group in 0..4 {s[tile*4+group]=SF[(mix((row*17+kb*13+group)as u64)&7)as usize];}
    }}}
    (q,s)
}
pub fn weight(n:usize,k:usize,col:usize,inner:usize,q:&[u8],s:&[u8])->Result<f64,String>{
    if n==0||n%128!=0||k==0||k%64!=0||col>=n||inner>=k||q.len()!=n*k/2||s.len()!=n*k/16{return Err("packed weight geometry mismatch".into());}
    let nb=col/128;let kb=inner/64;let ni=col%128;let ki=inner%64;
    let packed_row=(nb*(k/64)+kb)*128+ni;
    let byte=q[packed_row*32+ki/2];let code=if ki%2==0 {byte&15}else{byte>>4};
    Ok(fp4(code)*e4m3(s[packed_row*4+ki/16])?)
}
pub struct Oracle {pub output:Vec<f32>,pub max_abs_dot:f64,pub max_l1_dot:f64}
/// Compute all4 rows at once for reuse across M1..4 and all tile configurations.
/// Dense dot is accumulated independently in F64. Cast once toF32, multiply
/// original device-global F32 scale, then BF16 RNE (same required scale stage).
pub fn evaluate(n:usize,k:usize,rows:usize,a:&[f32],q:&[u8],s:&[u8],global:f32)->Result<Oracle,String>{
    if n==0||n%128!=0||k==0||k%64!=0||!(1..=4).contains(&rows)||a.len()!=rows*k||q.len()!=n*k/2||s.len()!=n*k/16
        ||!global.is_finite()||a.iter().any(|x|!x.is_finite()||x.to_bits()&0xffff!=0){return Err("invalid independent GEMV reference arguments".into());}
    if s.iter().any(|x|x&0x7f==0x7f){return Err("E4M3 NaN scale outside finite reference domain".into());}
    let mut sf=[0.0f64;256];for code in 0u8..=255 {sf[code as usize]=e4m3(code).unwrap_or(0.0);}
    let mut out=vec![0.;rows*n];let mut max_abs_dot=0.0f64;let mut max_l1_dot=0.0f64;
    for col in 0..n {
        let mut sums=[0.0f64;4];let mut l1=[0.0f64;4];
        for kb in 0..k/64 {
            let base=((col/128)*(k/64)+kb)*128+col%128;
            for pair in 0..32 {
                let byte=q[base*32+pair];let scale=sf[s[base*4+pair/8]as usize];
                let w0=fp4(byte&15)*scale;let w1=fp4(byte>>4)*scale;let index=kb*64+pair*2;
                for row in 0..rows {
                    let p0=f64::from(a[row*k+index])*w0;let p1=f64::from(a[row*k+index+1])*w1;
                    sums[row]+=p0;sums[row]+=p1;l1[row]+=p0.abs()+p1.abs();
                }
            }
        }
        for row in 0..rows {max_abs_dot=max_abs_dot.max(sums[row].abs());max_l1_dot=max_l1_dot.max(l1[row]);out[row*n+col]=bf16(sums[row]as f32*global);}
    }
    Ok(Oracle{output:out,max_abs_dot,max_l1_dot})
}
pub fn argmax(row:&[f32])->usize{
    let mut best=0;for i in 1..row.len(){if row[i]>row[best]{best=i;}}best
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn all_4064_finite_fp4_scale_products_are_exact_bf16(){
        let mut count=0;let mut max=0.0f64;let mut min=f64::INFINITY;
        for code in 0u8..16 {for sf in 0u8..=255 {if let Ok(scale)=e4m3(sf){
            let exact=fp4(code)*scale;let v=exact as f32;assert_eq!(f64::from(v),exact);assert_eq!(v.to_bits()&0xffff,0,"code{code} sf{sf}");
            assert_eq!(bf16(v).to_bits(),v.to_bits());count+=1;max=max.max(exact.abs());if exact!=0.0 {min=min.min(exact.abs());}
        }}}
        assert_eq!(count,4064);assert_eq!(min,2.0f64.powi(-10));assert_eq!(max,2688.0);
    }
    #[test]fn known_codes_nibbles_and_four_block_groups_are_independent(){
        assert_eq!(fp4(5),3.0);assert_eq!(fp4(10),-1.0);assert_eq!(e4m3(0x38).unwrap(),1.0);assert!(e4m3(0x7f).is_err());
        let n=128;let k=128;let mut q=vec![0;n*k/2];let mut s=vec![0;n*k/16];
        for kb in 0..2 {for col in 0..n {let base=kb*128+col;for pair in 0..32{q[base*32+pair]=0xa5;}for g in 0..4{s[base*4+g]=[0x28,0x30,0x38,0x40][g];}}}
        for col in [0,63,127]{for i in 0..k{let expected=(if i%2==0 {3.0}else{-1.0})*[0.25,0.5,1.0,2.0][(i%64)/16];assert_eq!(weight(n,k,col,i,&q,&s).unwrap(),expected);}}
    }
    #[test]fn global_scale_must_not_be_rounded_into_bf16_weights(){
        let alpha=1.1f32;let mut exact_dot=0.0f32;let mut wrong_dot=0.0f32;
        for _ in 0..16 {for w in [3.0f32,-1.0,-1.0,-1.0]{exact_dot+=w;wrong_dot+=bf16(w*alpha);}}
        assert_eq!(bf16(exact_dot*alpha),0.0);assert_eq!(wrong_dot,-0.125);assert!((wrong_dot-exact_dot).abs()>0.01);
    }
    #[test]fn explicit_second_nblock_second_kblock_high_nibble_beacon(){
        let(n,k)=(256,192);let mut q=vec![0;n*k/2];let mut sf=vec![0;n*k/16];
        // col129 -> Nblock1,row1; k65 -> Kblock1,high nibble1.
        // ((1*3+1)*128+1)=513 is a hand-declared physical row.
        q[513*32]=0x50;sf[513*4]=0x30;
        let mut a=vec![0.;4*k];for row in 0..4{a[row*k+[0,63,64,65][row]]=1.;}
        let r=evaluate(n,k,4,&a,&q,&sf,1.).unwrap();assert_eq!(r.output[3*n+129],1.5);
        assert_eq!(r.output.iter().filter(|x|**x!=0.0).count(),1);
    }
    #[test]fn constant_and_one_hot_have_analytic_answers_all_rows(){
        let n=128;let k=64;let q=vec![0x11;n*k/2];let s=vec![0x38;n*k/16];
        for rows in 1..=4{let a=vec![1.0;rows*k];let r=evaluate(n,k,rows,&a,&q,&s,1.0).unwrap();assert!(r.output.iter().all(|x|*x==32.0));
            let mut a=vec![0.;rows*k];for row in 0..rows{a[row*k+[0,15,16,63][row]]=1.0;}let r=evaluate(n,k,rows,&a,&q,&s,1.0).unwrap();assert!(r.output.iter().all(|x|*x==0.5));}
    }
}
