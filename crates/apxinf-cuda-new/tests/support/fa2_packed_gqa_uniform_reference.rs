//! Independent analytic raw-attention fixture: Q/K0 makes every exp weight1.
//! This isolates causal/layout/GQA errors. It DOES NOT certify nonzero softmax
//! P-BF16 rounding, arbitrary within-KV-group Q-head permutation, or real models.
pub const QH:usize=16;pub const KVH:usize=2;pub const D:usize=256;
pub const PREFIXES:[usize;2]=[40960,49152];
pub const FROZEN_MAX_ABS:f64=0.01;
const PAT:[i32;8]=[-7,-5,-3,-1,1,3,5,7];
#[derive(Clone,Copy,Debug)]pub struct Fixture{pub prefix:usize,pub queries:usize,pub binding_b:bool}
impl Fixture{
    pub fn new(prefix:usize,queries:usize,binding_b:bool)->Self{
        //128 is only the cheap CPU scalar-oracle cross-check, not a declared
        //.01 BF16 GPU domain. The long GPU gate is PREFIXES x T1..4 only.
        assert!([128,40960,49152].contains(&prefix)&&(1..=4).contains(&queries));
        Self{prefix,queries,binding_b}
    }
    pub fn valid(&self)->usize{self.prefix+self.queries}
    pub fn capacity(&self)->usize{(self.valid()+65).div_ceil(128)*128}
    pub fn coefficient(&self,row:usize)->f64{
        assert!(row<self.queries);if row+1==self.queries{8.0}else{[1.0,-2.0,3.0][row]}
    }
    pub fn base(&self,kv:usize,dim:usize)->f64{
        assert!(kv<KVH&&dim<D);let code=(dim%4)as f64/16.0;
        (if kv==0{0.25+code}else{-0.5-code})+if self.binding_b{0.125}else{0.0}
    }
    pub fn value_at(&self,position:usize,kv:usize,dim:usize)->f32{
        assert!(position<self.capacity());
        if position<self.prefix{(self.base(kv,dim)+PAT[position%8]as f64/16.0)as f32}
        else if position<self.valid(){(self.prefix as f64*self.coefficient(position-self.prefix))as f32}
        else{(self.prefix*16)as f32} //finite future poison; never mathematically visible
    }
    fn prefix_sum(&self,count:usize,kv:usize,dim:usize)->f64{
        assert!(count<=self.prefix);
        count as f64*self.base(kv,dim)+PAT[..count%8].iter().map(|x|*x as f64/16.0).sum::<f64>()
    }
    pub fn visible_mean(&self,visible:usize,kv:usize,dim:usize)->f64{
        assert!(visible>0&&visible<=self.capacity());let mut sum=self.prefix_sum(visible.min(self.prefix),kv,dim);
        for row in 0..visible.saturating_sub(self.prefix).min(self.queries){sum+=self.prefix as f64*self.coefficient(row);}
        if visible>self.valid(){sum+=(visible-self.valid())as f64*(self.prefix*16)as f64;}
        sum/visible as f64
    }
    pub fn expected_element(&self,row:usize,head:usize,dim:usize)->f64{
        assert!(row<self.queries&&head<QH);self.visible_mean(self.prefix+row+1,head/8,dim)
    }
    /// All T*16*256 elements, ideal prepared-domain attention, no output gate.
    /// Compare device BF16 against this f32 interface with the frozen .01 ceiling.
    pub fn expected_output(&self)->Vec<f32>{
        (0..self.queries).flat_map(|r|(0..QH).flat_map(move|h|(0..D).map(move|d|self.expected_element(r,h,d)as f32))).collect()
    }
    /// F64 ideal logsumexp diagnostic, [Hq,T] layout. No new LSE tolerance.
    pub fn expected_lse(&self)->Vec<f32>{(0..QH).flat_map(|_|(0..self.queries).map(|r|((self.prefix+r+1)as f64).ln()as f32)).collect()}
    pub fn max_correct_storage_error(&self)->f64{
        let mut max=0.0f64;for r in 0..self.queries{for h in 0..QH{for d in 0..D{let x=self.expected_element(r,h,d);max=max.max((numeric_bf16(x)-x).abs());}}}max
    }
    pub fn standard_fp32_exact_sum_abs_bound(&self)->f64{self.prefix as f64*(1.125+(0..self.queries).map(|r|self.coefficient(r).abs()).sum::<f64>())}
}
// Verbatim mathematical body from frozen analytic_contract.rs SHA88e23121...:
// numeric exponent/step nearest ties-even; no CUDA/half bit converter is copied.
pub fn numeric_bf16(x:f64)->f64{
    assert!(x.is_finite());
    if x == 0.0 { return x; }
    let magnitude = x.abs();
    let step = if magnitude < 2.0f64.powi(-126) { 2.0f64.powi(-133) }
        else { 2.0f64.powi(magnitude.log2().floor() as i32 - 7) };
    let units = magnitude / step;
    let lower = units.floor();
    let fraction = units - lower;
    let rounded = if fraction < 0.5 { lower } else if fraction > 0.5 { lower + 1.0 }
        else if lower as u64 % 2 == 0 { lower } else { lower + 1.0 };
    (rounded * step).copysign(x)
}
#[derive(Debug)]pub struct NegativeControl{pub label:&'static str,pub applicable:bool,pub max_abs:f64,pub differing_elements:usize}
pub fn negative_controls(f:Fixture)->Vec<NegativeControl>{
    let variants=["wrong_KV_group","unpatched_FA2_packed_seqlen_mask","m_without_div8_mask","transposed_O_time_unpack","drop_final_large_key","future_last_key_leak","physical_padding_leak","drop_half_prefix_keep_denominator","drop_half_prefix_renormalize","reverse_value_dimension"];
    variants.into_iter().map(|label|{
        let applicable=!(f.queries==1&&matches!(label,"m_without_div8_mask"|"transposed_O_time_unpack"|"future_last_key_leak"));
        let mut max=0.0f64;let mut different=0usize;
        for row in 0..f.queries{for head in 0..QH{for dim in 0..D{
            let kv=head/8;let g=head%8;let m=row*8+g;let n=f.prefix+row+1;let expected=f.expected_element(row,head,dim);
            let got=match label{
                "wrong_KV_group"=>f.visible_mean(n,1-kv,dim),
                "unpatched_FA2_packed_seqlen_mask"=>f.visible_mean((f.prefix+f.queries-8*f.queries+m+1).min(f.valid()),kv,dim),
                "m_without_div8_mask"=>f.visible_mean((f.prefix+m+1).min(f.valid()),kv,dim),
                "transposed_O_time_unpack"=>f.expected_element((g*f.queries+row)/8,head,dim),
                "drop_final_large_key"=>if row+1==f.queries{f.visible_mean(f.valid()-1,kv,dim)}else{expected},
                "future_last_key_leak"=>if row+1<f.queries{f.visible_mean(f.valid(),kv,dim)}else{expected},
                "physical_padding_leak"=>f.visible_mean(f.capacity(),kv,dim),
                "drop_half_prefix_keep_denominator"=>(f.visible_mean(n,kv,dim)*n as f64-f.prefix_sum(f.prefix/2,kv,dim))/n as f64,
                "drop_half_prefix_renormalize"=>(f.visible_mean(n,kv,dim)*n as f64-f.prefix_sum(f.prefix/2,kv,dim))/(n-f.prefix/2)as f64,
                "reverse_value_dimension"=>f.expected_element(row,head,D-1-dim),
                _=>unreachable!(),
            };
            let error=(got-expected).abs();max=max.max(error);different+=usize::from(error>0.01);
        }}}
        NegativeControl{label,applicable,max_abs:max,differing_elements:different}
    }).collect()
}
