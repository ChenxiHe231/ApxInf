//! CPU-only binary container/metrics; contains no CUDA or GDN recurrence.
use std::fs;
use std::io::{self, Write};
use std::path::Path;
pub const KH: usize = 16;
pub const VH: usize = 32;
pub const D: usize = 128;
pub const CW: usize = 4;
pub const QW: usize = KH * D;
pub const VW: usize = VH * D;
pub const QKV: usize = QW * 2 + VW;
pub const EPS: f32 = 1.0e-6;
pub const SECTIONS: usize = 19;
pub const MAGIC: &[u8; 8] = b"GDNI0001";

pub fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3))
}
pub fn count_kind(steps: usize, id: usize) -> (usize, usize) {
    let rows = steps + 1;
    match id {
        0 => (rows*QKV,2), 1 => (rows*VW,2), 2|3 => (rows*VH,2),
        4 => (QKV*CW,2), 5|6 => (VH,2), 7 => (D,2),
        8|15|17 => (QKV*CW,4), 9|16|18 => (VH*D*D,4),
        10 => (rows*QKV,2), 11|12 => (rows*VH,4), 13|14 => (rows*VW,2),
        _ => panic!("unknown section"),
    }
}
pub struct Bundle { pub steps: usize, pub arrays: Vec<Vec<u8>> }
impl Bundle {
    pub fn validate(&self) {
        assert!([64,128,2048].contains(&self.steps), "undeclared independent-math length");
        assert_eq!(self.arrays.len(),SECTIONS);
        for (id,a) in self.arrays.iter().enumerate() {
            let (count,kind)=count_kind(self.steps,id); assert_eq!(a.len(),count*kind,"section{id}");
            if kind==2 { assert!(a.chunks_exact(2).all(|b|u16::from_le_bytes([b[0],b[1]])&0x7f80!=0x7f80),"nonfinite BF16 section{id}"); }
            else { assert!(a.chunks_exact(4).all(|b|f32::from_le_bytes(b.try_into().unwrap()).is_finite()),"nonfinite F32 section{id}"); }
        }
    }
    pub fn write_new(&self,path:&Path)->io::Result<()> {
        self.validate(); let mut out=fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        out.write_all(MAGIC)?;
        for v in [self.steps as u32,KH as u32,VH as u32,D as u32,CW as u32,EPS.to_bits(),SECTIONS as u32] {out.write_all(&v.to_le_bytes())?;}
        for (id,a) in self.arrays.iter().enumerate() {
            let (count,kind)=count_kind(self.steps,id);
            out.write_all(&(id as u32).to_le_bytes())?;out.write_all(&(kind as u32).to_le_bytes())?;
            out.write_all(&(count as u64).to_le_bytes())?;out.write_all(&fnv(a).to_le_bytes())?;out.write_all(a)?;
        }
        out.sync_all()
    }
    pub fn read(path:&Path)->io::Result<Self> {
        assert!(fs::metadata(path)?.len()<=256*1024*1024,"oversized bundle");
        let data=fs::read(path)?; assert!(data.len()<=256*1024*1024,"oversized bundle");
        let mut at=0usize;
        fn take<'a>(b:&'a[u8],at:&mut usize,n:usize)->&'a[u8] {
            let end=at.checked_add(n).expect("extent overflow"); assert!(end<=b.len(),"truncated bundle");
            let a=&b[*at..end];*at=end;a
        }
        assert_eq!(take(&data,&mut at,8),MAGIC);
        let mut h=[0u32;7];for x in &mut h {*x=u32::from_le_bytes(take(&data,&mut at,4).try_into().unwrap());}
        assert_eq!(&h[1..],&[KH as u32,VH as u32,D as u32,CW as u32,EPS.to_bits(),SECTIONS as u32]);
        let steps=h[0] as usize; assert!([64,128,2048].contains(&steps));
        let mut arrays=Vec::new();
        for expected_id in 0..SECTIONS {
            let id=u32::from_le_bytes(take(&data,&mut at,4).try_into().unwrap()) as usize;
            let kind=u32::from_le_bytes(take(&data,&mut at,4).try_into().unwrap()) as usize;
            let count=u64::from_le_bytes(take(&data,&mut at,8).try_into().unwrap()) as usize;
            let hash=u64::from_le_bytes(take(&data,&mut at,8).try_into().unwrap());
            assert_eq!(id,expected_id);assert_eq!((count,kind),count_kind(steps,id));
            let a=take(&data,&mut at,count.checked_mul(kind).unwrap()).to_vec();assert_eq!(fnv(&a),hash,"section{id} changed");arrays.push(a);
        }
        assert_eq!(at,data.len(),"trailing bytes");let b=Self{steps,arrays};b.validate();Ok(b)
    }
    pub fn f32_values(&self,id:usize)->Vec<f32> {
        let (_,kind)=count_kind(self.steps,id);
        if kind==2 {self.arrays[id].chunks_exact(2).map(|b|f32::from_bits((u16::from_le_bytes([b[0],b[1]]) as u32)<<16)).collect()}
        else {self.arrays[id].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect()}
    }
}
pub fn u16_bytes(values:Vec<u16>)->Vec<u8>{values.into_iter().flat_map(u16::to_le_bytes).collect()}
pub fn f32_bytes(values:Vec<f32>)->Vec<u8>{values.into_iter().flat_map(f32::to_le_bytes).collect()}

#[derive(Debug)]
pub struct Metrics {pub cosine:f64,pub rel:f64,pub maxabs:f64,pub worst_row_rel:f64,pub worst_row_cos:f64,pub worst_row:usize}
// v2 changes measurement only: exact zero-norm cases, no epsilon floors.
// Each finite F32 is converted to F64 before multiplication/subtraction.
// At the container's <=256MiB extent (indeed at any addressable slice length),
// squared values and their sums cannot overflow/underflow F64.
fn relative_error(error_sq: f64, want_sq: f64) -> f64 {
    if want_sq == 0.0 {
        if error_sq == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (error_sq / want_sq).sqrt()
    }
}
fn cosine_similarity(dot: f64, got_sq: f64, want_sq: f64) -> f64 {
    if got_sq == 0.0 && want_sq == 0.0 {
        1.0
    } else if got_sq == 0.0 || want_sq == 0.0 {
        0.0
    } else {
        dot / (got_sq.sqrt() * want_sq.sqrt())
    }
}
pub fn metrics(got: &[f32], want: &[f32], row_width: usize) -> Metrics {
    assert_eq!(got.len(), want.len());
    assert!(!got.is_empty(), "empty measurements cannot certify precision");
    assert!(row_width > 0 && got.len() % row_width == 0);
    assert!(got.iter().chain(want).all(|x| x.is_finite()));
    let mut total = [0.0f64; 4];
    let mut m = Metrics { cosine: 0.0, rel: 0.0, maxabs: 0.0,
        worst_row_rel: 0.0, worst_row_cos: 1.0, worst_row: 0 };
    for (row, (ga, wa)) in got.chunks_exact(row_width)
        .zip(want.chunks_exact(row_width)).enumerate() {
        let mut r = [0.0f64; 4];
        for (&g, &w) in ga.iter().zip(wa) {
            let (g, w) = (g as f64, w as f64);
            r[0] += g*w;
            r[1] += g*g;
            r[2] += w*w;
            r[3] += (g-w)*(g-w);
            m.maxabs = m.maxabs.max((g-w).abs());
        }
        for i in 0..4 { total[i] += r[i]; }
        let rel = relative_error(r[3], r[2]);
        let cos = cosine_similarity(r[0], r[1], r[2]);
        if rel > m.worst_row_rel { m.worst_row_rel = rel; m.worst_row = row; }
        m.worst_row_cos = m.worst_row_cos.min(cos);
    }
    m.rel = relative_error(total[3], total[2]);
    m.cosine = cosine_similarity(total[0], total[1], total[2]);
    m
}
