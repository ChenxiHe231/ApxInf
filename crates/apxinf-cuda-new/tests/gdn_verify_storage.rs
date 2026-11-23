//! Default readonly-source/prefix verifier against the full-snapshot reference.
use apxinf_core::{DType, Device, Shape, Tensor};
use apxinf_cuda_new::{capture, ops, transfers, CudaBuffer, CudaContext, CudaStream};
const H: usize = 16;
const V: usize = 32;
const D: usize = 128;
const S: usize = V * D * D;
const W: usize = (2 * H + V) * D;
fn raw(t: &Tensor) -> Vec<u8> {
    transfers::to_cpu(t)
        .unwrap()
        .storage()
        .as_cpu()
        .unwrap()
        .to_vec()
}
fn f32s(n: usize, phase: usize, scale: f32, bias: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| (((i * 17 + phase) % 127) as f32 / 127.0 * scale + bias).to_le_bytes())
        .collect()
}
fn bf16s(n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let v = (((i * 13) % 97) as f32 - 48.0) / 512.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}
struct G {
    all: Tensor,
    t: Tensor,
    before: Vec<u8>,
}
impl G {
    fn new(c: &CudaContext, bytes: Vec<u8>, shape: Vec<usize>, dtype: DType) -> Self {
        let mut before = vec![0xa5; bytes.len() + 512];
        before[256..256 + bytes.len()].copy_from_slice(&bytes);
        let host = Tensor::from_raw(
            Shape::new(vec![before.len()]),
            DType::I8,
            Device::Cpu,
            before.clone(),
        )
        .unwrap();
        let b = CudaBuffer::alloc(before.len(), c.device_id()).unwrap();
        let all = b
            .as_tensor(Shape::new(vec![before.len()]), DType::I8)
            .unwrap();
        c.synchronize().unwrap();
        transfers::copy_cpu_to_cuda(&host, &all).unwrap();
        CudaStream::default_stream().synchronize().unwrap();
        let t = b
            .view(256, bytes.len())
            .unwrap()
            .as_tensor(Shape::new(shape), dtype)
            .unwrap();
        Self { all, t, before }
    }
    fn guard(&self) {
        let r = raw(&self.all);
        assert_eq!(&r[..256], &self.before[..256]);
        assert_eq!(&r[r.len() - 256..], &self.before[r.len() - 256..]);
    }
    fn readonly(&self) {
        assert_eq!(raw(&self.all), self.before);
    }
}
#[test]
#[ignore = "exclusive CUDA; real geometry, bitwise, guards, Graph replay and source lifetime"]
fn verifier_readonly_source_prefix_exact() {
    let c = CudaContext::new(0).unwrap();
    for t in [1usize, 2, 3, 4, 8] {
        let q = G::new(&c, bf16s(t * W), vec![t, W], DType::BF16);
        let decay = G::new(&c, f32s(t * V, 3, 0.5, -0.8), vec![t, V], DType::F32);
        let beta = G::new(&c, f32s(t * V, 7, 0.7, 0.1), vec![t, V], DType::F32);
        let shape = vec![V, D, D];
        let initial = f32s(S, 11, 0.05, -0.025);
        let source = G::new(&c, initial.clone(), shape.clone(), DType::F32);
        // Separate-source modes must ignore the destination's previous contents.
        let states: Vec<_> = (0..2)
            .map(|i| {
                G::new(
                    &c,
                    if i == 1 {
                        vec![0xff; S * 4]
                    } else {
                        initial.clone()
                    },
                    shape.clone(),
                    DType::F32,
                )
            })
            .collect();
        let out: Vec<_> = (0..2)
            .map(|_| G::new(&c, vec![0xa5; t * V * D * 2], vec![t, V, D], DType::BF16))
            .collect();
        let cp: Vec<_> = (0..2)
            .map(|_| G::new(&c, vec![0xa5; t * S * 4], vec![t, V, D, D], DType::F32))
            .collect();
        let call = |i: usize| {
            ops::gdn_recurrent_verify_from_source(
                &c,
                &states[i + 1].t,
                &q.t,
                &decay.t,
                &beta.t,
                &out[i + 1].t,
                &cp[i + 1].t,
                t,
                V,
                H,
                [0, H * D, 2 * H * D],
                W,
                &source.t,
            )
        };
        let mut graphs: Vec<_> = (0..1).map(|i| capture(&c, || call(i)).unwrap()).collect();
        for step in 0..32 {
            let src_before = raw(&source.all);
            ops::gdn_recurrent_verify(
                &c,
                &states[0].t,
                &q.t,
                &decay.t,
                &beta.t,
                &out[0].t,
                &cp[0].t,
                t,
                V,
                H,
                [0, H * D, 2 * H * D],
                W,
            )
            .unwrap();
            for i in 0..1 {
                if step % 2 == 0 {
                    call(i).unwrap();
                } else {
                    graphs[i].replay().unwrap();
                }
            }
            c.synchronize().unwrap();
            let sr = raw(&states[0].t);
            let or = raw(&out[0].t);
            let cr = raw(&cp[0].t);
            for i in 0..1 {
                assert_eq!(raw(&states[i + 1].t), sr);
                assert_eq!(raw(&out[i + 1].t), or);
                let cc = raw(&cp[i + 1].t);
                let live = (t - 1) * S * 4;
                assert_eq!(&cc[..live], &cr[..live]);
                assert!(cc[live..].iter().all(|&b| b == 0xa5));
            }
            assert_eq!(raw(&source.all), src_before);
            assert!(sr
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()));
            transfers::copy_cuda_to_cuda(&c, &states[0].t, &source.t).unwrap();
            c.synchronize().unwrap();
        }
        c.synchronize().unwrap();
        q.readonly();
        decay.readonly();
        beta.readonly();
        source.guard();
        for g in states.iter().chain(out.iter()).chain(cp.iter()) {
            g.guard();
        }
        let frozen = raw(&states[1].t);
        let frozen_cp = raw(&cp[1].t);
        let frozen_out = raw(&out[1].t);
        for bad in [
            [usize::MAX, H * D, 2 * H * D],
            [0, usize::MAX, 2 * H * D],
            [0, H * D, usize::MAX],
        ] {
            assert!(ops::gdn_recurrent_verify(
                &c,
                &states[1].t,
                &q.t,
                &decay.t,
                &beta.t,
                &out[1].t,
                &cp[1].t,
                t,
                V,
                H,
                bad,
                W
            )
            .is_err());
            assert!(ops::gdn_recurrent_verify_from_source(
                &c,
                &states[1].t,
                &q.t,
                &decay.t,
                &beta.t,
                &out[1].t,
                &cp[1].t,
                t,
                V,
                H,
                bad,
                W,
                &source.t
            )
            .is_err());
        }
        assert!(ops::gdn_recurrent_verify_from_source(
            &c,
            &states[1].t,
            &q.t,
            &decay.t,
            &beta.t,
            &out[1].t,
            &cp[1].t,
            t,
            V,
            H,
            [0, H * D, 2 * H * D],
            W,
            &states[1].t
        )
        .is_err());
        let overlap = CudaBuffer::from_tensor(&states[1].all)
            .unwrap()
            .view(272, S * 4)
            .unwrap()
            .as_tensor(Shape::new(shape.clone()), DType::F32)
            .unwrap();
        assert!(ops::gdn_recurrent_verify_from_source(
            &c,
            &states[1].t,
            &q.t,
            &decay.t,
            &beta.t,
            &out[1].t,
            &cp[1].t,
            t,
            V,
            H,
            [0, H * D, 2 * H * D],
            W,
            &overlap
        )
        .is_err());
        c.synchronize().unwrap();
        assert_eq!(raw(&states[1].t), frozen);
        assert_eq!(raw(&cp[1].t), frozen_cp);
        assert_eq!(raw(&out[1].t), frozen_out);
        // The captured source-mode graph retains every allocation it references.
        ops::gdn_recurrent_verify(
            &c,
            &states[0].t,
            &q.t,
            &decay.t,
            &beta.t,
            &out[0].t,
            &cp[0].t,
            t,
            V,
            H,
            [0, H * D, 2 * H * D],
            W,
        )
        .unwrap();
        c.synchronize().unwrap();
        let expected = raw(&out[0].t);
        let kept = out[1].t.clone();
        let survivor = graphs.remove(0);
        drop(graphs);
        let _ = call;
        drop(overlap);
        drop(q);
        drop(decay);
        drop(beta);
        drop(source);
        drop(states);
        drop(out);
        drop(cp);
        survivor.replay().unwrap();
        c.synchronize().unwrap();
        assert_eq!(raw(&kept), expected);
        println!("PASS verifier storage T{t}:32 updates,default source+prefix,eager/Graph,exact output/state/prefix,readonly,guards,alias rejection,captured lifetimes");
    }
}
