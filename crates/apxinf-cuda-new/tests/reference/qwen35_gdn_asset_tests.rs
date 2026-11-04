//! CPU-only tests of the isolated measurement-v2 asset module. No CUDA, process spawning, or network.
//! Run in an owned feature devlocal directory; see parser-metric-README.md.
#[path = "../support/qwen35_gdn_math_asset.rs"]
mod asset;
use asset::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct OwnedTemp { root: PathBuf }
impl OwnedTemp {
    fn new() -> Self {
        let root = PathBuf::from(std::env::var_os("APXINF_GDN_MATH_TEST_TMP")
            .expect("set APXINF_GDN_MATH_TEST_TMP to an existing feature devlocal directory"));
        assert!(root.is_absolute());
        let root = root.canonicalize().unwrap();
        assert!(root.components().any(|c| c.as_os_str() == "devlocal"),
                "temporary files must remain under a feature devlocal directory");
        let root = root.join(format!("parser-metric-{}-{}", std::process::id(),
                                    NEXT.fetch_add(1, Ordering::Relaxed)));
        fs::create_dir(&root).unwrap(); // refuse existing directories
        Self { root }
    }
    fn file(&self, name: &str) -> PathBuf { self.root.join(name) }
}
impl Drop for OwnedTemp {
    fn drop(&mut self) { fs::remove_dir_all(&self.root).unwrap(); }
}
fn rejects(path: &Path) -> bool {
    match std::panic::catch_unwind(|| Bundle::read(path)) {
        Err(_) => true,
        Ok(result) => result.is_err(),
    }
}
fn must_reject(path: &Path, label: &str) {
    assert!(rejects(path), "invalid container accepted: {label}");
    println!("CPU-PARSER-NEGATIVE rejected={label}");
}
fn patch_u32(b: &mut [u8], at: usize, x: u32) { b[at..at+4].copy_from_slice(&x.to_le_bytes()); }
fn patch_u64(b: &mut [u8], at: usize, x: u64) { b[at..at+8].copy_from_slice(&x.to_le_bytes()); }
fn sections() -> Vec<(usize, usize, usize)> {
    let mut at = 36;
    (0..SECTIONS).map(|id| {
        let (count, kind) = count_kind(64, id);
        let d = (at, at+24, count*kind);
        at += 24+count*kind;
        d
    }).collect()
}
fn write_and_reject(path: &Path, bytes: &[u8], label: &str) {
    fs::write(path, bytes).unwrap();
    must_reject(path, label);
}
#[test]
fn parser_accepts_actual_57_sections() {
    let root = PathBuf::from(std::env::var_os("APXINF_GDN_MATH_ASSET_DIR")
        .expect("set APXINF_GDN_MATH_ASSET_DIR to the frozen real Orin CPU asset directory"));
    assert!(root.is_absolute());
    // Frozen fingerprints of the actual assets, not regenerated mathematical output.
    // SHA256 remains externally verified by actual-assets-manifest.json.
    let inventory: [(usize, u64, [u64; SECTIONS]); 3] = FROZEN_INVENTORY;
    let mut total = 0;
    for (steps, size, expected) in inventory {
        let path = root.join(format!("gdn-math-{steps}.bin"));
        assert_eq!(fs::metadata(&path).unwrap().len(), size);
        let b = Bundle::read(&path).unwrap();
        assert_eq!(b.steps, steps);
        assert_eq!(b.arrays.len(), SECTIONS);
        for (id, raw) in b.arrays.iter().enumerate() {
            let (count, kind) = count_kind(steps, id);
            assert_eq!(raw.len(), count*kind);
            assert_eq!(fnv(raw), expected[id], "actual asset changed steps={steps} section={id}");
            println!("CPU-PARSER-ACTUAL steps={steps} section={id} elements={count} kind={kind} fnv={:016x} finite=true", expected[id]);
            total += 1;
        }
    }
    assert_eq!(total, 57);
    println!("CPU-PARSER-ACTUAL total_sections=57 matching_frozen_fnv=true gpu_run=false");
}
#[test]
fn parser_rejects_corrupt_container() {
    let temp = OwnedTemp::new();
    let file = temp.file("small64.bin");
    // 64 is the smallest allowed shape. One synthetic zero-valued, finite 10.5MB
    // parser fixture is reused; this is not a new mathematical oracle asset.
    let b = Bundle { steps: 64, arrays: (0..SECTIONS).map(|id| {
        let (n, k) = count_kind(64, id); vec![0; n*k]
    }).collect() };
    b.write_new(&file).unwrap();
    drop(b);
    assert!(!rejects(&file));
    let original = fs::read(&file).unwrap();
    let descriptors = sections();
    assert_eq!(descriptors.last().map(|d| d.1+d.2), Some(original.len()));
    let mut mutations = 0;
    for len in [0, 7, 35, 59, 61, original.len()-1] {
        write_and_reject(&file, &original[..len], &format!("truncation-{len}"));
        mutations += 1;
    }
    for (offset, value, label) in [
        (0, 0u32, "magic"), (12, 15, "header-geometry"),
        (28, EPS.to_bits()+1, "header-epsilon"), (32, 18, "header-section-count"),
        (8, 63, "unsupported-steps"), (40, 4, "section-kind"), (36, 1, "section-order"),
    ] {
        let mut changed = original.clone();
        patch_u32(&mut changed, offset, value);
        write_and_reject(&file, &changed, label); mutations += 1;
    }
    let mut changed = original.clone();
    patch_u64(&mut changed, 44, count_kind(64, 0).0 as u64+1);
    write_and_reject(&file, &changed, "section-count"); mutations += 1;
    let mut changed = original.clone(); changed[52] ^= 1;
    write_and_reject(&file, &changed, "stored-FNV"); mutations += 1;
    let mut changed = original.clone(); changed[60] ^= 1;
    write_and_reject(&file, &changed, "payload-FNV-mismatch"); mutations += 1;
    // Recompute FNV after injecting nonfinite data, so the finite-domain
    // guard is tested independently rather than failing only checksum validation.
    for (id, value, label) in [
        (0, 0x7f80u32, "BF16-inf"), (0, 0x7fc0, "BF16-NaN"),
        (8, 0x7f800000, "F32-inf"), (8, 0x7fc00000, "F32-NaN"),
    ] {
        let mut changed = original.clone();
        let (descriptor, payload, len) = descriptors[id];
        let (_, kind) = count_kind(64, id);
        changed[payload..payload+kind].copy_from_slice(&value.to_le_bytes()[..kind]);
        let hash = fnv(&changed[payload..payload+len]);
        patch_u64(&mut changed, descriptor+16, hash);
        write_and_reject(&file, &changed, label); mutations += 1;
    }
    let mut changed = original.clone(); changed.push(0);
    write_and_reject(&file, &changed, "trailing-byte"); mutations += 1;
    // Sparse metadata-sized file: no giant payload is generated or read.
    let oversized = temp.file("oversized-sparse.bin");
    fs::File::create(&oversized).unwrap().set_len(256*1024*1024+1).unwrap();
    must_reject(&oversized, "oversized-metadata"); mutations += 1;
    println!("CPU-PARSER-NEGATIVE total_rejected={mutations} gpu_run=false");
    assert_eq!(mutations, 22);
}
fn close(got: f64, want: f64) { assert!((got-want).abs() <= 1e-13, "{got} != {want}"); }
#[test]
fn metric_exact_identity_normal_range() {
    let v = [0.0, -0.0, 1.0, -2.0, 0.25, -0.5, 3.5, -4.0];
    let m = metrics(&v, &v, 4);
    close(m.cosine, 1.0); close(m.worst_row_cos, 1.0);
    assert_eq!((m.rel, m.maxabs, m.worst_row_rel), (0.0, 0.0, 0.0));
}
#[test]
fn metric_zero_norm_contract() {
    let zz = metrics(&[0.0, -0.0], &[0.0, -0.0], 2);
    assert_eq!((zz.cosine, zz.rel, zz.maxabs, zz.worst_row_cos), (1.0, 0.0, 0.0, 1.0));
    let zw = metrics(&[0.0, 0.0], &[1.0, -1.0], 2);
    assert_eq!((zw.cosine, zw.rel, zw.maxabs), (0.0, 1.0, 1.0));
    let wz = metrics(&[1.0, -1.0], &[0.0, 0.0], 2);
    assert_eq!((wz.cosine, wz.maxabs), (0.0, 1.0));
    assert_eq!(wz.rel, f64::INFINITY, "nonzero error / zero reference norm");
}
#[test]
fn metric_signed_error_exact_examples() {
    let scale = metrics(&[2.0, -2.0], &[1.0, -1.0], 2);
    close(scale.cosine, 1.0); close(scale.rel, 1.0); close(scale.maxabs, 1.0);
    let opposite = metrics(&[-1.0, 1.0], &[1.0, -1.0], 2);
    close(opposite.cosine, -1.0); close(opposite.rel, 2.0); close(opposite.maxabs, 2.0);
}
#[test]
fn metric_single_bad_token_cannot_hide_in_global_cosine() {
    let want = vec![1.0; 2048*4];
    let mut got = want.clone(); got[917*4..918*4].fill(-1.0);
    let m = metrics(&got, &want, 4);
    assert!(m.cosine >= 0.999 && m.rel <= 0.05,
            "fixture must demonstrate the frozen global gate can miss this single bad token");
    assert_eq!(m.worst_row, 917);
    close(m.worst_row_rel, 2.0); close(m.worst_row_cos, -1.0);
    assert!(!(m.worst_row_cos >= 0.999 && m.worst_row_rel <= 0.05));
    println!("CPU-METRIC one_bad_token global_cos={} global_rel={} row={} row_cos={} row_rel={} row_gate_rejects=true",
             m.cosine,m.rel,m.worst_row,m.worst_row_cos,m.worst_row_rel);
}
#[test]
fn metric_nonfinite_and_shape_rejected() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(std::panic::catch_unwind(|| metrics(&[nonfinite], &[1.0], 1)).is_err());
        assert!(std::panic::catch_unwind(|| metrics(&[1.0], &[nonfinite], 1)).is_err());
    }
    assert!(std::panic::catch_unwind(|| metrics(&[1.0], &[1.0,2.0], 1)).is_err());
    assert!(std::panic::catch_unwind(|| metrics(&[1.0], &[1.0], 0)).is_err());
    assert!(std::panic::catch_unwind(|| metrics(&[1.0], &[1.0], 2)).is_err());
}
// These mathematical expectations expose v1 defects and must pass v2.
// No should_panic/ignore annotation hides a failure.
#[test]
fn metric_v2_fixed_tiny_nonzero_identity_should_have_cosine_one() {
    let x = f32::MIN_POSITIVE;
    let m = metrics(&[x], &[x], 1);
    println!("CPU-METRIC-V2-FIX tiny identity got_cos={} mathematical_cos=1",m.cosine);
    close(m.cosine, 1.0);
}
#[test]
fn metric_v2_fixed_tiny_opposite_should_have_relative_error_two() {
    let x = f32::MIN_POSITIVE;
    let m = metrics(&[-x], &[x], 1);
    println!("CPU-METRIC-V2-FIX tiny opposite got_rel={} mathematical_rel=2",m.rel);
    close(m.rel, 2.0);
}
#[test]
fn metric_v2_fixed_empty_input_should_be_rejected() {
    assert!(std::panic::catch_unwind(|| metrics(&[], &[], 1)).is_err());
}
#[test]
fn metric_v2_minimum_f32_subnormal_and_tiny_identity() {
    for x in [f32::from_bits(1), f32::from_bits(2), f32::MIN_POSITIVE,
              f32::from_bits(0x00010000), 1.0e-20] {
        let identity = metrics(&[x,-x], &[x,-x], 1);
        close(identity.cosine, 1.0); close(identity.worst_row_cos, 1.0);
        assert_eq!((identity.rel,identity.maxabs,identity.worst_row_rel),(0.0,0.0,0.0));
        let opposite = metrics(&[-x,x], &[x,-x], 1);
        close(opposite.cosine,-1.0); close(opposite.rel,2.0);
        close(opposite.worst_row_cos,-1.0); close(opposite.worst_row_rel,2.0);
    }
}
#[test]
fn metric_v2_maximum_f32_domain_does_not_overflow() {
    let x=f32::MAX;
    let identity=metrics(&[x,-x,x,-x],&[x,-x,x,-x],2);
    close(identity.cosine,1.0);close(identity.worst_row_cos,1.0);
    assert_eq!((identity.rel,identity.maxabs),(0.0,0.0));
    let opposite=metrics(&[-x,x,-x,x],&[x,-x,x,-x],2);
    close(opposite.cosine,-1.0);close(opposite.rel,2.0);
    assert_eq!(opposite.maxabs,2.0*(x as f64));
    assert!(opposite.maxabs.is_finite());
}
const FROZEN_INVENTORY: [(usize,u64,[u64;SECTIONS]);3] = [
    (64,10503404,[0x8e40ab48a83fefebu64,0xd3ac784366c67ab2u64,0x7498a8cb2e1a5e0au64,0x60f42b5f4d26a77fu64,0x67ccd9289eb66facu64,0xca840c6e1a9b3b84u64,0xdd1fb0851d2bb664u64,0xbdbe39a9c1fae042u64,0xf25dc328fa9b0085u64,0xc1d82745395b8440u64,0x28946f7b1551ec4cu64,0xb0dbbda798aa2d23u64,0x60b9042448986de2u64,0xa2fec7fe2605bccau64,0xc727bbab56d77d8au64,0xd4d1366ee09a8774u64,0x8c12ae890987243au64,0xcf498fb339730831u64,0xab9fc742ebed0ac2u64]),
    (128,14197996,[0xcff692e786b45b84u64,0xfe476455e19108d7u64,0x28ab6792f0e03552u64,0x0d985526e8a9c84au64,0x67ccd9289eb66facu64,0xca840c6e1a9b3b84u64,0xdd1fb0851d2bb664u64,0xbdbe39a9c1fae042u64,0xf25dc328fa9b0085u64,0xc1d82745395b8440u64,0xb0d4fc13b2023b35u64,0x3c11ec92d2460eb9u64,0xfdbcbbffd99dbf52u64,0x7d9f4418254d48dau64,0x83806fd1dbd9968bu64,0x2b989ddba37dd2c2u64,0x19166dcd8911cc71u64,0x8cc2fedf7680503du64,0xe89fbdf579fc1947u64]),
    (2048,125035756,[0x0dec86863b588b57u64,0xb8613a678465647fu64,0xef82d5f68e3f77dfu64,0xde2abf615c427137u64,0x67ccd9289eb66facu64,0xca840c6e1a9b3b84u64,0xdd1fb0851d2bb664u64,0xbdbe39a9c1fae042u64,0xf25dc328fa9b0085u64,0xc1d82745395b8440u64,0x815ea272c3dfa366u64,0x92a222cbd6ff1faau64,0x9df21019a1b06c13u64,0x5bbd10ca4786d599u64,0x1cfcf88d9822eea2u64,0xcc7c3e15eed35a5fu64,0xa9b157441c2c33c9u64,0x735c877c318f7b55u64,0xc47f8e3a2c68cb30u64]),
];
