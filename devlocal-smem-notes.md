# GDN chunk-scan shared-memory reduction -- working notes

Hardware (verified, not re-derived): 20 SMs, 228 KiB shared/SM, 1536 threads/SM,
65536 registers/SM, CUDA 13.2, sm_110. Launch is 48 blocks x 128 threads.

## Baseline (HEAD 40ec495)

    shared 208.75 KiB   registers 163   stack 512 B   spill 0
    blocks/SM = min(floor(228/208.75), reg limit 3) = 1
    e2e 2048-token prompt, second `prefill split`: gdn 2601.38 ms
      (smem-baseline.log, 2026-09-24 11:55, box shared with two other agents;
       the 2333 ms in the task brief was measured on a quieter box)

## Buffer lifetime map (one chunk iteration)

    phase                       q    k    pw      T      M    v/nv/kcd chain
    A load                      W    W    -       -      -    W
    B l2norm                    RW   RW   -       -      -    -
    C cum scan                  -    -    -       -      -    -
    D pairwise decay            -    -    W(dec)  -      -    -
    E ut_system + intra_attn    R    R    R(dec)  W(at)  W    -
    F forward substitution      -    -    W(inv)  -      R    -       <- M dies
    G w = beta*(v - e^cum*kS)   -    R    -       -      -    RW
    H v_new = T_inv @ w         -    -    R(inv)  -      -    RW      <- pw dies
    I out = q@S + intra@v_new   R    -    -       R(at)  -    R       <- q,T die
    J state fold                R    R    -       -      -    R       <- k dies

Aliasing alone buys almost nothing: only M dies early (end of F) and nothing
after it wants 16 KiB. pw is already double-duty (pairwise decay, then the UT
inverse). The real win is that the whole v -> nv/kcd -> v_new chain, 96 KiB of
the 209, never leaves the owning thread's v-dim column once the update is
re-associated:

    v_new = T @ (beta*v) - (T @ (beta*e^cum*k)) @ S      needs kcd across k
          = T @ ( beta * (v - e^cum * (k @ S)) )         all at column d

Same operation count -- the contraction over k just moves in front of the
triangular solve instead of after it. Zero precision cost, 96 KiB freed.
That leaves a 5-buffer floor of 112.75 KiB, so going lower needs a narrower
dtype or a shorter chunk.

## Variant ladder (one TU, selected by APXINF_GDN_SMEM_VARIANT)

v  CHUNK  q/k    matrices  shared      regs  spill(st/ld)  blocks/SM
1  64     f32    f32       112.75 KiB  238   0/0           2   (smem 2, reg 2)
2  32     f32    f32        44.38 KiB  128   240/416       4   (smem 5, reg 4)
3  64     bf16   f32        80.75 KiB  230   0/0           2   (smem 2, reg 2)
4  64     bf16   bf16       56.75 KiB  168   144/288       3   (smem 4, reg 3)
5  32     bf16   f32        28.38 KiB   80   644/1012      6
6  32     f32    f32        44.38 KiB  208   0/0           2   (control)
7  16     f32    f32        19.19 KiB   80   620/928       6
8  32     f32    f32        44.38 KiB  168   20/36         3   (control)

6 / 8 / 2 are the same arithmetic at 2 / 3 / 4 blocks per SM, so they isolate
the occupancy lever from the dtype lever. 3 isolates the accuracy cost of bf16
q/k at unchanged occupancy (still 2 blocks/SM, so any speed change there is
not occupancy).

Correction to the previous run's reasoning: it read "73 KiB => 3 blocks/SM"
off shared memory alone. Registers bind first here -- 3 blocks/SM needs
<= 168 registers/thread (65536 / (3*128), rounded to the 256-per-warp
allocation granularity), and the natural f32 kernel wants 238.

## Measurements

(appended as they land)

### Round 1 -- single shot, 2048-token prompt, second `prefill split` (smem-perf1.log)

    variant                                 blocks/SM   gdn ms
    HEAD 40ec495, separate build                    1   2601.4
    1   64 f32 f32                                  2   4895.5  <- discard
    6   32 f32 f32                                  2   2019.7
    8   32 f32 f32                                  3   1884.6
    2   32 f32 f32                                  4   2224.0
    7   16 f32 f32                                  6   2301.6
    3   64 bf16 f32                                 2   2146.3
    4   64 bf16 bf16                                3   2032.5
    5   32 bf16 f32                                 6   2428.2

Variant 1 ran first in a fresh binary (140 s vs 35 s for every later run) so it
paid the cold GEMM autotune; its number is not comparable and is re-measured in
round 2.

Two things fall out of this and neither matches the previous runs story:

### Round 1 -- single shot, 2048-token prompt, second `prefill split` (smem-perf1.log)

    variant                                 blocks/SM   gdn ms
    HEAD 40ec495, separate build                    1   2601.4
    1   64 f32 f32                                  2   4895.5  <- discard
    6   32 f32 f32                                  2   2019.7
    8   32 f32 f32                                  3   1884.6
    2   32 f32 f32                                  4   2224.0
    7   16 f32 f32                                  6   2301.6
    3   64 bf16 f32                                 2   2146.3
    4   64 bf16 bf16                                3   2032.5
    5   32 bf16 f32                                 6   2428.2

Variant 1 ran first in a fresh binary (140 s vs 35 s for every later run) so it
paid the cold GEMM autotune; its number is not comparable and is re-measured in
round 2.

Two things fall out of this, and neither matches the previous run's story:

  - Occupancy is not monotone. 6 -> 8 -> 2 is the same arithmetic at 2 -> 3 -> 4
    blocks/SM and it goes 2020 -> 1885 -> 2224. Three blocks wins; the fourth
    block costs more in register spill (240 B store / 416 B load, against 20/36
    at three blocks) than it returns in latency hiding. Six blocks (7, 5) is
    worse still.
  - bf16 buys nothing. Variant 4 is the previous run's "v4" shape -- bf16 q/k
    and bf16 matrices, 56.75 KiB, 3 blocks/SM -- and at 2032 ms it is slower
    than variant 8, which is 3 blocks/SM with every buffer still f32. Shared
    memory stops being the binding constraint once the chunk is shortened, so
    there is nothing left for the dtype demotion to buy.

So the lever is the chunk length, not the dtype, and the occupancy it unlocks
is worth about 7% on top (2020 -> 1885), not the bulk of the win.

Round 2 adds variant 0 -- the original 209 KiB kernel, compiled into the same
binary at its original 163 registers -- so the baseline becomes an A/B inside
one build rather than a comparison across two.

### Round 2 -- min of 3-4 rounds, all variants in ONE binary (smem-perf2.log)

    v  shape                 shared      blk/SM  gdn ms (min)   vs v0
    0  chunk64 f32 original  208.75 KiB     1       2535.8      1.00x
    1  chunk64 f32           112.75 KiB     2       2713.7      0.93x
    6  chunk32 f32            44.38 KiB     2       1971.0      1.29x
    8  chunk32 f32            44.38 KiB     3       1872.5      1.35x
    2  chunk32 f32            44.38 KiB     4       2246.6      1.13x
    4  chunk64 bf16/bf16      56.75 KiB     3       1990.4      1.27x
    9  chunk32 f32/bf16       38.38 KiB     3       1882.6      1.35x

Spread within a variant is 1-3%, so the ordering is solid.

The headline is variant 1. It halves the shared block with no rounding change
and doubles blocks per SM, which is the entire premise of this task, and it is
**7% slower than the 209 KiB original**. Occupancy was not the thing to buy.

What is actually expensive is the chunk length. Holding blocks/SM fixed at 2,
going from chunk 64 (v1, 2714 ms) to chunk 32 (v6, 1971 ms) is 1.38x on its
own. The scan's per-chunk cost is dominated by terms that grow faster than the
chunk count falls: the forward substitution is O(C^3) over C columns, so O(C^2)
per thread per chunk and O(L*C) over the sequence, and the ut/attn product is
O(C^2 * D) per chunk, O(L*C*D) over the sequence. Both halve when C halves.
Only the state fold, O(D^2) per chunk, doubles, and it is the small term.
Shortening the chunk is a real arithmetic reduction, not a scheduling trick;
the shared-memory saving is a side effect of it.

Occupancy is then worth a further 5% on top and is not monotone: 2 -> 3 -> 4
blocks/SM at chunk 32 goes 1971 -> 1873 -> 2247. The fourth block has to fit
in 128 registers/thread and pays 240 B of spill stores per thread; the third
fits in 168 with 20 B. Past three blocks the spill traffic costs more than the
extra latency hiding returns.

### Accuracy, against the layer-0 reference tensors (smem-acc2.log)

Note: `chunk_scan_matches_reference_tensors_layer0` silently `return`s when it
cannot find `devlocal/qwen38-nvfp4/reference-tensors/seq8`, and a fresh
worktree does not have it -- `devlocal` is in .git/info/exclude and does not
come along. It reports "ok" while testing nothing. Symlinking devlocal from
the main checkout makes it real; the first pass of this sweep was measuring
only the decode-equivalence test without noticing.

    v  shape                 core_attn_out    recurrent_state    decode-equiv
                             cosine   relL2   cosine    relL2    worst cos/relL2
    0  chunk64 f32 original  0.999996 0.00282 0.999997 0.00230  1.000000 0.00087
    1  chunk64 f32           0.999996 0.00282 0.999997 0.00230  1.000000 0.00087
    6  chunk32 f32           0.999996 0.00282 0.999997 0.00230  1.000000 0.00087
    8  chunk32 f32           0.999996 0.00282 0.999997 0.00230  1.000000 0.00087
    2  chunk32 f32           0.999996 0.00282 0.999997 0.00230  1.000000 0.00087
    7  chunk16 f32           0.999996 0.00282 0.999997 0.00230  1.000000 0.00087
    9  chunk32 f32/bf16      0.999997 0.00263 0.999997 0.00251  0.999999 0.00148
    4  chunk64 bf16/bf16     0.999996 0.00285 0.999995 0.00319  0.999999 0.00113
    3  chunk64 bf16/f32      0.999994 0.00394 0.999996 0.00291  1.000000 0.00000
    5  chunk32 bf16/f32      0.999994 0.00394 0.999996 0.00291  1.000000 0.00000

Every f32 shape, at every chunk length, reproduces the baseline to all printed
digits. The re-association and the chunk change are both exact in the sense
that matters here.

**bf16 on q/k is not free.** Variants 3 and 5 are the ones that demote the q/k
planes, and they are the ones that move: core_attn_out relL2 0.00282 -> 0.00394,
a 40% increase in error, and cosine 0.999996 -> 0.999994. That still clears the
0.9999 bar, but it does not hold "at current values", so on the brief's own
terms it is a fail. The matrix planes are the cheap ones to demote (variant 9,
relL2 0.00263, inside the noise) -- the opposite of the previous run's guess
that k and v_new were the safe pair.

The direction of the error is worth recording because it is a trap. On the
decode-equivalence test the bf16-q/k variants score relL2 exactly 0.00000,
a *perfect* score, while the accurate f32 variants score 0.00087. That is not
because they are better: the decode path L2-normalizes in place into a bf16
tensor, so rounding our shared q/k to bf16 makes the two paths agree on the
same rounding error. Scored against the actual reference the same variants are
the worst of the set. A test that compares two of your own kernels rewards
matching mistakes.

None of this ends up mattering for speed: bf16 buys occupancy that a shorter
chunk buys more cheaply, and variant 8 -- every buffer still f32 -- is the
fastest shape measured.

### Round 3 -- does the chunk-length trend continue below 32? (smem-perf3.log)

    v   shape              shared      blk/SM  regs  spill st/ld   gdn ms (min)
    8   chunk32 f32        44.38 KiB      3    168    20 /   36      1840.4
    11  chunk16 f32        19.19 KiB      3    168     0 /    0      1842.0
    10  chunk16 f32        19.19 KiB      2    232     0 /    0      2190.1
    12  chunk16 f32        19.19 KiB      4    128   208 /  320      2006.0
    13  chunk32 f32        44.38 KiB      5     96   512 /  816      2653.0

No. Chunk 16 at three blocks per SM is 1842 against chunk 32's 1840 -- a tie
inside the 1-3% run-to-run spread, and this is the clean comparison, since
chunk 16 at three blocks is the one shape in the whole sweep that compiles
with zero spill. The 64 -> 32 step was worth 1.38x and the 32 -> 16 step is
worth nothing, so the superlinear terms have stopped being what the kernel
spends its time on by chunk 32. Going shorter only multiplies the per-chunk
fixed costs -- the state reload, the barriers around the substitution -- against
twice as many chunks.

Variant 13 closes the occupancy question from the other side: chunk 32 at five
blocks per SM has to fit in 96 registers, spills 512 B per thread, and lands at
2653 ms, worse than the 209 KiB original. The ladder 2 / 3 / 4 / 5 blocks at
chunk 32 reads 1971 / 1840 / 2247 / 2653. Three is the peak and the fall-off
past it tracks spill volume (20 B / 240 B / 512 B), not anything about shared
memory.

### Result

    original chunk scan   208.75 KiB   1 block/SM    2535.8 ms
    shipped default       44.38 KiB    3 blocks/SM   1840.4 ms     1.38x

    core_attn_out          cosine 0.999996  relL2 0.00282   (unchanged)
    recurrent_state_final  cosine 0.999997  relL2 0.00230   (unchanged)
    decode equivalence     cosine 1.000000  relL2 0.00087   (unchanged)

Both gate tests pass at exactly their previous values, because no buffer was
demoted: the shared block is f32 throughout and the 4.7x reduction is entirely
the re-association plus the shorter chunk.

### What was freed by aliasing vs demoted

    freed, no precision cost   v, nv (v_new), kcd   96.00 KiB   re-association
    freed, no precision cost   half of q, k, pw, at, ut
                                                    56.38 KiB   chunk 64 -> 32
    demoted to bf16                                  0.00 KiB   none shipped

`pw` was already doing double duty before this change (pairwise decay, then the
UT inverse) and `ut` dies at the end of the substitution, but nothing after it
wants 16 KiB, so pure aliasing had nothing left to give. Every byte saved here
came from not materializing a buffer at all, or from making every buffer
shorter.

### Gate, exact command from the brief, shipped default, no env override

    crates/apxinf-cuda-new/test-new.sh test -p apxinf-cuda \
      --test qwen38_gdn_prefill -- --include-ignored --nocapture

    core_attn_out (l0_16):        cosine 0.999996  relL2 0.00282
    recurrent_state_final (l0_15): cosine 0.999997  relL2 0.00230
    48 heads:                     cosine 1.000000, worst relL2 0.00087 (head 17)
    test result: ok. 2 passed; 0 failed

Identical to the values the brief lists as current, to every printed digit.

    APXINF_QWEN38_PROMPT_LEN=2048 ... --test qwen38_end_to_end --release
    prefill split: attention 310.69 ms (16 layers)   gdn 1892.48 ms (48 layers)

1892 ms on a single shot with two other agents building; 1840 ms as the min of
four rounds. Against the original kernel's 2536 ms measured in the same binary
that is 1.38x, and against the 2333 ms in the brief (quieter box) 1.27x.

Reproducing any row of the tables above needs two things: symlink `devlocal`
from the main checkout, or the reference-tensor test passes without running,
and set APXINF_GDN_SMEM_VARIANT. The map is in gdn_chunk_scan.
