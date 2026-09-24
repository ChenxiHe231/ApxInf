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
