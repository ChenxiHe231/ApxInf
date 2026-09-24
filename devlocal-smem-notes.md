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
