# gpu_classify Benchmark Results

**Hardware**: NVIDIA DGX Spark — Grace CPU (ARM Neoverse V2) + Blackwell GB10 GPU (NVLink-C2C)
**Date**: 2026-02-22
**Rule storage**: cudaMallocManaged (1024-rule capacity, ~80 KB)
**Kernel mode**: persistent (adaptive; activates after 4 consecutive frames ≥ 128 pkts)
**Kernel optimisation**: tiled shared-memory caching (256-rule tile + per-tile early-exit vote)

Test parameters: 256-packet frames, 50 000 reps = 12.8 M packets per measurement.
`kern µs` = GPU-only dispatch time (clock_gettime inside `gpu_classify_launch_kernel`);
`GPU µs/fr − kern µs ≈ 31 µs` = constant VPP feature-arc + buffer-management overhead.

---

## Validation

ACL is confirmed to be correctly applied:

- `test_bench_00b_acl_functional` sends 1 deny-matched packet → drops; 1 pass packet → forwards.
- `show acl-plugin interface` shows `input acl(s): 0` on sw_if_index 1 (pg0).

---

## Baseline — pure VPP forwarding (no feature)

```
  Baseline: 9.669 Mpps   26.5 µs/frame
```

This is the floor: pg → ip4-input → ip4-unicast arc → ip4-lookup → pg-output.
All other scenarios add feature overhead on top of this.

---

## Scenario 1 — No-match (full linear scan)

Traffic hits **none** of the N deny rules; GPU scans all N, no match, passes.
ACL has N deny rules on distinct ports + one permit-all fallthrough.

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.429        34.5       4.1       7.077        36.2     1.05x
       8       6.665        38.4       8.1       7.084        36.1     0.94x
      64       3.592        71.3      40.3       7.070        36.2     0.51x
     256       1.411       181.5     149.9       7.070        36.2     0.20x
    1024       0.410       625.0     590.1       7.093        36.1     0.06x
```

**GPU**: O(N) linear scan — kern µs ≈ (N rules × ~0.58 µs/rule) with tiled shmem.
The tiled shared-memory caching reduces kern µs by **~27%** vs global-memory access at
all rule counts (e.g. 64 rules: 55.4 → 40.3 µs; 1024 rules: 810.2 → 590.1 µs).

**ACL**: flat ~36-37 µs regardless of N. The ACL plugin places all port-based deny rules
sharing the same (src_mask, dst_mask, proto_mask, port_mask) pattern into a **single hash
table**. Hash-table lookup is O(1), so adding more entries doesn't slow it down.

**ACL overhead** = 36.5 − 26.5 = **~10 µs** constant above baseline (the hash-table probe
itself). This is confirmed real: functional validation proves the ACL drops matched packets,
and `show acl-plugin interface` confirms it is applied.

---

## Scenario 2 — First-match (best case)

Traffic always matches rule 0. Both plugins exit after exactly 1 comparison.

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.449        34.4       3.7       7.198        35.6     1.03x
       8       7.433        34.4       4.6       7.226        35.4     1.03x
      64       7.258        35.3       4.0       7.243        35.3     1.00x
     256       6.867        37.3       7.9       7.236        35.4     0.95x
    1024       6.782        37.7       7.3       7.231        35.4     0.94x
```

GPU kern µs at small N (≤ 64 rules): ~4 µs flat — only one tile of 256 rules is ever
loaded, and the second tile's early-exit vote fires immediately.

At large N (256–1024 rules): kern µs rises slightly (7–8 µs) because one full 256-rule
tile must be loaded cooperatively before rule 0 can be checked.  The overhead is the tile
load cost (~20 KB × 256 threads) plus two block-wide __syncthreads() calls (early-exit
vote), not additional rule checks.  GPU/ACL remains ≥ 0.94× at all rule counts.

---

## Scenario 3 — Last-match (full scan + match)

Traffic matches rule N-1 only; both plugins scan all N rules before matching.
Nearly identical to Scenario 1 (same number of comparisons, different terminal action).

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.472        34.3       3.9       7.190        35.6     1.04x
       8       6.502        39.4       8.9       7.222        35.4     0.90x
      64       3.576        71.6      39.5       7.207        35.5     0.50x
     256       1.410       181.6     149.7       7.237        35.4     0.19x
    1024       0.410       624.9     590.5       7.215        35.5     0.06x
```

---

## Scenario 4 — Diverse dst prefix lengths (1024 rules)

Fixed N=1024 dst-only prefix rules, varying number of distinct dst prefix lengths from 1
to 21.  Traffic matches none.  Each distinct prefix length creates one ACL hash table.

Rule addresses: generated as `(global_idx << (32-plen))` in 0.0.0.0–63.255.255.255 space
(safely away from 172.16.x.x test traffic). COMBO_LENGTHS = [/32, /31, …, /12].

```
  Tables    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL  dst lengths
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       0.453       564.7     531.5       7.137        35.9     0.06x   /32
       2       0.453       565.6     530.1       7.122        35.9     0.06x   /32 /31
       4       0.453       565.5     531.7       7.144        35.8     0.06x   /32…/29
       8       0.453       564.8     531.5       6.633        38.6     0.07x   /32…/25
      16       0.454       564.3     532.7       5.218        49.1     0.09x   /32…/17
      21       0.459       557.6     522.2       4.570        56.0     0.10x   /32…/12
```

**GPU**: flat ~564–565 µs/frame (always scans all 1024 rules, O(N)); kern µs ≈ 530–532
µs for 1024 rules.  Tiled shmem reduces kern µs by **~22%** vs global-memory access
(previously 685–690 µs).  Dst-only prefix rules are faster per rule than port-based rules
(fail at dst_ip check rather than dst_port), which explains the lower absolute kern µs
vs Scenario 1's 590 µs at 1024 rules.

**ACL multi-table probe overhead**:
- K=1..4: ACL flat at ~36-37 µs — probe overhead too small to measure
- K=8: ACL rises to 38.6 µs (+2.7 µs above K=1, +8%)
- K=16: ACL 49.1 µs (+13.2 µs, +37%)
- K=21: ACL 56.0 µs (+20.1 µs, +56%)

**Observation**: The ACL IS doing K hash-table probes per packet, but the overhead only
becomes measurable at K ≥ 8 (with 128+ rules per table causing cache pressure). With 64
rules and K=1..16 (old Scenario 4), table sizes were too small (4 rules/table) for the
per-probe overhead to register. With 1024 rules, the larger tables create enough cache
pressure to make K-probe scaling visible.

**Per-probe cost**: approximately 0.9–1.3 µs per frame of 256 packets (~3.5–5 ns/pkt/probe),
rising slightly as more tables compete for L2 cache.

---

## Summary

| Metric | Value |
|--------|-------|
| Baseline VPP forwarding | 26.5 µs/frame |
| ACL overhead (O(1) hash table) | ~10 µs constant above baseline |
| GPU overhead (VPP arc) | ~31 µs constant above kern µs |
| GPU kern µs @ 1 rule | ~4 µs |
| GPU kern µs @ 64 rules | ~40 µs |
| GPU kern µs @ 1024 rules | ~590 µs |
| GPU kern µs scaling | linear with N (O(N) scan, ~0.58 µs/rule with tiled shmem) |
| Tiled shmem speedup (no-match) | ~27% reduction in kern µs at all rule counts |

**Crossover point**: GPU beats ACL at N = 1 rule on no-match workloads (GPU/ACL = 1.05×).
At N = 8 GPU/ACL = 0.94× (6% slower than ACL).  At N ≥ 64 the ACL plugin is faster.
The tiled optimisation narrows the gap at all N (e.g. 64 rules: 0.42× → 0.51×).

**First-match**: GPU/ACL ≥ 0.94× at any rule count.  At small N (≤ 64) both GPU and ACL
are neck-and-neck (~34-35 µs).  At N=1024, GPU kern µs = 7.3 µs (one 256-rule tile loaded
+ early-exit vote on tile 2) vs ACL ~10 µs classification overhead; total frame times
37.7 vs 35.4 µs (GPU 6% slower, vs 24% slower with the full-preload design).

**Mask diversity**: The ACL plugin's multi-table design has measurable per-probe overhead
at K ≥ 8 distinct prefix lengths with 1024 rules. Each distinct dst prefix length creates
one hash table; K tables → K probes per packet at ~1–1.3 µs/frame (3.5–5 ns/pkt/probe).
At K=21, ACL costs 56 µs/frame (+56% vs K=1). At 64 rules with K=1..16 the overhead is
below measurement noise (tables too small to cause cache pressure).

---

## Architecture notes

- ACL plugin: stateless multi-field hash-table classification, one table per unique mask
  combination. O(1) per packet per table; K distinct mask types → K probes per packet.
  Proven correct via functional test: `test_bench_00b_acl_functional` verifies deny/permit
  works, `show acl-plugin interface` confirms the feature is enabled inbound on pg0.
  Multi-probe overhead: ~1–1.3 µs/frame per additional probe (measurable at K ≥ 8 with
  1024 rules; below noise at K ≤ 16 with only 64 rules).

- gpu_classify: linear scan of all rules by 256 GPU threads in parallel (one thread per
  packet slot). O(N) in rule count but all 256 packets are processed simultaneously. The
  crossover vs ACL depends on rule count and workload.

  **Tiled shared-memory caching** (`gpu_classify_tiled` device function):
  Rules are loaded from global memory into shared memory 256 at a time (one tile).
  Before each tile, a block-wide vote via `atomicOr` checks whether any thread still needs
  to classify; if all threads have matched, the block exits early.
  - Shmem: 256 × 80 B (rule tile) + 4 B (vote flag) = 20 484 bytes (< 48 KB default;
    no `cudaFuncSetAttribute` needed).
  - No-match speedup: ~27% reduction in kern µs (ld.shared vs ld.global.nc stalls).
  - First-match benefit: loads only 1 tile (256 rules) before early exit, keeping
    kern µs at ~4–8 µs for workloads where the first matching rule is in the first tile.

- Persistent kernel eliminates the ~30 µs cudaStreamSynchronize round-trip; kern µs
  measures only the GPU-side classification time. VPP feature-arc overhead (~31 µs/frame)
  dominates at low rule counts.
