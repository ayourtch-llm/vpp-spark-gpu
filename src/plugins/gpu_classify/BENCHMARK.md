# gpu_classify Benchmark Results

**Hardware**: NVIDIA DGX Spark — Grace CPU (ARM Neoverse V2) + Blackwell GB10 GPU (NVLink-C2C)
**Date**: 2026-02-23
**Rule storage**: cudaMallocManaged (1024-rule capacity, ~80 KB)
**Kernel mode**: persistent (adaptive; activates after 4 consecutive frames ≥ 128 pkts)
**Kernel optimisation**: tiled shmem (on-demand) + persistent full-table shmem cache (persistent kernel)

Test parameters: 256-packet frames, 50 000 reps = 12.8 M packets per measurement.
`kern µs` = GPU-only dispatch time (clock_gettime inside `gpu_classify_launch_kernel`);
`GPU µs/fr − kern µs ≈ 30 µs` = constant VPP feature-arc + buffer-management overhead.

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
       1       7.565        33.8       3.6       7.063        36.2     1.07x
       8       6.764        37.8       7.6       7.105        36.0     0.95x
      64       3.663        69.9      38.6       7.107        36.0     0.52x
     256       1.445       177.2     145.4       7.071        36.2     0.20x
    1024       0.419       610.5     575.8       6.867        37.3     0.06x
```

**GPU**: O(N) linear scan — kern µs ≈ (N rules × ~0.56 µs/rule).
The persistent-kernel shmem cache eliminates per-frame global-memory rule traffic.
Rules are read from on-chip shmem (~4-8 cycle latency) on every frame.

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
       1       7.522        34.0       4.2       7.185        35.6     1.05x
       8       7.581        33.8       3.9       7.273        35.2     1.04x
      64       7.553        33.9       4.1       7.209        35.5     1.05x
     256       7.612        33.6       4.3       7.254        35.3     1.05x
    1024       7.622        33.6       3.5       7.232        35.4     1.05x
```

GPU kern µs ≈ **4 µs flat at all rule counts** — the persistent shmem cache means the
full rule table is already on-chip; the classify loop hits rule 0 and breaks immediately
regardless of N.  GPU/ACL = **1.05× at all rule counts** (GPU faster than ACL).

This is a significant result: the GPU's parallel-per-packet execution (256 packets
classified simultaneously) fully compensates for the sequential per-rule scan when rules
are hot in shmem and the first rule fires.

---

## Scenario 3 — Last-match (full scan + match)

Traffic matches rule N-1 only; both plugins scan all N rules before matching.
Nearly identical to Scenario 1 (same number of comparisons, different terminal action).

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.608        33.6       4.3       7.220        35.5     1.05x
       8       6.787        37.7       7.5       7.157        35.8     0.95x
      64       3.666        69.8      39.6       7.178        35.7     0.51x
     256       1.438       178.1     146.0       7.247        35.3     0.20x
    1024       0.420       610.2     576.1       7.162        35.7     0.06x
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
       1       0.464       552.0     518.7       7.082        36.2     0.07x   /32
       2       0.463       553.1     517.8       7.037        36.4     0.07x   /32 /31
       4       0.464       551.4     516.6       7.080        36.2     0.07x   /32…/29
       8       0.465       551.1     517.1       6.572        39.0     0.07x   /32…/25
      16       0.464       551.9     517.3       5.200        49.2     0.09x   /32…/17
      21       0.472       542.9     512.1       4.605        55.6     0.10x   /32…/12
```

**GPU**: flat ~551–553 µs/frame (always scans all 1024 rules, O(N)); kern µs ≈ 517–519
µs for 1024 rules.  Persistent shmem cache eliminates per-frame global-memory loads;
rules are classified directly from on-chip shmem.

**ACL multi-table probe overhead**:
- K=1..4: ACL flat at ~36-37 µs — probe overhead too small to measure
- K=8: ACL rises to 39.0 µs (+2.8 µs above K=1, +8%)
- K=16: ACL 49.2 µs (+13.0 µs, +36%)
- K=21: ACL 55.6 µs (+19.4 µs, +54%)

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
| GPU overhead (VPP arc) | ~30 µs constant above kern µs |
| GPU kern µs @ 1 rule | ~4 µs |
| GPU kern µs @ 64 rules | ~39 µs |
| GPU kern µs @ 1024 rules | ~576 µs |
| GPU kern µs scaling | linear with N (O(N) scan, ~0.56 µs/rule) |
| First-match kern µs | ~4 µs flat at any rule count |

**No-match crossover**: GPU beats ACL at N = 1 rule (GPU/ACL = 1.07×).
At N = 8 GPU/ACL = 0.95×.  At N ≥ 64 ACL is faster (O(1) hash table vs O(N) linear scan).

**First-match**: GPU/ACL = **1.05× at every rule count** (GPU faster than ACL).
With the persistent shmem cache, the full rule table is already on-chip for every frame;
the classify loop hits rule 0 and breaks in ~4 µs regardless of total rule count N.

**Mask diversity**: The ACL plugin's multi-table design has measurable per-probe overhead
at K ≥ 8 distinct prefix lengths with 1024 rules. Each distinct dst prefix length creates
one hash table; K tables → K probes per packet at ~1–1.3 µs/frame (3.5–5 ns/pkt/probe).
At K=21, ACL costs 56 µs/frame (+54% vs K=1). At 64 rules with K=1..16 the overhead is
below measurement noise (tables too small to cause cache pressure).

---

## Kernel optimisation history

| Stage | No-match 1024r kern µs | First-match 1024r kern µs | Change |
|-------|----------------------|--------------------------|--------|
| Baseline (global mem, ld.global.nc) | 810 | 4 | — |
| + Tiled shmem (256-rule tiles, per-tile early-exit vote) | 590 (-27%) | 7 (+75%) | No-match speedup; regression on first-match |
| + Persistent shmem cache (full table, reload on rule change) | 576 (-29%) | **4 (-3%)** | Regression fixed; steady-state has zero rule loads |

The two-stage design uses different shmem strategies for the two kernel paths:
- **On-demand kernel** (sparse traffic): 256-rule tiled shmem, 20 KB, no `cudaFuncSetAttribute`
- **Persistent kernel** (sustained traffic): 1024-rule full cache, 80 KB, `cudaFuncSetAttribute`
  reloads only when `ctrl->rule_version` changes (CPU increments it in `gpu_classify_update_rules`)

---

## Architecture notes

- ACL plugin: stateless multi-field hash-table classification, one table per unique mask
  combination. O(1) per packet per table; K distinct mask types → K probes per packet.
  Proven correct via functional test: `test_bench_00b_acl_functional` verifies deny/permit
  works, `show acl-plugin interface` confirms the feature is enabled inbound on pg0.
  Multi-probe overhead: ~1–1.3 µs/frame per additional probe (measurable at K ≥ 8 with
  1024 rules; below noise at K ≤ 16 with only 64 rules).

- gpu_classify: linear scan of all rules by 256 GPU threads in parallel (one thread per
  packet slot). O(N) in rule count but all 256 packets are processed simultaneously.

  **On-demand kernel** (`gpu_classify_tiled`): rules loaded 256 at a time from global
  memory into shmem per invocation; per-tile block-wide vote enables early exit when all
  threads have matched. 20 KB shmem, no `cudaFuncSetAttribute`.

  **Persistent kernel** (`gpu_classify_from_shmem`): full rule table (≤ 80 KB) loaded
  into shmem once at startup and on each `rule_version` change; every frame classifies
  directly from already-hot shmem with zero global-memory rule traffic.  First-match
  kern µs is ~4 µs flat at any rule count.  80 KB shmem requires `cudaFuncSetAttribute`
  (called once at init; Blackwell supports ≤ 256 KB per block with the opt-in).

- Persistent kernel eliminates the ~30 µs cudaStreamSynchronize round-trip; kern µs
  measures only the GPU-side classification time. VPP feature-arc overhead (~30 µs/frame)
  dominates at low rule counts.
