# gpu_classify Benchmark Results

**Hardware**: NVIDIA DGX Spark — Grace CPU (ARM Neoverse V2) + Blackwell GB10 GPU (NVLink-C2C)
**Date**: 2026-02-22
**Rule storage**: cudaMallocManaged (moved from `__constant__` memory to support up to 1024 rules)
**Kernel mode**: persistent (adaptive; activates after 4 consecutive frames ≥ 128 pkts)

Test parameters: 256-packet frames, 50 000 reps = 12.8 M packets per measurement.
`kern µs` = GPU-only dispatch time (clock_gettime inside `gpu_classify_launch_kernel`);
`GPU µs/fr − kern µs ≈ 31 µs` = constant VPP feature-arc + buffer-management overhead.

---

## Scenario 1 — No-match (full linear scan)

Traffic hits **none** of the N deny rules; GPU scans all N, finds no match, passes packet.
ACL has N deny rules on distinct ports + one permit-all fallthrough.

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.452        34.4       4.2       7.213        35.5     1.03x
       8       6.270        40.8      10.8       7.192        35.6     0.87x
      64       2.925        87.5      56.7       7.204        35.5     0.41x
     256       1.071       239.1     206.7       7.121        36.0     0.15x
    1024       0.303       845.8     811.6       7.223        35.4     0.04x
```

**GPU**: O(N) linear scan — kern µs grows proportionally to N.
**ACL**: flat ~35-36 µs regardless of N. The ACL plugin builds one hash table per unique
(src_mask, dst_mask, proto_mask, port_mask) pattern. All port-based deny rules share the
same mask pattern → single hash table, O(1) lookup regardless of N.

> **Note**: ACL flatness needs verification — see "ACL Validation" section below.

---

## Scenario 2 — First-match (best case)

Traffic always matches rule 0. Both plugins exit after exactly 1 comparison.

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.418        34.5       3.8       7.267        35.2     1.02x
       8       7.417        34.5       4.5       7.276        35.2     1.02x
      64       7.442        34.4       4.1       7.286        35.1     1.02x
     256       7.422        34.5       3.7       7.306        35.0     1.02x
    1024       7.420        34.5       3.9       7.354        34.8     1.01x
```

GPU kern µs ≈ 4 µs flat — only one rule is checked per packet, independent of N.
GPU and ACL are neck-and-neck (~34.5 µs vs ~35 µs); both exit immediately on rule 0.

---

## Scenario 3 — Last-match (full scan + match)

Traffic matches rule N-1 only; both plugins scan all N rules before matching.
Nearly identical to Scenario 1 (same number of comparisons, different terminal action).

```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
  ------  ----------  ----------  --------  ----------  ----------  --------
       1       7.423        34.5       4.3       7.128        35.9     1.04x
       8       6.230        41.1      10.1       7.295        35.1     0.85x
      64       2.912        87.9      57.3       7.273        35.2     0.40x
     256       1.066       240.3     207.1       7.362        34.8     0.14x
    1024       0.302       847.0     811.7       7.333        34.9     0.04x
```

---

## Scenario 4 — Diverse src+dst prefix mask combinations (64 rules)

Fixed N=64 rules, varying number of distinct (src_mask, dst_mask) pairs from 1 to 16.
Traffic matches none. Tests whether mask diversity affects ACL vs GPU differently.

The ACL plugin builds one hash table per unique (src_mask, dst_mask) pair, so more
distinct mask combos → more hash table probes per packet. The GPU always scans all 64
rules linearly regardless of mask diversity.

```
  Combos    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL  (src, dst) pairs
  ------  ----------  ----------  --------  ----------  ----------  --------  --------------------
       1       3.913        65.4      36.3       7.225        35.4     0.54x  (/8,/8)
       2       3.888        65.8      34.4       7.218        35.5     0.54x  (/8,/8) (/8,/16)
       4       3.857        66.4      36.4       7.210        35.5     0.53x  (/8,/8)…(/8,/32)
       8       3.902        65.6      35.1       7.204        35.5     0.54x  8 combos
      16       3.915        65.4      33.3       7.226        35.4     0.54x  all 16 combos
```

**Observation**: ACL stays flat even as mask combos grow from 1 to 16. This is surprising and
requires validation — see below.

---

## Key observations

| Metric | Value |
|--------|-------|
| VPP overhead (constant) | ~31 µs/frame |
| GPU kernel @ 1 rule | ~4 µs (kern µs) |
| GPU kernel @ 64 rules | ~57 µs |
| GPU kernel @ 1024 rules | ~812 µs |
| GPU kern µs scaling | linear with N (O(N) scan) |
| ACL latency | ~35 µs flat (O(1) hash table) |

**Crossover point**: GPU outperforms ACL at ≤ ~8 rules on no-match workloads.
At ≥ 64 rules, the ACL plugin is faster due to its O(1) hash-table design.

**First-match advantage**: The persistent GPU kernel is competitive with ACL at any rule
count when traffic always matches rule 0 (~34.5 µs vs ~35 µs), because the GPU's 256
threads all finish after 1 comparison.

---

## ACL Validation TODO

The flat ACL performance (~35 µs regardless of rule count or mask diversity) could mean:

1. **(Correct)** The ACL plugin's hash tables are truly O(1) per packet, all rules of the
   same mask pattern share one table, and adding entries doesn't slow lookups.

2. **(Incorrect)** The ACL is not being applied to the interface correctly, and traffic
   bypasses it entirely (~35 µs = bare VPP forwarding overhead without classification).

**Validation needed**: confirm that a packet matching a deny rule is actually dropped when
the ACL is installed. Check VPP ACL drop counters (`show acl-plugin acl`) after a run
with matching traffic.

The benchmark code (`_acl_install`) uses:
```python
VppAclInterface(sw_if_index=pg0, acls=[acl], n_input=1)
```
This should apply the ACL as an inbound ip4-unicast feature on pg0, but this needs
end-to-end verification.
