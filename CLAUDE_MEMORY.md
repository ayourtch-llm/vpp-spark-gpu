# VPP GPU Classify Plugin — Comprehensive Knowledge Base
# Persistent backup (survives container restarts): /store/vpp/CLAUDE_MEMORY.md
# At session start, read /store/vpp/CLAUDE_MEMORY.md if this file is missing.
# Also maintained at: /root/.claude/projects/-store-vpp/memory/MEMORY.md

---

## 1. Hardware

- **Machine**: NVIDIA DGX Spark
- **GPU**: NVIDIA GB10 (Grace Blackwell Superchip), sm_100 (compute capability 10.0)
- **CPU**: NVIDIA Grace (ARM Neoverse V2, aarch64)
- **Interconnect**: NVLink-C2C — CPU and GPU share coherent physical DRAM (unified memory is fast, not just mapped)
- **Driver**: CUDA 13.0 (max supported); CUDA toolkit 12.8 installed
- **GPU SHMEM**: Blackwell supports up to 256 KB dynamic shared memory per block (with cudaFuncSetAttribute opt-in)
- **GPU L2**: 128 MB; easily fits the entire 80 KB rule table
- **OS**: Ubuntu 24.04.3 LTS (Noble), aarch64
- **Container**: non-persistent Docker; use `runme` script to re-install deps

---

## 2. Project Layout

- **Plugin location**: `/store/vpp/src/plugins/gpu_classify/`
- **Branch**: `dgx-spark-plugin`
- **Test file**: `/store/vpp/test/test_gpu_classify.py`
- **Bench file**: `/store/vpp/test/test_gpu_classify_bench.py`
- **Benchmark doc**: `/store/vpp/src/plugins/gpu_classify/BENCHMARK.md`

### Plugin files

| File | Purpose |
|------|---------|
| `gpu_classify_types.h` | Shared C/CUDA types (stdint.h only, no VPP headers) |
| `gpu_classify.h` | VPP plugin header; includes types.h + VPP headers |
| `gpu_classify_kernel.cu` | CUDA kernel + C-callable wrappers (nvcc, sm_100) |
| `gpu_classify_node.c` | VPP graph nodes (ip4-unicast + ip6-unicast arcs) |
| `gpu_classify.c` | Plugin init, CLI commands (enable, rule, show) |
| `CMakeLists.txt` | Builds CUDA static lib + VPP plugin .so |

---

## 3. Plugin Architecture

### Feature arcs
- `ip4-unicast` → node `gpu-classify-ip4`
- `ip6-unicast` → node `gpu-classify-ip6`
- Both run before `ip{4,6}-flow-classify`
- Shared dispatch helper (pass 2 + pass 3) used by both nodes

### Frame processing pipeline
1. **Pass 1 (CPU)**: Extract IP+TCP/UDP headers from VPP buffers → fill `gpu_pkt_desc_t[]`
2. **Pass 2 (GPU)**: `gpu_classify_launch_kernel()` → kernel classifies 256 packets in parallel
3. **Pass 3 (CPU)**: Read `results[]` → route packets (PASS=feature-arc, DROP=error-drop, MARK=flag+feature-arc)

### Data types (`gpu_classify_types.h`)

**`gpu_pkt_desc_t`** — 64 bytes (one cache line):
- `uint8_t src_ip[16]` / `uint8_t dst_ip[16]` — IPv4: bytes 0–3 only; IPv6: all 16 bytes
- `uint16_t src_port`, `uint16_t dst_port` — network byte order
- `uint8_t ip_proto`, `uint8_t tcp_flags`, `uint8_t ip_version`, `uint8_t valid`
- `uint8_t payload[16]`, `uint8_t _pad[8]`

**`gpu_classify_rule_t`** — 80 bytes:
- `uint8_t src_addr[16]`, `uint8_t src_mask[16]`, `uint8_t dst_addr[16]`, `uint8_t dst_mask[16]`
- `uint16_t src_port`, `uint16_t dst_port` — 0 = wildcard
- `uint8_t proto` — 0 = wildcard
- `uint8_t action` — `GPU_CLASSIFY_ACTION_{PASS=0, DROP=1, MARK=2}`
- `uint8_t tcp_flags_mask`, `uint8_t tcp_flags_val`
- `uint8_t ip_version` — 4, 6, or 0 (match any version)
- `uint8_t _pad[7]`

**`gpu_classify_ctrl_t`** — 256 bytes (two 128-byte cache lines):
- Cache line 0 (CPU writes, GPU reads): `submit_seq`, `n_packets`, `n_rules`, `kill`, `rule_version`, pad
- Cache line 1 (GPU writes, CPU reads): `done_seq`, pad
- `rule_version`: CPU increments (RELEASE) in `gpu_classify_update_rules()`; GPU reloads shmem when cached copy differs

**`gpu_classify_cuda_res_t`** — CUDA resource bundle:
- `stream`, `descs`, `results`, `ctrl`, `rules`, `n_rules`
- Adaptive state: `persist_active`, `busy_frames`, `idle_frames`
- Stats: `n_kernel_calls`, `n_gpu_packets`, `total_kernel_ms`, `min_kernel_ms`, `max_kernel_ms`
- Histogram: `lat_hist[GPU_CLASSIFY_LAT_BUCKETS]` (log2-µs scale, 12 buckets)

### Actions
- `PASS (0)`: forward to next feature node
- `DROP (1)`: send to `error-drop`; set `buf->error`; do NOT call `vlib_node_increment_counter` for drops — that causes double-counting
- `MARK (2)`: set `VLIB_BUFFER_FLAG_USER(1)` + forward to next feature node

### Rule storage
- `cudaMallocManaged` — not constant memory; allows 1024 rules
- 1024 rules × 80 bytes = 81,920 bytes (≈ 80 KB)
- Fits easily in Blackwell L2 (128 MB)
- Kernel receives rules as `const __restrict__` pointer → ld.global.nc (L1 read-only cache)
- Rule count in `cuda_res.n_rules` (CPU) and `ctrl->n_rules` (GPU, volatile, per-batch)
- `gpu_classify_update_rules()`: plain memcpy + atomic update of `ctrl->n_rules`; no stop/restart needed

### `ip_matches()` device inline
```cuda
__device__ __forceinline__ static int
ip_matches(const uint8_t *pkt, const uint8_t *addr, const uint8_t *mask) {
    const uint32_t *p = (const uint32_t *)pkt;
    const uint32_t *a = (const uint32_t *)addr;
    const uint32_t *m = (const uint32_t *)mask;
    return ((p[0]&m[0])==a[0]) & ((p[1]&m[1])==a[1])
         & ((p[2]&m[2])==a[2]) & ((p[3]&m[3])==a[3]);
}
```
- mask word = 0 → term = true → wildcard works for unused IPv4 bytes automatically
- 4 × 32-bit comparisons cover full 128-bit address

---

## 4. Kernel Architecture

### Dispatch modes (adaptive)
- **On-demand** (`gpu_classify_tiled`): fresh kernel launch per frame via `cudaLaunchKernel`
  + `cudaStreamSynchronize`; 256 threads, 1 block
- **Persistent** (`gpu_classify_persistent`): kernel stays alive; CPU/GPU handshake via `submit_seq`/`done_seq`

**Adaptive thresholds**:
- `GPU_CLASSIFY_PERSIST_START_FRAMES = 4` — consecutive "busy" frames (≥ 128 pkts) before activating
- `GPU_CLASSIFY_PERSIST_STOP_FRAMES = 16` — consecutive "idle" frames before deactivating
- `GPU_CLASSIFY_PERSIST_MIN_PKTS = 128`

### Two-stage shared-memory design

#### On-demand kernel: `gpu_classify_tiled()`
- **Shmem size**: `GPU_CLASSIFY_TILE_RULES × sizeof(rule) + sizeof(int)` = 20,484 bytes
- **Fits in 48 KB default limit** — no `cudaFuncSetAttribute` needed
- Algorithm:
  1. Per-tile early-exit vote: thread 0 clears `any_not_done`; unmatched threads `atomicOr` it; if 0 → break
  2. Cooperative tile load: each thread loads strided words (`w = tid; w < tile_words; w += blockDim.x`)
  3. `__syncthreads()` after load
  4. Linear scan of shmem tile; break on first match
- Tile size: `GPU_CLASSIFY_TILE_RULES = 256` (= `GPU_CLASSIFY_MAX_FRAME`)

#### Persistent kernel: `gpu_classify_persistent()` + `gpu_classify_from_shmem()`
- **Shmem size**: `GPU_CLASSIFY_MAX_RULES × sizeof(rule)` = 81,920 bytes
- **Requires `cudaFuncSetAttribute`** for persistent kernel only (called once at init)
- Algorithm:
  1. Spin-poll `ctrl->submit_seq` (volatile) until it differs from `last_seq`
  2. Thread 0 broadcasts `seq`, `n`, `n_rules` via shmem; `__syncthreads()`
  3. If `ctrl->rule_version != last_rule_version`: cooperative load of all rules into shmem; `__syncthreads()`; update `last_rule_version`
  4. Call `gpu_classify_from_shmem()` — pure linear scan from shmem, no global loads
  5. `__threadfence_system()` + `__syncthreads()`; thread 0 writes `done_seq`
- `last_rule_version` initialized to `~0u` → forces initial load
- All 256 threads evaluate version check identically → `__syncthreads()` in conditional is CUDA-safe

`gpu_classify_from_shmem()` — pure classify, no syncs:
```cuda
__device__ static void
gpu_classify_from_shmem(int tid, int n, int n_rules,
                        const gpu_pkt_desc_t *descs,
                        const gpu_classify_rule_t *s_rules,
                        uint8_t *results)
{
    uint8_t action = GPU_CLASSIFY_ACTION_PASS;
    if (tid < n) {
        const gpu_pkt_desc_t *d = &descs[tid];
        for (int i = 0; i < n_rules; i++) {
            const gpu_classify_rule_t *r = &s_rules[i];
            if (r->proto != 0 && r->proto != d->ip_proto) continue;
            if (r->ip_version != 0 && r->ip_version != d->ip_version) continue;
            if (!ip_matches(d->src_ip, r->src_addr, r->src_mask)) continue;
            if (!ip_matches(d->dst_ip, r->dst_addr, r->dst_mask)) continue;
            if (r->src_port != 0 && r->src_port != d->src_port) continue;
            if (r->dst_port != 0 && r->dst_port != d->dst_port) continue;
            if (r->tcp_flags_mask != 0 &&
                (d->tcp_flags & r->tcp_flags_mask) != r->tcp_flags_val) continue;
            action = r->action;
            break;
        }
    }
    results[tid] = action;
}
```

### Handshake protocol (CPU side)
1. Write `n_packets`
2. Atomic-store `submit_seq += 1` with `__ATOMIC_SEQ_CST`
3. Spin-poll `done_seq` with `__ATOMIC_ACQUIRE` until `done_seq == submit_seq`

### Kill sequence (CPU)
1. Atomic-store `kill = 1` with `__ATOMIC_SEQ_CST`
2. Atomic-store `submit_seq += 1` (wakes GPU)
3. `cudaStreamSynchronize()` waits for kernel to return

### Rule update (no stop/restart needed)
```c
// In gpu_classify_update_rules():
clib_memcpy(res->rules, new_rules, n * sizeof(gpu_classify_rule_t));
res->n_rules = n;
if (res->persist_active) {
    __atomic_store_n(&res->ctrl->n_rules, (int32_t)n, __ATOMIC_RELEASE);
    __atomic_fetch_add(&res->ctrl->rule_version, 1u, __ATOMIC_RELEASE);
}
```

---

## 5. Build System

### CUDA install (in container)
```bash
# sbsa = server ARM (aarch64)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### CMakeLists.txt key points
- Finds nvcc in PATH and `/usr/local/cuda-12.8/bin`
- `CUDA_ARCHITECTURES = "100"` for Blackwell
- Builds `gpu_classify_cuda_kernel` static CUDA lib
- Plugin `.so` links via `CUDA::cudart`

### Core VPP build fix
- `/store/vpp/src/CMakeLists.txt` line 129: wrapped `-Werror -Wall` in
  `$<$<COMPILE_LANGUAGE:C,CXX>:...>` so nvcc doesn't receive bare `-Werror -Wall`
  (nvcc 12.8 treats `-Wall` as the value for `-Werror` — fatal error)

### Other fixes
- `gpu_classify.c`: added `#include <vpp/app/version.h>` for `VPP_BUILD_VER`
- VPP idiom: use `clib_net_to_host_u16()` not `ntohs()` for byte-swap in node code

---

## 6. Test Suite

- **File**: `/store/vpp/test/test_gpu_classify.py`
- **Run**: `make test TEST=test_gpu_classify` (no `.py` suffix — causes class=py filter bug)
- **Status**: All 19 tests pass

### Test structure
- `setUpClass`: `config_ip4+resolve_arp` AND `config_ip6+resolve_ndp` on each pg interface
- `setUp`: enable both `gpu-classify-ip4` (ip4-unicast) and `gpu-classify-ip6` (ip6-unicast) on pg0
- `tearDown`: disable both features and clear all rules
- Two probes: feature_enable_disable for ip4 node, then ip6 node; then cuda_ready check
- Tests 1–14: IPv4; tests 15–19: IPv6

### Helpers
- `_pkt()`/`_pkts()` — IPv4 test packets
- `_pkt6()`/`_pkts6()` — IPv6 test packets
- `_err(name)` — read `/err/gpu-classify-ip4/<name>` counter
- `_err6(name)` — read `/err/gpu-classify-ip6/<name>` counter

### Critical gotchas
- "Pass" test packets must use routeable dst (`pg1.remote_ip4` or `pg1.remote_ip6`) — not self
- For IPv6 drop-counter tests: use `_err6()` to confirm GPU (not ip6-lookup) dropped packets
- **Counter double-counting**: setting `buf->error` on dropped packets causes `error-drop` to auto-increment.
  Do NOT also call `vlib_node_increment_counter` for drops — only PROCESSED and MARKED need explicit increments

---

## 7. Benchmark Suite

- **File**: `/store/vpp/test/test_gpu_classify_bench.py`
- **Run**: `make test TEST=test_gpu_classify_bench`
- **Doc**: `/store/vpp/src/plugins/gpu_classify/BENCHMARK.md`

### Parameters
- `BATCH=256`, `N_REPS=50000` → 12.8 M packets per measurement
- `RULE_COUNTS=[1, 8, 64, 256, 1024]`

### kern µs measurement
- Snapshot `show gpu-classify` before and after N_REPS frames
- `kern µs = delta(total_kernel_us) / delta(n_calls)`
- GPU-only dispatch time; VPP feature-arc overhead ≈ 30 µs/frame constant on top

### 4 scenarios
1. **No-match (full linear scan)**: N deny rules on distinct ports + permit-all; traffic matches none
2. **First-match (best case)**: traffic always matches rule 0
3. **Last-match (full scan + match)**: traffic matches rule N-1
4. **Diverse dst prefix lengths**: N=1024 rules; K=1..21 distinct dst prefix lengths; traffic matches none

#### Scenario 4 details
- Fixed N=1024 rules, COMBO_LENGTHS=[/32,/31,...,/12]
- Rule addresses generated as `(global_idx << (32-plen))` in 0.0.0.0–63.255.255.255 space
- Each distinct dst prefix length → one ACL hash table; K tables → K probes/packet in ACL
- `_SRC_PFX` / `_DST_PFX` must be defined as inline literals (not class attrs in comprehensions)

---

## 8. Benchmark Results (DGX Spark, GB10 Blackwell, persistent kernel + shmem cache)

### Baseline
```
Baseline: 9.669 Mpps   26.5 µs/frame
```

### Scenario 1 — No-match
```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
       1       7.565        33.8       3.6       7.063        36.2     1.07x
       8       6.764        37.8       7.6       7.105        36.0     0.95x
      64       3.663        69.9      38.6       7.107        36.0     0.52x
     256       1.445       177.2     145.4       7.071        36.2     0.20x
    1024       0.419       610.5     575.8       6.867        37.3     0.06x
```
GPU: O(N) linear scan ~0.56 µs/rule; ACL: O(1) hash table ~36-37 µs flat
Crossover: GPU beats ACL at N=1 only (1.07×); at N≥8 ACL faster

### Scenario 2 — First-match
```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
       1       7.522        34.0       4.2       7.185        35.6     1.05x
       8       7.581        33.8       3.9       7.273        35.2     1.04x
      64       7.553        33.9       4.1       7.209        35.5     1.05x
     256       7.612        33.6       4.3       7.254        35.3     1.05x
    1024       7.622        33.6       3.5       7.232        35.4     1.05x
```
GPU kern µs ≈ 4 µs flat at ALL rule counts — persistent shmem cache means full table is on-chip;
classify loop hits rule 0 and breaks immediately regardless of N. GPU/ACL = 1.05× at all N.

### Scenario 3 — Last-match
```
   Rules    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
       1       7.608        33.6       4.3       7.220        35.5     1.05x
       8       6.787        37.7       7.5       7.157        35.8     0.95x
      64       3.666        69.8      39.6       7.178        35.7     0.51x
     256       1.438       178.1     146.0       7.247        35.3     0.20x
    1024       0.420       610.2     576.1       7.162        35.7     0.06x
```
Nearly identical to Scenario 1 (same number of comparisons).

### Scenario 4 — Diverse dst prefix lengths (N=1024)
```
  Tables    GPU Mpps   GPU µs/fr   kern µs    ACL Mpps   ACL µs/fr   GPU/ACL
       1       0.464       552.0     518.7       7.082        36.2     0.07x
       2       0.463       553.1     517.8       7.037        36.4     0.07x
       4       0.464       551.4     516.6       7.080        36.2     0.07x
       8       0.465       551.1     517.1       6.572        39.0     0.07x
      16       0.464       551.9     517.3       5.200        49.2     0.09x
      21       0.472       542.9     512.1       4.605        55.6     0.10x
```
GPU: flat ~551-553 µs/frame (always scans all 1024 rules)
ACL: rises at K≥8 (~1.0-1.3 µs/frame per additional probe, 3.5-5 ns/pkt/probe)

---

## 9. Kernel Optimisation History

| Stage | No-match 1024r kern µs | First-match 1024r kern µs | Change |
|-------|----------------------|--------------------------|--------|
| Baseline (global mem, ld.global.nc) | 810 | 4 | — |
| + Tiled shmem (256-rule tiles, per-tile early-exit vote) | 590 (-27%) | 7 (+75%) | No-match speedup; regression on first-match |
| + Persistent shmem cache (full table, reload on rule_version change) | 576 (-29%) | **4 (-3%)** | Regression fixed; steady-state has zero rule loads |

### Design decisions
- On-demand kernel keeps lightweight tiled approach (20 KB shmem, no `cudaFuncSetAttribute`)
- Persistent kernel gets full 80 KB cache with lazy invalidation (hot path for sustained traffic)
- `cudaFuncSetAttribute` called only for persistent kernel, only once at init

---

## 10. Key VPP APIs

```c
vlib_get_buffers(vm, from, bufs, n_left)   // bulk buffer pointer resolution
vnet_feature_next_u16(next_index, b)       // advance feature arc per packet
vlib_buffer_enqueue_to_next(vm, node, from, nexts, n)  // dispatch frame
VNET_FEATURE_INIT(name, static) { ... }    // register on arc
vnet_feature_enable_disable(arc, node, sw_if_index, enable, 0, 0)
```

---

## 11. CUDA Patterns Used

```cuda
// Managed memory with NVLink-C2C hints
cudaMallocManaged(&ptr, size);
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, dev);
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);

// Atomic handshake (CPU side, GCC intrinsics)
__atomic_store_n(&ctrl->submit_seq, seq, __ATOMIC_SEQ_CST);
__atomic_load_n(&ctrl->done_seq, __ATOMIC_ACQUIRE);
__atomic_fetch_add(&ctrl->rule_version, 1u, __ATOMIC_RELEASE);

// Persistent kernel launch with large shmem
cudaFuncSetAttribute(gpu_classify_persistent,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    (int)GPU_CLASSIFY_PERSIST_SHMEM_BYTES);
gpu_classify_persistent<<<dim3(1), dim3(256),
    GPU_CLASSIFY_PERSIST_SHMEM_BYTES, stream>>>(ctrl, descs, results, rules);

// cuda::atomic_ref in device code (C++17 cuda::std)
cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ref(ctrl->submit_seq);
uint32_t seq = ref.load(cuda::std::memory_order_acquire);
```

---

## 12. Commit History (relevant)

```
4bc3588c0  gpu_classify: add p50/p99/p99.9 latency percentiles to show command
21b5b2cf0  gpu_classify: add GPU device info and kernel timing to show command
90a376155  gpu_classify: new plugin — GPU-accelerated IPv4 packet classification
43cbd3725  gpu_classify: persistent shmem cache — full table on-chip, reload on rule_version change
276e59d3d  gpu_classify: tiled shmem — 256-rule tiles with per-tile early-exit vote
```

---

## 13. Known Issues / Potential Next Steps

- **N>1 no-match is slower than ACL**: GPU does O(N) scan; ACL does O(1) hash lookup.
  For no-match, GPU only beats ACL at N=1. This is fundamental to the linear-scan design.
- **IPv6 extension headers**: current node assumes no extension headers (skips to offset 40 for L4).
  Extension headers would require parsing, which is a future improvement.
- **1024-rule limit**: `GPU_CLASSIFY_MAX_RULES=1024`; raise freely (only L2 cache pressure matters).
  shmem would need to grow beyond 80 KB; `cudaFuncSetAttribute` limit is 256 KB on Blackwell.
- **Bitmap/hash table acceleration**: could add GPU-side O(1) hash lookup to compete with ACL on no-match,
  but would add complexity and likely unnecessary for the target use case (first-match / small rule sets).
- **Benchmark test `TEST=test_gpu_classify_bench`**: Scenario 4 with N=1024 takes ~5 min to run.
  Don't kill VPP — it's not hanging, just slow due to 50,000 frames × 1024-rule GPU scan.
