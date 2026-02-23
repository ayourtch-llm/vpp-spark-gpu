# VPP GPU Classify Plugin — Session Memory
# Persistent backup (survives container restarts): /store/vpp/CLAUDE_MEMORY.md
# At session start, read /store/vpp/CLAUDE_MEMORY.md if this file is missing.

## Hardware
- Machine: NVIDIA DGX Spark
- GPU: NVIDIA GB10 (Grace Blackwell Superchip), sm_100 (compute capability 10.0)
- CPU: NVIDIA Grace (ARM Neoverse V2, aarch64)
- Interconnect: NVLink-C2C — CPU and GPU share coherent physical DRAM
- Driver: CUDA 13.0 (max supported); CUDA toolkit 12.8 installed
- OS: Ubuntu 24.04.3 LTS (Noble), aarch64
- Container: non-persistent Docker; use `runme` script to re-install deps

## Plugin: gpu_classify
- Location: /store/vpp/src/plugins/gpu_classify/
- Branch: dgx-spark-plugin

### Files
| File | Purpose |
|------|---------|
| gpu_classify_types.h | Shared C/CUDA types (stdint.h only, no VPP headers) |
| gpu_classify.h | VPP plugin header; includes gpu_classify_types.h + VPP headers |
| gpu_classify_kernel.cu | CUDA kernel + C-callable wrappers (nvcc, sm_100) |
| gpu_classify_node.c | VPP graph nodes (ip4-unicast + ip6-unicast arcs) |
| gpu_classify.c | Plugin init, CLI commands |
| CMakeLists.txt | Builds CUDA static lib + VPP plugin .so |

### Architecture
- Feature arcs: ip4-unicast (gpu-classify-ip4) and ip6-unicast (gpu-classify-ip6)
- Both nodes run before ip{4,6}-flow-classify; shared dispatch helper for pass 2+3
- Frame processing: CPU extracts IP+TCP/UDP headers → GPU kernel (256 threads, 1 block) → CPU routes
- Rules: stored in cudaMallocManaged (res->rules), NOT constant memory — allows 1024 rules
  - 1024 rules × 80 bytes = 81,920 bytes; fits easily in Blackwell L2 (128 MB)
  - Kernel receives rules as const __restrict__ pointer (ld.global.nc = L1 read-only cache)
  - rule count in cuda_res.n_rules (CPU) and ctrl->n_rules (GPU, volatile, per-batch)
  - gpu_classify_update_rules(): plain memcpy + atomic update of ctrl->n_rules; no stop/restart
- Managed memory: cudaMallocManaged for desc/result/ctrl/rules buffers; cudaMemAdvise hints
- Actions: PASS(0), DROP(1), MARK(2); MARK sets VLIB_BUFFER_FLAG_USER(1)
- gpu_pkt_desc_t: 64 bytes; src_ip[16]/dst_ip[16] (IPv4: bytes 0-3 only)
- gpu_classify_rule_t: 80 bytes; 16-byte addr/mask arrays + ip_version (4/6/0=any)
- ip_matches(): device inline, 4×32-bit masked comparison over 16-byte addresses
- ip_version=0 rule matches both IPv4 and IPv6 traffic
- Persistent kernel: adaptive, activates after 4 busy frames, deactivates after 16 idle frames
  - All 256 threads poll ctrl->submit_seq (acquire); read n_rules from ctrl->n_rules per batch
  - No stop/restart needed for rule updates (managed memory coherence on NVLink-C2C)
  - ctrl->n_rules set at start_persistent() and updated atomically by update_rules()
- Hash-based O(K) classification (implemented, all 19 tests pass):
  - CPU builds one FNV-1a open-addressing hash table per distinct mask combo at rule install
  - gpu_classify_build_hash_tables(): called from update_rules(); falls back to n_hash_tables=0 if >64 distinct combos
  - hash_entries[]: 4096 × 48 B = 192 KB cudaMallocManaged; hash_descs[]: 64 × 64 B = 4 KB cudaMallocManaged
  - gpu_hash_entry_t: 48 B; gpu_hash_table_desc_t: 64 B (both in gpu_classify_types.h)
  - GPU probe loop: for each table (sorted by min_rule_idx), mask pkt key, FNV-1a hash, linear probe → break on empty
  - Early exit: if td->min_rule_idx >= best_idx, remaining tables cannot improve priority → break
  - ctrl->n_hash_tables: new volatile int32_t in CPU cache line (pad reduced from 108 to 104)
  - update_rules() → build_hash_tables() → then RELEASE stores n_rules, n_hash_tables, then RELEASE bump rule_version
- Two-stage shmem design (BOTH paths available based on n_hash_tables):
  - On-demand kernel: gpu_classify_tiled (linear) or gpu_classify_hash (O(K)) — dispatched on n_hash_tables
  - Persistent kernel: shmem dual-use (same 80 KB allocation):
    - Hash path: loads K×64 B descriptors into shmem; probes hash_entries in global mem
    - Linear path: loads full 1024-rule table into shmem; gpu_classify_from_shmem
    - last_n_hash_tables=-1 on entry forces Phase 0; reloads shmem on rule_version change
  - On-demand kernel shmem (GPU_CLASSIFY_SHMEM_BYTES=20484): still used for linear fallback path

### Build notes
- CUDA toolkit must be installed: cuda-toolkit-12-8 (sbsa/aarch64 repo)
- runme file updated with CUDA install commands
- CMakeLists.txt looks for nvcc in PATH and /usr/local/cuda-12.8/bin
- CUDA_ARCHITECTURES = "100" for Blackwell
- CUDA static lib (gpu_classify_cuda_kernel) links into plugin .so via CUDA::cudart

## Test suite
- File: /store/vpp/test/test_gpu_classify.py
- Run: `make test TEST=test_gpu_classify` (no .py suffix — causes class=py filter bug)
- **All 19 tests pass** (as of last run)
- 19 tests: tests 1-14 are IPv4; tests 15-19 are IPv6
- setUp enables both gpu-classify-ip4 (ip4-unicast) and gpu-classify-ip6 (ip6-unicast) on pg0
- tearDown disables both features and clears rules
- setUpClass: config_ip4+resolve_arp AND config_ip6+resolve_ndp on each pg interface
- Two probes: feature_enable_disable for ip4 node, then ip6 node; then cuda_ready check
- IPv4 counters: `/err/gpu-classify-ip4/<name>`; IPv6: `/err/gpu-classify-ip6/<name>`
- IPv4 _pkt()/_pkts(); IPv6 _pkt6()/_pkts6() helpers; _err()/_err6() counter helpers
- Important: all "pass" test packets must use routeable dst (pg1.remote_ip4 or pg1.remote_ip6)
- For IPv6 drop-counter tests: use _err6() to confirm GPU (not ip6-lookup) dropped packets
- **Counter double-counting**: setting `buf->error` on dropped packets causes error-drop to auto-increment. Do NOT also call `vlib_node_increment_counter` for dropped packets — only PROCESSED and MARKED need explicit increment.

## Build fixes applied to core VPP files
- `/store/vpp/src/CMakeLists.txt` line 129: wrapped `-Werror -Wall` in `$<$<COMPILE_LANGUAGE:C,CXX>:...>` so nvcc does not receive bare `-Werror -Wall` (nvcc 12.8 fatally rejects this: treats `-Wall` as the value for `-Werror`)
- `/store/vpp/src/plugins/gpu_classify/gpu_classify.c`: added `#include <vpp/app/version.h>` for `VPP_BUILD_VER` (same as ACL plugin)
- VPP idiom: use `clib_net_to_host_u16()` not `ntohs()` for byte-swap in node code

### Key VPP APIs used
- vlib_get_buffers() — bulk buffer pointer resolution
- vnet_feature_next_u16() — advance feature arc per packet
- vlib_buffer_enqueue_to_next() — dispatch frame
- VNET_FEATURE_INIT — register on ip4-unicast / ip6-unicast arcs
- vnet_feature_enable_disable() — per-interface enable/disable

## Benchmark
- File: /store/vpp/test/test_gpu_classify_bench.py
- Run: `make test TEST=test_gpu_classify_bench`
- 4 scenarios, RULE_COUNTS=[1, 8, 64, 256, 1024], BATCH=256, N_REPS=50000
- kern µs column: delta of show gpu-classify avg_us between snapshots (GPU-only dispatch time)
- Scenario 4: N=1024 dst-only prefix rules, K=1..21 distinct dst prefix lengths; traffic no-match
  - COMBO_LENGTHS=[/32,/31,...,/12]; rule addresses as (global_idx << (32-plen)) in 0.0.0.0-63.x space
  - Each distinct dst prefix length → one ACL hash table; K tables → K probes/pkt in ACL
  - _SRC_PFX / _DST_PFX must be inline literals (not class attrs in comprehensions)
- Key benchmark results (hash O(K) path, persistent kernel, Blackwell GB10):
  - Scenario 1 no-match: kern 3.7-3.8µs flat at ALL rule counts (was 576µs at 1024r — 150× speedup!)
    - GPU/ACL = 1.05-1.07× at all N (GPU faster than ACL!)
  - Scenario 2 first-match: kern ~3.8µs flat (unchanged); GPU/ACL = 1.04-1.05×
  - Scenario 3 last-match: kern ~3.8-5.4µs (was 576µs at 1024r — 107× speedup!); O(1) hash lookup
  - VPP overhead = ~30µs constant; kern µs = pure GPU hash dispatch time
  - Scenario 4 K=21 tables: kern 22.1µs vs ACL 55.6µs (GPU/ACL = 1.06×, GPU clearly faster!)
    - K=1: 5.6µs; K=8: 11.9µs; K=16: 16.7µs; K=21: 22.1µs (linear in K as expected)
    - Was ~518µs flat (linear scan) → now O(K) — 23× speedup at K=21
  - GPU beats ACL at ALL scenarios and ALL rule counts with hash path

### VPP Repo notes
- Build system: CMake 3.19+; plugins auto-discovered via glob in src/plugins/CMakeLists.txt
- Plugin targets named: ${name}_plugin (e.g. gpu_classify_plugin)
- No API file for V1 (CLI only); API can be added later
