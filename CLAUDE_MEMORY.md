# VPP GPU Classify Plugin — Session Memory
# Persistent backup (survives container restarts): /store/vpp/CLAUDE_MEMORY.md
# At session start, read /store/vpp/CLAUDE_MEMORY.md if this file is missing.

## Hardware
- Machine: NVIDIA DGX Spark
- GPU: NVIDIA GB10 (Grace Blackwell Superchip), sm_100 (compute capability 10.0)
- CPU: NVIDIA Grace (ARM Neoverse V2, aarch64)
- Interconnect: NVLink-C2C — CPU and GPU share coherent physical DRAM
- Driver: CUDA 13.0 (max supported); CUDA toolkit 12.8 to be installed
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
- Rules: stored in GPU constant memory (64 rules × 80 bytes = 5120 bytes)
- Managed memory: cudaMallocManaged for desc/result buffers; cudaMemAdvise for NVLink-C2C hints
- Actions: PASS(0), DROP(1), MARK(2); MARK sets VLIB_BUFFER_FLAG_USER(1)
- gpu_pkt_desc_t: 64 bytes; src_ip[16]/dst_ip[16] (IPv4: bytes 0-3 only)
- gpu_classify_rule_t: 80 bytes; 16-byte addr/mask arrays + ip_version (4/6/0=any)
- ip_matches(): device inline, 4×32-bit masked comparison over 16-byte addresses
- ip_version=0 rule matches both IPv4 and IPv6 traffic

### Build notes
- CUDA toolkit must be installed: cuda-toolkit-12-8 (sbsa/aarch64 repo)
- runme file updated with CUDA install commands
- CMakeLists.txt looks for nvcc in PATH and /usr/local/cuda-12.8/bin
- CUDA_ARCHITECTURES = "100" for Blackwell
- CUDA static lib (gpu_classify_cuda_kernel) links into plugin .so via CUDA::cudart

## Test suite
- File: /store/vpp/test/test_gpu_classify.py
- Run: `make test TEST=test_gpu_classify.py`
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

### VPP Repo notes
- Build system: CMake 3.19+; plugins auto-discovered via glob in src/plugins/CMakeLists.txt
- Plugin targets named: ${name}_plugin (e.g. gpu_classify_plugin)
- No API file for V1 (CLI only); API can be added later
