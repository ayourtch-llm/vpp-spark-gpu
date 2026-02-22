/* SPDX-License-Identifier: Apache-2.0
 * GPU Packet Classifier — CUDA kernel and C-callable wrappers.
 *
 * Target: NVIDIA Blackwell GB10 (DGX Spark), sm_100.
 *
 * Architecture highlights exploited here:
 *
 *  NVLink-C2C (Grace ↔ Blackwell):
 *    CPU and GPU share the same physical DRAM with full cache coherency.
 *    cudaMallocManaged() buffers are accessible from both sides with
 *    ~no migration cost; cudaMemAdvise() is used as a placement hint only.
 *
 *  Constant memory (64 KB, broadcast):
 *    All 32 threads in a warp read the same rule in the same cycle via
 *    a single broadcast — no bank conflicts, effectively "free" reads.
 *    64 rules × 80 bytes = 5 120 bytes consumed.
 *
 *  One thread per packet:
 *    Block = 256 threads (== VLIB_FRAME_SIZE).  Grid = 1 block per call.
 *    Each thread independently walks the rule list and writes one result.
 */

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Pull in only the shared type definitions — no VPP headers needed on
 * the device-code compilation path.                                   */
#include "gpu_classify_types.h"

/* ================================================================== */
/* Device-side data                                                    */
/* ================================================================== */

/** Rules in constant memory.  Written once (or rarely) from the host
 *  via cudaMemcpyToSymbol; read every invocation by every thread.    */
__constant__ gpu_classify_rule_t d_rules[GPU_CLASSIFY_MAX_RULES];
__constant__ int		 d_n_rules;

/* ================================================================== */
/* Device-side helpers                                                 */
/* ================================================================== */

/**
 * @brief Test whether a 16-byte IP address matches a prefix.
 *
 * Performs four 32-bit masked comparisons covering the full 128-bit
 * address.  When mask words are zero the corresponding comparison is
 * always true, so a fully-zero mask unconditionally matches (wildcard).
 *
 * Both IPv4 and IPv6 use this helper:
 *  - IPv4: only bytes 0–3 are non-zero; bytes 4–15 of mask and addr
 *    are 0, so the upper comparisons are always true.
 *  - IPv6: all 16 bytes participate.
 *
 * @param pkt   16-byte packet address (must be 4-byte aligned).
 * @param addr  16-byte rule address   (must be 4-byte aligned).
 * @param mask  16-byte prefix mask    (must be 4-byte aligned).
 * @return      1 if (pkt & mask) == addr, 0 otherwise.
 */
__device__ __forceinline__ static int
ip_matches (const uint8_t *pkt, const uint8_t *addr, const uint8_t *mask)
{
  const uint32_t *p = (const uint32_t *) pkt;
  const uint32_t *a = (const uint32_t *) addr;
  const uint32_t *m = (const uint32_t *) mask;
  return ((p[0] & m[0]) == a[0]) & ((p[1] & m[1]) == a[1]) &
	 ((p[2] & m[2]) == a[2]) & ((p[3] & m[3]) == a[3]);
}

/* ================================================================== */
/* Kernel                                                              */
/* ================================================================== */

/**
 * @brief Classify one VPP frame of packets in parallel.
 *
 * @param descs     Array of GPU_CLASSIFY_MAX_FRAME packet descriptors
 *                  populated by the host.
 * @param results   Output array; kernel writes one GPU_CLASSIFY_ACTION_*
 *                  byte per packet (and GPU_CLASSIFY_ACTION_PASS for
 *                  unused slots beyond n_packets).
 * @param n_packets Number of live packets in this invocation (≤ MAX_FRAME).
 */
__global__ void
gpu_classify_kernel (const gpu_pkt_desc_t *__restrict__ descs,
		     uint8_t *__restrict__ results, int n_packets)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  /* Threads past the live range fill their result slot with PASS so
   * the host loop can iterate unconditionally over GPU_CLASSIFY_MAX_FRAME. */
  if (tid >= n_packets)
    {
      if (tid < GPU_CLASSIFY_MAX_FRAME)
	results[tid] = GPU_CLASSIFY_ACTION_PASS;
      return;
    }

  const gpu_pkt_desc_t *d      = &descs[tid];
  uint8_t		action = GPU_CLASSIFY_ACTION_PASS; /* default */

  /*
   * Walk rules sequentially; first match wins.
   *
   * All threads in a warp visit the same rule index in lockstep.
   * d_rules is in constant memory → broadcast read (single transaction
   * for the entire warp, zero latency after L1 cache warm-up).
   *
   * Divergence due to 'continue' only affects per-thread predicate
   * evaluation, not the memory access pattern — all warps keep pace.
   */
  for (int i = 0; i < d_n_rules; i++)
    {
      const gpu_classify_rule_t *r = &d_rules[i];

      /* ---- Protocol ------------------------------------------------ */
      if (r->proto != 0 && r->proto != d->ip_proto)
	continue;

      /* ---- IP version ---------------------------------------------- */
      if (r->ip_version != 0 && r->ip_version != d->ip_version)
	continue;

      /* ---- Source IP prefix ---------------------------------------- */
      if (!ip_matches (d->src_ip, r->src_addr, r->src_mask))
	continue;

      /* ---- Destination IP prefix ------------------------------------ */
      if (!ip_matches (d->dst_ip, r->dst_addr, r->dst_mask))
	continue;

      /* ---- Source port --------------------------------------------- */
      if (r->src_port != 0 && r->src_port != d->src_port)
	continue;

      /* ---- Destination port ---------------------------------------- */
      if (r->dst_port != 0 && r->dst_port != d->dst_port)
	continue;

      /* ---- TCP flags ----------------------------------------------- */
      if (r->tcp_flags_mask != 0 &&
	  (d->tcp_flags & r->tcp_flags_mask) != r->tcp_flags_val)
	continue;

      /* All predicates satisfied → apply rule action. */
      action = r->action;
      break;
    }

  results[tid] = action;
}

/* ================================================================== */
/* Host-side helpers                                                   */
/* ================================================================== */

/**
 * Map a kernel duration in microseconds to a log2-us histogram bucket.
 *
 *   bucket  0 : [    0,    1) us
 *   bucket  k : [ 2^(k-1), 2^k ) us   for k = 1 … GPU_CLASSIFY_LAT_BUCKETS-2
 *   bucket 11 : [ 1024,  ∞ ) us   (overflow)
 */
static int
lat_us_to_bucket (float us)
{
  if (us < 1.0f)
    return 0;
  int   b     = 1;
  float bound = 2.0f;
  while (bound <= us && b < GPU_CLASSIFY_LAT_BUCKETS - 1)
    {
      bound *= 2.0f;
      b++;
    }
  return b;
}

/* ================================================================== */
/* C-callable host wrappers                                           */
/* ================================================================== */

extern "C"
{

  /* ---------------------------------------------------------------- */
  int
  gpu_classify_cuda_init (gpu_classify_cuda_res_t *res)
  {
    cudaError_t err;

    /* Create a dedicated CUDA stream for ordered asynchronous dispatch. */
    err =
      cudaStreamCreate (reinterpret_cast<cudaStream_t *> (&res->stream));
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaStreamCreate: %s\n",
		 cudaGetErrorString (err));
	return -1;
      }

    /* Allocate unified memory for packet descriptors.
     *
     * On Grace Blackwell (NVLink-C2C) this memory lives in the shared
     * physical DRAM; both CPU and GPU access it natively without page
     * migration.  cudaMemAdvise is a soft placement hint only.        */
    err = cudaMallocManaged (reinterpret_cast<void **> (&res->descs),
			     GPU_CLASSIFY_MAX_FRAME * sizeof (gpu_pkt_desc_t));
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaMallocManaged (descs): %s\n",
		 cudaGetErrorString (err));
	goto fail_stream;
      }

    err = cudaMallocManaged (reinterpret_cast<void **> (&res->results),
			     GPU_CLASSIFY_MAX_FRAME * sizeof (uint8_t));
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaMallocManaged (results): %s\n",
		 cudaGetErrorString (err));
	goto fail_descs;
      }

    /* Hint preferred locations:
     *   descs   → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   results → GPU writes, CPU reads  → prefer GPU-side DRAM.     */
    cudaMemAdvise (res->descs,
		   GPU_CLASSIFY_MAX_FRAME * sizeof (gpu_pkt_desc_t),
		   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise (res->results, GPU_CLASSIFY_MAX_FRAME * sizeof (uint8_t),
		   cudaMemAdviseSetPreferredLocation, 0 /* device 0 */);

    /* Create CUDA events used for per-frame kernel timing. */
    cudaEvent_t ev_start, ev_stop;
    err = cudaEventCreate (&ev_start);
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaEventCreate (start): %s\n",
		 cudaGetErrorString (err));
	goto fail_descs;
      }
    err = cudaEventCreate (&ev_stop);
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaEventCreate (stop): %s\n",
		 cudaGetErrorString (err));
	cudaEventDestroy (ev_start);
	goto fail_descs;
      }
    res->ev_start = reinterpret_cast<void *> (ev_start);
    res->ev_stop  = reinterpret_cast<void *> (ev_stop);

    /* Initialise statistics. */
    res->n_kernel_calls  = 0;
    res->n_gpu_packets   = 0;
    res->total_kernel_ms = 0.0f;
    res->min_kernel_ms   = FLT_MAX;
    res->max_kernel_ms   = 0.0f;

    /* Seed constant memory with zero rules. */
    {
      int zero = 0;
      cudaMemcpyToSymbol (d_n_rules, &zero, sizeof (int), 0,
			  cudaMemcpyHostToDevice);
    }

    return 0;

  fail_descs:
    cudaFree (res->descs);
    res->descs = nullptr;
  fail_stream:
    cudaStreamDestroy (reinterpret_cast<cudaStream_t> (res->stream));
    res->stream = nullptr;
    return -1;
  }

  /* ---------------------------------------------------------------- */
  void
  gpu_classify_cuda_cleanup (gpu_classify_cuda_res_t *res)
  {
    if (res->ev_stop)
      {
	cudaEventDestroy (reinterpret_cast<cudaEvent_t> (res->ev_stop));
	res->ev_stop = nullptr;
      }
    if (res->ev_start)
      {
	cudaEventDestroy (reinterpret_cast<cudaEvent_t> (res->ev_start));
	res->ev_start = nullptr;
      }
    if (res->results)
      {
	cudaFree (res->results);
	res->results = nullptr;
      }
    if (res->descs)
      {
	cudaFree (res->descs);
	res->descs = nullptr;
      }
    if (res->stream)
      {
	cudaStreamDestroy (reinterpret_cast<cudaStream_t> (res->stream));
	res->stream = nullptr;
      }
  }

  /* ---------------------------------------------------------------- */
  int
  gpu_classify_update_rules (gpu_classify_rule_t *rules, uint32_t n_rules)
  {
    if (n_rules > GPU_CLASSIFY_MAX_RULES)
      return -1;

    cudaError_t err;

    if (n_rules > 0)
      {
	err =
	  cudaMemcpyToSymbol (d_rules, rules,
			      n_rules * sizeof (gpu_classify_rule_t), 0,
			      cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	  return -1;
      }

    int n = static_cast<int> (n_rules);
    err =
      cudaMemcpyToSymbol (d_n_rules, &n, sizeof (int), 0,
			  cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : -1;
  }

  /* ---------------------------------------------------------------- */
  int
  gpu_classify_launch_kernel (gpu_classify_cuda_res_t *res, uint32_t n_packets)
  {
    if (n_packets == 0)
      return 0;

    cudaStream_t stream   = reinterpret_cast<cudaStream_t> (res->stream);
    cudaEvent_t  ev_start = reinterpret_cast<cudaEvent_t> (res->ev_start);
    cudaEvent_t  ev_stop  = reinterpret_cast<cudaEvent_t> (res->ev_stop);

    /* One block of 256 threads — one thread per packet slot.
     * Threads in slots [n_packets, 255] write PASS to their result.  */
    dim3 block (GPU_CLASSIFY_MAX_FRAME);
    dim3 grid (1);

    cudaEventRecord (ev_start, stream);
    gpu_classify_kernel<<<grid, block, 0, stream>>> (res->descs, res->results,
						     static_cast<int> (
						       n_packets));
    cudaEventRecord (ev_stop, stream);

    /* Wait for all GPU writes to be visible to the CPU.
     * On NVLink-C2C the synchronise cost is very low (~microseconds)
     * compared to the PCIe round-trip on discrete GPU systems.       */
    cudaError_t err = cudaStreamSynchronize (stream);
    if (err != cudaSuccess)
      return -1;

    /* Accumulate per-frame timing statistics. */
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime (&elapsed_ms, ev_start, ev_stop);

    res->n_kernel_calls++;
    res->n_gpu_packets += n_packets;
    res->total_kernel_ms += elapsed_ms;
    if (elapsed_ms < res->min_kernel_ms)
      res->min_kernel_ms = elapsed_ms;
    if (elapsed_ms > res->max_kernel_ms)
      res->max_kernel_ms = elapsed_ms;

    res->lat_hist[lat_us_to_bucket (elapsed_ms * 1000.0f)]++;

    return 0;
  }

  /* ---------------------------------------------------------------- */
  int
  gpu_classify_get_device_info (gpu_classify_device_info_t *info)
  {
    int device = 0;
    if (cudaGetDevice (&device) != cudaSuccess)
      return -1;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties (&prop, device) != cudaSuccess)
      return -1;

    strncpy (info->name, prop.name, sizeof (info->name) - 1);
    info->name[sizeof (info->name) - 1] = '\0';
    info->compute_major      = prop.major;
    info->compute_minor      = prop.minor;
    info->total_mem_bytes    = (uint64_t) prop.totalGlobalMem;
    info->sm_count           = prop.multiProcessorCount;
    info->clock_rate_khz     = prop.clockRate;
    info->mem_clock_rate_khz = prop.memoryClockRate;
    info->mem_bus_width_bits = prop.memoryBusWidth;
    return 0;
  }

} /* extern "C" */
