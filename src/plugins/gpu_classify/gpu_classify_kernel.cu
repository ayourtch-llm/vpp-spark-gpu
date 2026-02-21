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
 *    64 rules × 24 bytes = 1 536 bytes consumed.
 *
 *  One thread per packet:
 *    Block = 256 threads (== VLIB_FRAME_SIZE).  Grid = 1 block per call.
 *    Each thread independently walks the rule list and writes one result.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

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

      /* ---- Source IP prefix ---------------------------------------- */
      if (r->src_mask != 0 && (d->src_ip4 & r->src_mask) != r->src_addr)
	continue;

      /* ---- Destination IP prefix ------------------------------------ */
      if (r->dst_mask != 0 && (d->dst_ip4 & r->dst_mask) != r->dst_addr)
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

    cudaStream_t stream = reinterpret_cast<cudaStream_t> (res->stream);

    /* One block of 256 threads — one thread per packet slot.
     * Threads in slots [n_packets, 255] write PASS to their result.  */
    dim3 block (GPU_CLASSIFY_MAX_FRAME);
    dim3 grid (1);

    gpu_classify_kernel<<<grid, block, 0, stream>>> (res->descs, res->results,
						     static_cast<int> (
						       n_packets));

    /* Wait for all GPU writes to be visible to the CPU.
     * On NVLink-C2C the synchronise cost is very low (~microseconds)
     * compared to the PCIe round-trip on discrete GPU systems.       */
    cudaError_t err = cudaStreamSynchronize (stream);
    return (err == cudaSuccess) ? 0 : -1;
  }

} /* extern "C" */
