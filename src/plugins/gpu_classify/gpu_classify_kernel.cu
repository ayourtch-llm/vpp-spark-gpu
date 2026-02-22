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
 *  Managed-memory rule table (cudaMallocManaged, const __restrict__):
 *    Rules are stored in NVLink-C2C managed memory, passed to the kernel
 *    as a const __restrict__ pointer.  The compiler emits ld.global.nc
 *    (L1 read-only cache) loads; all threads in a warp accessing the same
 *    rule index receive a single broadcast read.  The Blackwell L2 cache
 *    (128 MB) easily fits the full rule table (1024 × 80 B = 80 KB).
 *    This approach removes the 64 KB constant-memory limit and eliminates
 *    the cudaMemcpyToSymbol null-stream deadlock that affected the old
 *    __constant__ design.
 *
 *  One thread per packet:
 *    Block = 256 threads (== VLIB_FRAME_SIZE).  Grid = 1 block per call.
 *    Each thread independently walks the rule list and writes one result.
 *
 *  Persistent kernel (adaptive):
 *    A single block stays resident on the GPU and polls a shared control
 *    block for work.  This eliminates the cudaStreamSynchronize round-trip
 *    (~30-35 µs) that dominates the on-demand path.  The kernel activates
 *    after GPU_CLASSIFY_PERSIST_START_FRAMES consecutive "busy" frames
 *    (≥ GPU_CLASSIFY_PERSIST_MIN_PKTS packets) and deactivates after
 *    GPU_CLASSIFY_PERSIST_STOP_FRAMES consecutive "idle" frames, saving
 *    SM resources and power during sparse traffic.
 */

#include <cuda_runtime.h>
#include <cuda/atomic>   /* cuda::atomic_ref, thread_scope_system (C++17 / libcudacxx) */
#include <float.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Pull in only the shared type definitions — no VPP headers needed on
 * the device-code compilation path.                                   */
#include "gpu_classify_types.h"

/* ================================================================== */
/* Device-side data                                                    */
/* ================================================================== */
/*
 * Rules and rule count are no longer stored in __constant__ memory.
 * They live in cudaMallocManaged memory, passed to each kernel as a
 * const __restrict__ pointer (ld.global.nc = L1 read-only cache path)
 * and an int.  This removes the 64 KB constant-memory limit (allows
 * up to GPU_CLASSIFY_MAX_RULES = 1024 rules) and eliminates the
 * cudaMemcpyToSymbol null-stream deadlock.
 */

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

/**
 * @brief Apply the rule table to a single packet descriptor.
 *
 * Extracted as a shared device inline so both the on-demand kernel
 * and the persistent kernel can call it without code duplication.
 *
 * @param d       Pointer to the packet descriptor to classify.
 * @param rules   Rule table (cudaMallocManaged, const __restrict__).
 * @param n_rules Number of active rules in the table.
 * @return        GPU_CLASSIFY_ACTION_* value for the first matching rule,
 *                or GPU_CLASSIFY_ACTION_PASS if no rule matches.
 */
__device__ __forceinline__ static uint8_t
gpu_classify_match_packet (const gpu_pkt_desc_t *d,
			   const gpu_classify_rule_t *__restrict__ rules,
			   int n_rules)
{
  uint8_t action = GPU_CLASSIFY_ACTION_PASS; /* default: no match */

  /*
   * Walk rules sequentially; first match wins.
   *
   * All threads in a warp visit the same rule index in lockstep.
   * rules is marked const __restrict__: the compiler emits ld.global.nc
   * (L1 read-only cache) loads, which broadcast to all threads in a warp
   * for the same index — zero bank-conflict penalty.
   *
   * Divergence due to 'continue' only affects per-thread predicate
   * evaluation, not the memory access pattern — all warps keep pace.
   */
  for (int i = 0; i < n_rules; i++)
    {
      const gpu_classify_rule_t *r = &rules[i];

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

  return action;
}

/* ================================================================== */
/* Kernels                                                             */
/* ================================================================== */

/**
 * @brief Classify one VPP frame of packets in parallel (on-demand).
 *
 * @param descs     Array of GPU_CLASSIFY_MAX_FRAME packet descriptors
 *                  populated by the host.
 * @param results   Output array; kernel writes one GPU_CLASSIFY_ACTION_*
 *                  byte per packet (and GPU_CLASSIFY_ACTION_PASS for
 *                  unused slots beyond n_packets).
 * @param n_packets Number of live packets in this invocation (≤ MAX_FRAME).
 * @param rules     Rule table (cudaMallocManaged, const __restrict__).
 * @param n_rules   Number of active rules.
 */
__global__ void
gpu_classify_kernel (const gpu_pkt_desc_t *__restrict__ descs,
		     uint8_t *__restrict__ results, int n_packets,
		     const gpu_classify_rule_t *__restrict__ rules, int n_rules)
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

  results[tid] = gpu_classify_match_packet (&descs[tid], rules, n_rules);
}

/**
 * @brief Persistent GPU classifier — stays resident, polls for work.
 *
 * One block of GPU_CLASSIFY_MAX_FRAME threads is launched once and
 * loops indefinitely until the CPU sets ctrl->kill.  Thread 0 polls
 * ctrl->submit_seq for new work; when detected it broadcasts via shared
 * memory, all threads classify in parallel, then thread 0 signals
 * completion via ctrl->done_seq.
 *
 * Power efficiency: __nanosleep(100) in the poll loop lets the SM
 * deschedule the warp while waiting (~100 ns sleep interval).
 *
 * @param ctrl     CPU↔GPU handshake control block (managed memory).
 * @param descs    Packet descriptor array (managed memory, CPU-written).
 * @param results  Result array (managed memory, GPU-written).
 * @param rules    Rule table (cudaMallocManaged, const __restrict__).
 *                 Rule count is read per batch from ctrl->n_rules so
 *                 the CPU can update rules without stopping the kernel.
 */
__global__ void
gpu_classify_persistent (gpu_classify_ctrl_t *ctrl,
			 const gpu_pkt_desc_t *descs, uint8_t *results,
			 const gpu_classify_rule_t *__restrict__ rules)
{
  int tid = threadIdx.x; /* blockIdx.x == 0 always (single-block launch) */

  /*
   * System-scope atomic reference type for the handshake fields.
   *
   * cuda::thread_scope_system guarantees visibility across GPU and CPU
   * (required for NVLink-C2C / managed memory).  Plain volatile reads
   * only bypass GPU L1; they do NOT guarantee CPU writes are visible to
   * the GPU (ld.volatile is a weakly-ordered load per PTX ISA).
   *
   * memory_order_acquire on submit_seq load: once we see a new seq,
   * all CPU writes before the CPU's release-store (n_packets, kill) are
   * guaranteed visible to this GPU thread.
   *
   * memory_order_release on done_seq store: pairs with the CPU's
   * acquire-load.  The CPU is guaranteed to see all results[] writes
   * once it observes the new done_seq value.
   */
  using sys_u32 = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

  uint32_t last_seq = 0; /* tracks the last seq each thread processed  */

  for (;;)
    {
      /*
       * ---- Phase 1: ALL threads poll submit_seq in lockstep ----------
       *
       * IMPORTANT: all 256 threads participate in the poll — no
       * divergence before __syncthreads().  A design where only thread 0
       * polls while threads 1-255 block at __syncthreads() from a
       * different control-flow path is undefined in CUDA (all threads
       * must reach any __syncthreads() call from the same location).
       *
       * __nanosleep(100): power-efficient ~100 ns sleep; the SM
       * deschedules the warp so other warps (or the idle SM power gate)
       * can save energy while waiting.  Available on sm_70+.
       *
       * Acquire load: pairs with the CPU's SEQ_CST store of submit_seq,
       * guaranteeing that n_packets / kill written before the CPU's
       * release are visible to ALL GPU threads once they see the new seq.
       */
      uint32_t seq;
      do
	{
	  __nanosleep (100);
	  seq = sys_u32 (ctrl->submit_seq).load (cuda::memory_order_acquire);
	}
      while (seq == last_seq);

      last_seq = seq;

      /* ---- Phase 2: check kill flag — all threads exit cleanly ------- */
      if (ctrl->kill)
	return;

      /* ---- Phase 3: classify ---------------------------------------- */
      int n        = (int) ctrl->n_packets;
      int n_rules  = (int) ctrl->n_rules;
      if (tid < n)
	results[tid] = gpu_classify_match_packet (&descs[tid], rules, n_rules);
      else if (tid < GPU_CLASSIFY_MAX_FRAME)
	results[tid] = GPU_CLASSIFY_ACTION_PASS;

      /* ---- Phase 4: signal completion -------------------------------- */
      /*
       * __threadfence_system(): called by ALL threads to flush their
       * result writes out of the GPU L2 to be visible system-wide (CPU).
       *
       * __syncthreads(): all threads call this from the same location —
       * no divergence.  Ensures every thread has finished writing results
       * AND completed its __threadfence_system() before thread 0 signals.
       *
       * done_seq release-store (thread 0 only): pairs with the CPU's
       * acquire-load of done_seq.  The CPU is guaranteed to see all
       * results[] writes once it observes the new done_seq value.
       */
      __threadfence_system ();
      __syncthreads ();

      if (tid == 0)
	sys_u32 (ctrl->done_seq).store (seq, cuda::memory_order_release);
    }
}

/* ================================================================== */
/* Host-side helpers                                                   */
/* ================================================================== */

/**
 * Map a round-trip duration in microseconds to a log2-us histogram bucket.
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

/**
 * @brief Architecture-appropriate CPU spin-wait hint.
 *
 * On Grace (aarch64): YIELD instruction — signals the CPU pipeline that
 * this thread is in a spin-wait, reducing power and allowing SMT peers
 * to make progress.
 */
static inline void
host_cpu_pause (void)
{
#if defined(__aarch64__)
  __asm__ volatile ("yield" ::: "memory");
#elif defined(__x86_64__)
  __asm__ volatile ("pause" ::: "memory");
#else
  __asm__ volatile ("" ::: "memory"); /* compiler barrier only */
#endif
}

/* ================================================================== */
/* Persistent-kernel lifecycle helpers (C++ linkage, file-local)      */
/* ================================================================== */

/**
 * @brief Launch the persistent classification kernel.
 *
 * Resets the ctrl handshake block to zero, then asynchronously launches
 * gpu_classify_persistent into res->stream.  On success sets
 * res->persist_active = 1.
 */
static void
gpu_classify_start_persistent (gpu_classify_cuda_res_t *res)
{
  gpu_classify_ctrl_t *ctrl   = res->ctrl;
  cudaStream_t	       stream = reinterpret_cast<cudaStream_t> (res->stream);

  /* Reset handshake counters.  The persistent kernel starts with
   * last_seq = 0 so it must see submit_seq transition away from 0.
   * Use atomic stores for the non-volatile submit_seq / done_seq.   */
  __atomic_store_n (&ctrl->submit_seq, 0u, __ATOMIC_RELAXED);
  __atomic_store_n (&ctrl->done_seq,   0u, __ATOMIC_RELAXED);
  ctrl->kill	  = 0;
  ctrl->n_packets = 0;
  ctrl->n_rules   = (int32_t) res->n_rules;

  /* Full fence: all resets must be globally visible before the GPU
   * kernel executes its first polling iteration.                     */
  __atomic_thread_fence (__ATOMIC_SEQ_CST);

  gpu_classify_persistent<<<dim3 (1), dim3 (GPU_CLASSIFY_MAX_FRAME), 0,
			     stream>>> (ctrl, res->descs, res->results,
				       res->rules);

  cudaError_t err = cudaGetLastError ();
  if (err == cudaSuccess)
    {
      res->persist_active = 1;
    }
  else
    {
      fprintf (stderr, "gpu_classify: start_persistent launch failed: %s\n",
	       cudaGetErrorString (err));
    }
}

/**
 * @brief Stop the persistent classification kernel and wait for exit.
 *
 * Sets the kill flag and wakes the polling loop by incrementing
 * submit_seq, then blocks on cudaStreamSynchronize() until the kernel
 * has returned.  Resets persist_active and the hysteresis counters.
 */
static void
gpu_classify_stop_persistent (gpu_classify_cuda_res_t *res)
{
  gpu_classify_ctrl_t *ctrl   = res->ctrl;
  cudaStream_t	       stream = reinterpret_cast<cudaStream_t> (res->stream);

  /* 1. Signal kill (SEQ_CST so the seq increment below is ordered after). */
  __atomic_store_n (&ctrl->kill, 1u, __ATOMIC_SEQ_CST);

  /* 2. Increment submit_seq to wake the GPU polling loop.
   * Read submit_seq atomically (it's now a plain uint32_t accessed by
   * the GPU via cuda::atomic_ref; read with RELAXED since the SEQ_CST
   * store below already provides the necessary ordering).             */
  uint32_t cur_seq = __atomic_load_n (&ctrl->submit_seq, __ATOMIC_RELAXED);
  __atomic_store_n (&ctrl->submit_seq, cur_seq + 1, __ATOMIC_SEQ_CST);

  /* 3. Wait for the kernel to observe the kill flag and return. */
  cudaStreamSynchronize (stream);

  res->persist_active = 0;
  res->busy_frames    = 0;
  res->idle_frames    = 0;
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

    /* Allocate the persistent-kernel control block. */
    err = cudaMallocManaged (reinterpret_cast<void **> (&res->ctrl),
			     sizeof (gpu_classify_ctrl_t));
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaMallocManaged (ctrl): %s\n",
		 cudaGetErrorString (err));
	goto fail_results;
      }
    memset (res->ctrl, 0, sizeof (gpu_classify_ctrl_t));

    /* Allocate the rule table in managed memory.
     *
     * Rules are written by the CPU (CLI/API handlers) and read by the
     * GPU kernel.  On NVLink-C2C the preferred CPU-side location means
     * the CPU writes are cheap (same physical DRAM) and the GPU fetches
     * via the read-only L1 cache path (ld.global.nc).                */
    err = cudaMallocManaged (reinterpret_cast<void **> (&res->rules),
			     GPU_CLASSIFY_MAX_RULES *
			       sizeof (gpu_classify_rule_t));
    if (err != cudaSuccess)
      {
	fprintf (stderr, "gpu_classify: cudaMallocManaged (rules): %s\n",
		 cudaGetErrorString (err));
	goto fail_ctrl;
      }
    memset (res->rules, 0,
	    GPU_CLASSIFY_MAX_RULES * sizeof (gpu_classify_rule_t));
    res->n_rules = 0;

    /* Hint preferred locations:
     *   descs   → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   rules   → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   results → GPU writes, CPU reads  → prefer GPU-side DRAM.
     *   ctrl    → mixed read/write       → no preference hint.       */
    cudaMemAdvise (res->descs,
		   GPU_CLASSIFY_MAX_FRAME * sizeof (gpu_pkt_desc_t),
		   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise (res->rules,
		   GPU_CLASSIFY_MAX_RULES * sizeof (gpu_classify_rule_t),
		   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise (res->results, GPU_CLASSIFY_MAX_FRAME * sizeof (uint8_t),
		   cudaMemAdviseSetPreferredLocation, 0 /* device 0 */);

    /* Initialise statistics. */
    res->n_kernel_calls  = 0;
    res->n_gpu_packets   = 0;
    res->total_kernel_ms = 0.0f;
    res->min_kernel_ms   = FLT_MAX;
    res->max_kernel_ms   = 0.0f;
    res->persist_active  = 0;
    res->busy_frames     = 0;
    res->idle_frames     = 0;

    return 0;

  fail_ctrl:
    cudaFree (res->ctrl);
    res->ctrl = nullptr;
  fail_results:
    cudaFree (res->results);
    res->results = nullptr;
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
    /* Stop the persistent kernel before freeing any shared memory. */
    if (res->persist_active)
      gpu_classify_stop_persistent (res);

    if (res->rules)
      {
	cudaFree (res->rules);
	res->rules = nullptr;
      }
    if (res->ctrl)
      {
	cudaFree (res->ctrl);
	res->ctrl = nullptr;
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
  gpu_classify_update_rules (gpu_classify_cuda_res_t *res,
			     gpu_classify_rule_t *rules, uint32_t n_rules)
  {
    if (n_rules > GPU_CLASSIFY_MAX_RULES)
      return -1;

    /*
     * Rules live in cudaMallocManaged memory (res->rules), so updating
     * them is a plain CPU memcpy — no CUDA API calls, no null-stream
     * interference, no need to stop/restart the persistent kernel.
     *
     * On NVLink-C2C (Grace ↔ Blackwell) managed memory is physically
     * shared DRAM with full cache coherency, so the GPU sees the new
     * rules immediately via the hardware coherence protocol.
     *
     * The persistent kernel reads ctrl->n_rules once per batch
     * (volatile, after the acquire load of submit_seq).  We store
     * n_rules first, then update ctrl->n_rules with a release fence,
     * so the kernel is guaranteed to see the new table on its next batch.
     */
    if (n_rules > 0)
      memcpy (res->rules, rules, n_rules * sizeof (gpu_classify_rule_t));

    res->n_rules = (int) n_rules;

    /* If the persistent kernel is live, propagate the new count. */
    if (res->persist_active)
      __atomic_store_n (&res->ctrl->n_rules, (int32_t) n_rules,
			__ATOMIC_RELEASE);

    return 0;
  }

  /* ---------------------------------------------------------------- */
  /**
   * @brief Dispatch one VPP frame to the GPU for classification.
   *
   * Adaptively switches between two dispatch modes:
   *
   *  On-demand (default): launches gpu_classify_kernel, calls
   *    cudaStreamSynchronize(), measures the full CPU round-trip.
   *    Simple, low-overhead for sparse traffic.
   *
   *  Persistent (activated after START_FRAMES busy frames): writes
   *    ctrl->n_packets and increments ctrl->submit_seq, then spins on
   *    ctrl->done_seq.  Eliminates the kernel-launch + synchronise
   *    overhead — the GPU block stays resident and begins classifying
   *    as soon as it polls the new submit_seq.
   *
   * Both paths measure the CPU-observed round-trip time via POSIX
   * clock_gettime(CLOCK_MONOTONIC).
   */
  int
  gpu_classify_launch_kernel (gpu_classify_cuda_res_t *res, uint32_t n_packets)
  {
    if (n_packets == 0)
      return 0;

    /* ---- Adaptive hysteresis -------------------------------------- */
    if (n_packets >= GPU_CLASSIFY_PERSIST_MIN_PKTS)
      {
	res->busy_frames++;
	res->idle_frames = 0;
      }
    else
      {
	res->idle_frames++;
	res->busy_frames = 0;
      }

    /* Activate persistent kernel once we see sustained busy traffic. */
    if (!res->persist_active &&
	res->busy_frames >= GPU_CLASSIFY_PERSIST_START_FRAMES)
      gpu_classify_start_persistent (res);

    /* Deactivate once traffic drops (saves SM resources / power). */
    if (res->persist_active &&
	res->idle_frames >= GPU_CLASSIFY_PERSIST_STOP_FRAMES)
      gpu_classify_stop_persistent (res);

    /* ---- Dispatch ------------------------------------------------- */
    struct timespec t0, t1;
    clock_gettime (CLOCK_MONOTONIC, &t0);

    if (res->persist_active)
      {
	/*
	 * Persistent path:
	 *   1. Write n_packets (must be visible before submit_seq).
	 *   2. Increment submit_seq with a full fence to signal the GPU.
	 *   3. Spin-poll done_seq with a CPU yield hint.
	 */
	gpu_classify_ctrl_t *ctrl = res->ctrl;
	uint32_t	     new_seq =
	  __atomic_load_n (&ctrl->submit_seq, __ATOMIC_RELAXED) + 1;

	ctrl->n_packets = (int32_t) n_packets;
	__atomic_store_n (&ctrl->submit_seq, new_seq, __ATOMIC_SEQ_CST);

	while (__atomic_load_n (&ctrl->done_seq, __ATOMIC_ACQUIRE) != new_seq)
	  host_cpu_pause ();
      }
    else
      {
	/*
	 * On-demand path: launch the one-shot kernel, then synchronise.
	 * One block of 256 threads — one thread per packet slot.
	 * Threads in slots [n_packets, 255] write PASS to their result.
	 */
	cudaStream_t stream = reinterpret_cast<cudaStream_t> (res->stream);
	dim3	     block (GPU_CLASSIFY_MAX_FRAME);
	dim3	     grid (1);

	gpu_classify_kernel<<<grid, block, 0, stream>>> (
	  res->descs, res->results, static_cast<int> (n_packets),
	  res->rules, res->n_rules);

	cudaError_t err = cudaStreamSynchronize (stream);
	if (err != cudaSuccess)
	  return -1;
      }

    clock_gettime (CLOCK_MONOTONIC, &t1);

    float elapsed_ms =
      (float) ((t1.tv_sec - t0.tv_sec) * 1000.0 +
		(t1.tv_nsec - t0.tv_nsec) / 1.0e6);

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
