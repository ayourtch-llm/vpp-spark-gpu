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

/**
 * @brief Apply the rule table to a single packet descriptor.
 *
 * Extracted as a shared device inline so both the on-demand kernel
 * and the persistent kernel can call it without code duplication.
 *
 * @param d  Pointer to the packet descriptor to classify.
 * @return   GPU_CLASSIFY_ACTION_* value for the first matching rule,
 *           or GPU_CLASSIFY_ACTION_PASS if no rule matches.
 */
__device__ __forceinline__ static uint8_t
gpu_classify_match_packet (const gpu_pkt_desc_t *d)
{
  uint8_t action = GPU_CLASSIFY_ACTION_PASS; /* default: no match */

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

  results[tid] = gpu_classify_match_packet (&descs[tid]);
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
 */
__global__ void
gpu_classify_persistent (gpu_classify_ctrl_t *ctrl,
			 const gpu_pkt_desc_t *descs, uint8_t *results)
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
      int n = (int) ctrl->n_packets;
      if (tid < n)
	results[tid] = gpu_classify_match_packet (&descs[tid]);
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

  /* Full fence: all resets must be globally visible before the GPU
   * kernel executes its first polling iteration.                     */
  __atomic_thread_fence (__ATOMIC_SEQ_CST);

  gpu_classify_persistent<<<dim3 (1), dim3 (GPU_CLASSIFY_MAX_FRAME), 0,
			     stream>>> (ctrl, res->descs, res->results);

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

    /* Hint preferred locations:
     *   descs   → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   results → GPU writes, CPU reads  → prefer GPU-side DRAM.
     *   ctrl    → mixed read/write       → no preference hint.       */
    cudaMemAdvise (res->descs,
		   GPU_CLASSIFY_MAX_FRAME * sizeof (gpu_pkt_desc_t),
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

    /* Seed constant memory with zero rules. */
    {
      int zero = 0;
      cudaMemcpyToSymbol (d_n_rules, &zero, sizeof (int), 0,
			  cudaMemcpyHostToDevice);
    }

    return 0;

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
     * cudaMemcpyToSymbol runs on the null (default) CUDA stream, which
     * implicitly synchronises with all blocking streams before executing.
     * If the persistent kernel is running on our named blocking stream,
     * the null-stream memcpy would wait forever for it to finish.
     *
     * Additionally, the GPU's L1 constant-memory cache is only invalidated
     * on kernel launch, so a running persistent kernel would NOT see any
     * constant-memory update until it is stopped and restarted.
     *
     * Fix: stop the persistent kernel before the memcpy and restart it
     * immediately after.  Rule updates are management-plane operations
     * (CLI-driven, rare), so the ~100 µs stop/restart overhead is fine.
     */
    int was_persistent = (res && res->persist_active);
    if (was_persistent)
      gpu_classify_stop_persistent (res);

    cudaError_t err;

    if (n_rules > 0)
      {
	err =
	  cudaMemcpyToSymbol (d_rules, rules,
			      n_rules * sizeof (gpu_classify_rule_t), 0,
			      cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	  {
	    if (was_persistent)
	      gpu_classify_start_persistent (res);
	    return -1;
	  }
      }

    int n = static_cast<int> (n_rules);
    err =
      cudaMemcpyToSymbol (d_n_rules, &n, sizeof (int), 0,
			  cudaMemcpyHostToDevice);

    if (was_persistent)
      gpu_classify_start_persistent (res);

    return (err == cudaSuccess) ? 0 : -1;
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
	  res->descs, res->results, static_cast<int> (n_packets));

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
