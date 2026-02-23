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
 *  Shared-memory rule caching — split design:
 *
 *    Although the L1 read-only cache normally hides global-memory latency
 *    via warp switching, the 8 warps in this kernel are perfectly
 *    synchronised — they all access the same rule index at the same
 *    cycle.  When all warps stall together the SM has no other warp to
 *    schedule, so L1 latency (~28-40 cycles per cache line) is exposed.
 *
 *    On-demand kernel — tiled shmem (gpu_classify_tiled):
 *      Rules are loaded 256 at a time from global → shared memory.
 *      A per-tile block-wide vote skips loading remaining tiles once
 *      all threads have matched.  Shmem = 256 × 80 B + 4 B = 20 484 B;
 *      fits within the 48 KB default limit, no cudaFuncSetAttribute.
 *      This path is used only during the first few frames before the
 *      persistent kernel activates.
 *
 *    Persistent kernel — full-table persistent cache (gpu_classify_from_shmem):
 *      The persistent kernel keeps its block resident for the entire VPP
 *      session.  Its shmem (GPU_CLASSIFY_PERSIST_SHMEM_BYTES = 80 KB) is
 *      allocated once at kernel launch and survives across frames.
 *      On startup (and whenever ctrl->rule_version changes), all 256
 *      threads cooperatively load the full rule table into shmem; every
 *      subsequent frame classifies directly from already-hot shmem with
 *      no global-memory traffic for rules at all.
 *      cudaFuncSetAttribute is called once at init to unlock > 48 KB
 *      for the persistent kernel only.
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

/** Number of rules loaded into shared memory per tile.
 *  Equals GPU_CLASSIFY_MAX_FRAME so one tile fills the block's warp
 *  complement in a single round of cooperative loads.                */
#define GPU_CLASSIFY_TILE_RULES  GPU_CLASSIFY_MAX_FRAME

/** Dynamic shared memory for the on-demand kernel (gpu_classify_kernel).
 *
 *  Layout:
 *    [0 .. TILE_RULES × 80)  gpu_classify_rule_t s_rules[TILE_RULES]
 *    [TILE_RULES × 80 .. +4) int any_not_done  (block-wide vote flag)
 *
 *  256 × 80 B + 4 B = 20 484 bytes — well within the 48 KB default.
 *  No cudaFuncSetAttribute opt-in is required.                       */
static const size_t GPU_CLASSIFY_SHMEM_BYTES =
  (size_t) GPU_CLASSIFY_TILE_RULES * sizeof (gpu_classify_rule_t) +
  sizeof (int);

/** Dynamic shared memory for the persistent kernel (gpu_classify_persistent).
 *
 *  The persistent kernel caches the full rule table across frames:
 *    [0 .. MAX_RULES × 80)  gpu_classify_rule_t s_rules[MAX_RULES]
 *
 *  1024 × 80 B = 81 920 bytes — exceeds the 48 KB default limit.
 *  cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)
 *  must be called at init time to opt in (Blackwell supports ≤ 256 KB
 *  per block with the opt-in).                                        */
static const size_t GPU_CLASSIFY_PERSIST_SHMEM_BYTES =
  (size_t) GPU_CLASSIFY_MAX_RULES * sizeof (gpu_classify_rule_t);

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
 * @brief FNV-1a hash over the 39-byte normalised packet key.
 *
 * The key covers: masked src IP (16 B) + masked dst IP (16 B) +
 * src_port (2 B) + dst_port (2 B) + proto (1 B) + ip_version (1 B) +
 * tcp_flags_val (1 B) = 39 bytes.
 *
 * Usable on both host (build_hash_tables) and device (gpu_classify_hash)
 * because all fields are plain arithmetic — no memory-space dependencies.
 *
 * @param msrc      16-byte masked source IP (4-byte aligned).
 * @param mdst      16-byte masked dest IP   (4-byte aligned).
 * @param msp       Masked/zero src_port.
 * @param mdp       Masked/zero dst_port.
 * @param mproto    Masked/zero ip_proto.
 * @param mver      Masked/zero ip_version.
 * @param mfl       pkt.tcp_flags & tcp_flags_mask.
 * @return          32-bit FNV-1a hash value.
 */
__device__ __host__ static uint32_t
gpu_fnv1a_hash (const uint8_t *msrc, const uint8_t *mdst,
		uint16_t msp, uint16_t mdp,
		uint8_t mproto, uint8_t mver, uint8_t mfl)
{
  uint32_t h = 0x811c9dc5u;
  for (int i = 0; i < 16; i++) { h ^= msrc[i]; h *= 0x01000193u; }
  for (int i = 0; i < 16; i++) { h ^= mdst[i]; h *= 0x01000193u; }
  h ^= (uint8_t) (msp >> 8);   h *= 0x01000193u;
  h ^= (uint8_t) (msp & 0xff); h *= 0x01000193u;
  h ^= (uint8_t) (mdp >> 8);   h *= 0x01000193u;
  h ^= (uint8_t) (mdp & 0xff); h *= 0x01000193u;
  h ^= mproto;                  h *= 0x01000193u;
  h ^= mver;                    h *= 0x01000193u;
  h ^= mfl;                     h *= 0x01000193u;
  return h;
}

/**
 * @brief Test whether a hash entry's key matches a normalised probe key.
 *
 * Performs four 32-bit XOR-reduced comparisons for each 16-byte IP array,
 * then five scalar comparisons for the remaining fields.
 *
 * @return  1 on full key match, 0 otherwise.
 */
__device__ __host__ static int
gpu_hash_keys_eq (const gpu_hash_entry_t *e,
		  const uint8_t *msrc, const uint8_t *mdst,
		  uint16_t msp, uint16_t mdp,
		  uint8_t mproto, uint8_t mver, uint8_t mfl)
{
  const uint32_t *es = (const uint32_t *) e->src_ip;
  const uint32_t *ss = (const uint32_t *) msrc;
  const uint32_t *ed = (const uint32_t *) e->dst_ip;
  const uint32_t *sd = (const uint32_t *) mdst;
  if ((es[0] ^ ss[0]) | (es[1] ^ ss[1]) | (es[2] ^ ss[2]) | (es[3] ^ ss[3]))
    return 0;
  if ((ed[0] ^ sd[0]) | (ed[1] ^ sd[1]) | (ed[2] ^ sd[2]) | (ed[3] ^ sd[3]))
    return 0;
  return ((e->src_port     == msp)    &
	  (e->dst_port     == mdp)    &
	  (e->proto        == mproto) &
	  (e->ip_version   == mver)   &
	  (e->tcp_flags_val == mfl));
}

/**
 * @brief Tiled shared-memory classification of one VPP frame.
 *
 * All 256 threads cooperatively load GPU_CLASSIFY_TILE_RULES rules at a
 * time from global memory into shared memory, then each thread checks
 * the tile against its own packet.  Before loading each new tile, all
 * threads vote via a shared-memory atomic (any_not_done): if every
 * active thread has already matched, the block exits early and skips
 * the remaining tiles.
 *
 * Correctness invariants:
 *  - my_matched separates "found a matching rule (any action)" from
 *    "no match yet".  A PASS action counts as matched — we stop at the
 *    first rule that fires, consistent with first-match semantics.
 *  - Padding threads (tid ≥ n_packets) start with my_matched = 1 so
 *    they never dereference d (which is nullptr for them) and always
 *    contribute PASS to the results array.
 *  - Every __syncthreads() in the early-exit vote at the top of each
 *    loop iteration is reached by all 256 threads — no divergence.
 *    The post-load __syncthreads() additionally guards the shmem rule
 *    area from being overwritten while any thread still reads it.
 *
 * @param tid       Linear thread index (threadIdx.x in a 1-block grid).
 * @param n_packets Number of live packets in this batch.
 * @param n_rules   Number of active rules.
 * @param descs     Packet descriptor array (managed memory, CPU-written).
 * @param rules     Rule table (cudaMallocManaged, const __restrict__).
 * @param results   Result byte array (GPU-written, one byte per slot).
 */
__device__ static void
gpu_classify_tiled (int tid, int n_packets, int n_rules,
		    const gpu_pkt_desc_t *descs,
		    const gpu_classify_rule_t *__restrict__ rules,
		    uint8_t *results)
{
  /*
   * Shared memory layout (GPU_CLASSIFY_SHMEM_BYTES bytes):
   *   s_rules     : gpu_classify_rule_t [GPU_CLASSIFY_TILE_RULES]
   *   any_not_done: int (immediately after s_rules, 4-byte aligned)
   */
  extern __shared__ uint8_t shmem[];
  const gpu_classify_rule_t *s_rules =
    (const gpu_classify_rule_t *) shmem;
  int *any_not_done =
    (int *) (shmem +
	     (size_t) GPU_CLASSIFY_TILE_RULES * sizeof (gpu_classify_rule_t));

  /* Pointer to this thread's packet descriptor; nullptr for padding. */
  const gpu_pkt_desc_t *d = (tid < n_packets) ? &descs[tid] : nullptr;

  /* Per-thread match state. Padding threads are "done" from the start
   * so they never dereference d and write PASS to results[].          */
  uint8_t my_action  = GPU_CLASSIFY_ACTION_PASS;
  int     my_matched = (tid >= n_packets);

  for (int tile_base = 0; tile_base < n_rules;
       tile_base += GPU_CLASSIFY_TILE_RULES)
    {
      /* ---- Early-exit vote ---------------------------------------- *
       * Thread 0 resets the flag; all threads vote whether they still  *
       * need to classify.  If no thread votes (every packet already    *
       * matched), the block breaks and skips remaining tiles.          *
       *                                                                *
       * The first __syncthreads() also guards the shmem rule area:     *
       * it prevents the next tile's load from overwriting s_rules      *
       * while any thread is still reading from the previous tile.      */
      if (tid == 0)
	*any_not_done = 0;
      __syncthreads ();
      if (!my_matched)
	atomicOr (any_not_done, 1);
      __syncthreads ();
      if (!*any_not_done)
	break;

      /* ---- Cooperative tile load ----------------------------------- *
       * All 256 threads load uint32_t words in strides of blockDim.x. *
       * This pattern is perfectly coalesced over global memory (128-B  *
       * cache lines, 32 uint32_t per line; 256 threads cover 8 lines   *
       * per iteration) and conflict-free over shared memory banks      *
       * (consecutive threads map to consecutive 32-bit banks).         */
      int tile_n = n_rules - tile_base;
      if (tile_n > GPU_CLASSIFY_TILE_RULES)
	tile_n = GPU_CLASSIFY_TILE_RULES;
      int		 tile_words =
	tile_n * (int) (sizeof (gpu_classify_rule_t) / sizeof (uint32_t));
      const uint32_t *src = (const uint32_t *) (rules + tile_base);
      uint32_t	     *dst = (uint32_t *) shmem;
      for (int w = tid; w < tile_words; w += blockDim.x)
	dst[w] = src[w];
      __syncthreads (); /* tile fully in shmem before any thread classifies */

      /* ---- Per-thread classification against the shmem tile -------- */
      if (!my_matched)
	{
	  for (int i = 0; i < tile_n; i++)
	    {
	      const gpu_classify_rule_t *r = &s_rules[i];

	      /* ---- Protocol ---------------------------------------- */
	      if (r->proto != 0 && r->proto != d->ip_proto)
		continue;

	      /* ---- IP version -------------------------------------- */
	      if (r->ip_version != 0 && r->ip_version != d->ip_version)
		continue;

	      /* ---- Source IP prefix -------------------------------- */
	      if (!ip_matches (d->src_ip, r->src_addr, r->src_mask))
		continue;

	      /* ---- Destination IP prefix -------------------------- */
	      if (!ip_matches (d->dst_ip, r->dst_addr, r->dst_mask))
		continue;

	      /* ---- Source port ------------------------------------- */
	      if (r->src_port != 0 && r->src_port != d->src_port)
		continue;

	      /* ---- Destination port -------------------------------- */
	      if (r->dst_port != 0 && r->dst_port != d->dst_port)
		continue;

	      /* ---- TCP flags --------------------------------------- */
	      if (r->tcp_flags_mask != 0 &&
		  (d->tcp_flags & r->tcp_flags_mask) != r->tcp_flags_val)
		continue;

	      /* First matching rule: record action and stop.          */
	      my_action  = r->action;
	      my_matched = 1;
	      break;
	    }
	}
      /*
       * Note: no __syncthreads() here.  The next iteration begins with
       * the early-exit vote's first __syncthreads(), which doubles as
       * the barrier that prevents the following tile load from starting
       * until all threads have finished reading the current s_rules[].
       */
    }

  /* Write result.  Padding threads write GPU_CLASSIFY_ACTION_PASS (0). */
  if (tid < GPU_CLASSIFY_MAX_FRAME)
    results[tid] = my_action;
}

/**
 * @brief Classify one VPP frame against rules already resident in shmem.
 *
 * Used by the persistent kernel after the full rule table has been
 * loaded into shared memory.  No tile loads, no block-wide votes —
 * each thread independently scans s_rules[0..n_rules) and breaks on
 * the first match.
 *
 * No __syncthreads() is needed here: shmem was loaded and synchronised
 * in the caller's Phase 0 (rule reload), and each thread writes only
 * its own results[tid] slot.
 *
 * @param tid     Linear thread index.
 * @param n       Number of live packets in this batch.
 * @param n_rules Number of active rules in s_rules[].
 * @param descs   Packet descriptor array (managed memory).
 * @param s_rules Rule table already in shared memory.
 * @param results Result byte array (one byte per slot).
 */
__device__ static void
gpu_classify_from_shmem (int tid, int n, int n_rules,
			 const gpu_pkt_desc_t *descs,
			 const gpu_classify_rule_t *s_rules,
			 uint8_t *results)
{
  uint8_t action = GPU_CLASSIFY_ACTION_PASS;

  if (tid < n)
    {
      const gpu_pkt_desc_t *d = &descs[tid];

      for (int i = 0; i < n_rules; i++)
	{
	  const gpu_classify_rule_t *r = &s_rules[i];

	  if (r->proto != 0 && r->proto != d->ip_proto)
	    continue;
	  if (r->ip_version != 0 && r->ip_version != d->ip_version)
	    continue;
	  if (!ip_matches (d->src_ip, r->src_addr, r->src_mask))
	    continue;
	  if (!ip_matches (d->dst_ip, r->dst_addr, r->dst_mask))
	    continue;
	  if (r->src_port != 0 && r->src_port != d->src_port)
	    continue;
	  if (r->dst_port != 0 && r->dst_port != d->dst_port)
	    continue;
	  if (r->tcp_flags_mask != 0 &&
	      (d->tcp_flags & r->tcp_flags_mask) != r->tcp_flags_val)
	    continue;

	  action = r->action;
	  break;
	}
    }

  results[tid] = action;
}

/**
 * @brief O(K) hash-table classification of one VPP frame.
 *
 * Iterates over K sorted table descriptors (lowest min_rule_idx first).
 * For each table, the thread computes a masked probe key, hashes it with
 * FNV-1a, and performs linear probing in the flat hash_entries[] array.
 *
 * First-match semantics are preserved via best_idx: once a match is found
 * at rule_idx R, any table whose min_rule_idx >= R cannot produce a
 * higher-priority match and the loop terminates early.
 *
 * No __syncthreads() is needed: shmem descriptors were loaded and
 * synchronised in the caller (persistent kernel Phase 0), and each
 * thread writes only its own results[tid] slot.
 *
 * @param tid           Linear thread index.
 * @param n_pkts        Number of live packets in this batch.
 * @param n_tables      Number of hash tables (K).
 * @param descs         Packet descriptor array (managed memory).
 * @param tdescs        Table descriptor array (shmem or global memory).
 * @param entries       Flat hash-entry array (managed memory, __restrict__).
 * @param results       Result byte array (one byte per slot).
 */
__device__ static void
gpu_classify_hash (int tid, int n_pkts, int n_tables,
		   const gpu_pkt_desc_t *descs,
		   const gpu_hash_table_desc_t *tdescs,
		   const gpu_hash_entry_t *__restrict__ entries,
		   uint8_t *results)
{
  uint8_t best_action = GPU_CLASSIFY_ACTION_PASS;
  int32_t best_idx    = 0x7fffffff;

  if (tid < n_pkts)
    {
      const gpu_pkt_desc_t *d = &descs[tid];

      for (int t = 0; t < n_tables; t++)
	{
	  const gpu_hash_table_desc_t *td = &tdescs[t];

	  /* Early exit: remaining tables have equal or higher min_rule_idx. */
	  if (td->min_rule_idx >= best_idx)
	    break;

	  /* Apply address masks (4 × 32-bit words = 16 bytes each). */
	  const uint32_t *src = (const uint32_t *) d->src_ip;
	  const uint32_t *sm  = (const uint32_t *) td->src_mask;
	  const uint32_t *dst = (const uint32_t *) d->dst_ip;
	  const uint32_t *dm  = (const uint32_t *) td->dst_mask;
	  uint32_t msr[4], mdr[4];
	  msr[0] = src[0] & sm[0]; msr[1] = src[1] & sm[1];
	  msr[2] = src[2] & sm[2]; msr[3] = src[3] & sm[3];
	  mdr[0] = dst[0] & dm[0]; mdr[1] = dst[1] & dm[1];
	  mdr[2] = dst[2] & dm[2]; mdr[3] = dst[3] & dm[3];

	  /* Compute normalised scalar key fields. */
	  uint16_t msp    = td->match_src_port    ? d->src_port   : 0;
	  uint16_t mdp    = td->match_dst_port    ? d->dst_port   : 0;
	  uint8_t  mproto = td->match_proto       ? d->ip_proto   : 0;
	  uint8_t  mver   = td->match_ip_version  ? d->ip_version : 0;
	  uint8_t  mfl    = d->tcp_flags & td->tcp_flags_mask;

	  /* Hash the full 39-byte key and compute the initial slot. */
	  uint32_t h    = gpu_fnv1a_hash ((const uint8_t *) msr,
					  (const uint8_t *) mdr,
					  msp, mdp, mproto, mver, mfl);
	  uint32_t slot = h & (td->n_slots - 1);
	  uint32_t base = td->entry_base;

	  /* Linear probe until empty sentinel or key match. */
	  for (uint32_t probe = 0; probe < td->n_slots; probe++)
	    {
	      const gpu_hash_entry_t *e = &entries[base + slot];
	      if (!e->valid)
		break; /* empty slot → no match in this table */
	      if (gpu_hash_keys_eq (e, (const uint8_t *) msr,
				    (const uint8_t *) mdr,
				    msp, mdp, mproto, mver, mfl))
		{
		  if (e->rule_idx < best_idx)
		    {
		      best_idx    = e->rule_idx;
		      best_action = e->action;
		    }
		  break; /* key is unique per table */
		}
	      slot = (slot + 1) & (td->n_slots - 1);
	    }
	}
    }

  /* Write result; padding threads write PASS. */
  if (tid < GPU_CLASSIFY_MAX_FRAME)
    results[tid] = best_action;
}

/* ================================================================== */
/* Kernels                                                             */
/* ================================================================== */

/**
 * @brief Classify one VPP frame of packets in parallel (on-demand).
 *
 * One block of GPU_CLASSIFY_MAX_FRAME threads; one thread per packet
 * slot.  Dispatches to the O(K) hash path when n_hash_tables > 0,
 * otherwise falls back to the tiled linear-scan path.
 *
 * @param descs         Array of GPU_CLASSIFY_MAX_FRAME packet descriptors.
 * @param results       Output array (one GPU_CLASSIFY_ACTION_* byte per slot).
 * @param n_packets     Number of live packets (≤ MAX_FRAME).
 * @param rules         Rule table (cudaMallocManaged, __restrict__).
 * @param n_rules       Number of active rules.
 * @param n_hash_tables Number of hash tables; 0 → linear scan fallback.
 * @param hash_descs    Table descriptors (cudaMallocManaged, __restrict__).
 * @param hash_entries  Flat hash-entry array (cudaMallocManaged, __restrict__).
 */
__global__ void
gpu_classify_kernel (const gpu_pkt_desc_t *__restrict__ descs,
		     uint8_t *__restrict__ results, int n_packets,
		     const gpu_classify_rule_t *__restrict__ rules, int n_rules,
		     int n_hash_tables,
		     const gpu_hash_table_desc_t *__restrict__ hash_descs,
		     const gpu_hash_entry_t *__restrict__ hash_entries)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_hash_tables > 0)
    gpu_classify_hash (tid, n_packets, n_hash_tables, descs,
		       hash_descs, hash_entries, results);
  else
    gpu_classify_tiled (tid, n_packets, n_rules, descs, rules, results);
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
 * @param ctrl         CPU↔GPU handshake control block (managed memory).
 * @param descs        Packet descriptor array (managed memory, CPU-written).
 * @param results      Result array (managed memory, GPU-written).
 * @param rules        Rule table (cudaMallocManaged, const __restrict__).
 * @param hash_descs   Table descriptors (cudaMallocManaged, __restrict__).
 * @param hash_entries Flat hash-entry array (cudaMallocManaged, __restrict__).
 */
__global__ void
gpu_classify_persistent (gpu_classify_ctrl_t *ctrl,
			 const gpu_pkt_desc_t *descs, uint8_t *results,
			 const gpu_classify_rule_t *__restrict__ rules,
			 const gpu_hash_table_desc_t *__restrict__ hash_descs,
			 const gpu_hash_entry_t *__restrict__ hash_entries)
{
  int tid = threadIdx.x; /* blockIdx.x == 0 always (single-block launch) */

  /*
   * Shared memory is dual-use — same allocation, different content:
   *
   *   Hash path (last_n_hash_tables > 0):
   *     s_descs: gpu_hash_table_desc_t [MAX_TABLES]  (K×64 B ≤ 4 KB)
   *
   *   Linear path (last_n_hash_tables == 0):
   *     s_rules: gpu_classify_rule_t   [MAX_RULES]   (up to 80 KB)
   *
   * GPU_CLASSIFY_PERSIST_SHMEM_BYTES = 80 KB covers both paths.
   * cudaFuncSetAttribute has already unlocked > 48 KB at init time.
   *
   * last_rule_version = ~0u forces the initial load on the very first
   * batch regardless of ctrl->rule_version at start.
   * last_n_hash_tables = -1 is re-set in Phase 0 before Phase 3 reads it.
   */
  extern __shared__ uint8_t shmem[];
  const gpu_classify_rule_t      *s_rules  = (const gpu_classify_rule_t *)  shmem;
  /* s_descs overlaps s_rules — valid only when last_n_hash_tables > 0 */

  /*
   * System-scope atomic reference type for the handshake fields.
   *
   * cuda::thread_scope_system guarantees visibility across GPU and CPU
   * (required for NVLink-C2C / managed memory).  Plain volatile reads
   * only bypass GPU L1; they do NOT guarantee CPU writes are visible to
   * the GPU (ld.volatile is a weakly-ordered load per PTX ISA).
   *
   * memory_order_acquire on submit_seq load: once we see a new seq,
   * all CPU writes before the CPU's release-store (n_packets, kill,
   * n_rules, rule_version, n_hash_tables) are guaranteed visible.
   *
   * memory_order_release on done_seq store: pairs with the CPU's
   * acquire-load.  The CPU is guaranteed to see all results[] writes
   * once it observes the new done_seq value.
   */
  using sys_u32 = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;

  uint32_t last_seq          = 0;    /* last batch seq processed           */
  uint32_t last_rule_version = ~0u;  /* ~0 forces load on first batch      */
  int32_t  last_n_hash_tables = -1;  /* -1: re-set unconditionally in Ph.0 */

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
       * guaranteeing that n_packets / kill / n_rules / rule_version
       * written before the CPU's store are all visible to GPU threads.
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

      /* Snapshot batch parameters (all visible after acquire of seq).   */
      int      n       = (int) ctrl->n_packets;
      int      n_rules = (int) ctrl->n_rules;
      int32_t  nh      = ctrl->n_hash_tables;
      uint32_t rv      = (uint32_t) ctrl->rule_version;

      /* ---- Phase 0: conditional shmem reload ------------------------- *
       * The CPU increments ctrl->rule_version (RELEASE) whenever it     *
       * updates the rule table.  All threads cooperatively reload shmem *
       * on version change, then __syncthreads() ensures every thread    *
       * sees the new content before classification begins.              *
       *                                                                  *
       *   Hash path  (nh > 0): load K×64 B descriptors into shmem.     *
       *   Linear path (nh == 0): load full N×80 B rule table.          *
       *                                                                  *
       * On frames where rules have not changed, shmem is already valid  *
       * and the load is skipped entirely — zero global-memory rule I/O. */
      if (rv != last_rule_version)
	{
	  last_rule_version   = rv;
	  last_n_hash_tables  = nh;
	  if (nh > 0)
	    {
	      /* Cooperative load of K table descriptors (K×64 B). */
	      int total_dw = nh * (int) (sizeof (gpu_hash_table_desc_t) /
					 sizeof (uint32_t));
	      const uint32_t *srcd = (const uint32_t *) hash_descs;
	      uint32_t       *dstd = (uint32_t *) shmem;
	      for (int w = tid; w < total_dw; w += blockDim.x)
		dstd[w] = srcd[w];
	    }
	  else
	    {
	      /* Cooperative load of full rule table (N×80 B). */
	      int	       total = n_rules *
				       (int) (sizeof (gpu_classify_rule_t) /
					      sizeof (uint32_t));
	      const uint32_t *src = (const uint32_t *) rules;
	      uint32_t	     *dst = (uint32_t *) shmem;
	      for (int w = tid; w < total; w += blockDim.x)
		dst[w] = src[w];
	    }
	  __syncthreads (); /* shmem fully loaded before any thread classifies */
	}

      /* ---- Phase 3: classify from shmem — no global rule traffic ----- */
      if (last_n_hash_tables > 0)
	{
	  const gpu_hash_table_desc_t *s_descs =
	    (const gpu_hash_table_desc_t *) shmem;
	  gpu_classify_hash (tid, n, (int) last_n_hash_tables,
			     descs, s_descs, hash_entries, results);
	}
      else
	{
	  gpu_classify_from_shmem (tid, n, n_rules, descs, s_rules, results);
	}

      /* ---- Phase 4: signal completion -------------------------------- *
       * __threadfence_system(): flush result writes from GPU L2 to be   *
       * visible system-wide (CPU).                                      *
       * __syncthreads(): wait for all threads to flush before thread 0  *
       * signals done_seq.                                               *
       * done_seq release-store pairs with the CPU's acquire-load.       */
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
 * @brief Build GPU hash tables from the current rule set.
 *
 * This function runs entirely on the CPU.  It partitions rules into groups
 * sharing the same (src_mask, dst_mask, port/proto/version/flags) combo,
 * builds an open-addressing hash table per group at 50 % load factor,
 * sorts the resulting descriptors by min_rule_idx (ascending), and copies
 * everything to the cudaMallocManaged buffers in @a res.
 *
 * If the number of distinct mask combos exceeds GPU_CLASSIFY_MAX_TABLES,
 * sets res->n_hash_tables = 0 to trigger the linear-scan fallback.
 *
 * Caller must hold the rules_lock.
 */
static void
gpu_classify_build_hash_tables (gpu_classify_cuda_res_t *res,
				const gpu_classify_rule_t *rules,
				int n_rules)
{
  if (n_rules == 0)
    {
      res->n_hash_tables = 0;
      return;
    }

  /* ---- Local working arrays (stack; MAX_TABLES = 64, MAX_RULES = 1024) */
  gpu_hash_table_desc_t descs[GPU_CLASSIFY_MAX_TABLES];
  int rule_table[GPU_CLASSIFY_MAX_RULES]; /* maps rule i → table index */
  int n_tables = 0;

  /* ---- Pass 1: find distinct mask combos ----------------------------- */
  for (int i = 0; i < n_rules; i++)
    {
      const gpu_classify_rule_t *r = &rules[i];

      uint8_t match_src_port   = (r->src_port != 0);
      uint8_t match_dst_port   = (r->dst_port != 0);
      uint8_t match_proto      = (r->proto != 0);
      uint8_t match_ip_version = (r->ip_version != 0);
      uint8_t tcp_flags_mask   = r->tcp_flags_mask;

      /* Search for an existing table with the same combo. */
      int t;
      for (t = 0; t < n_tables; t++)
	{
	  gpu_hash_table_desc_t *td = &descs[t];
	  if (memcmp (td->src_mask, r->src_mask, 16) == 0 &&
	      memcmp (td->dst_mask, r->dst_mask, 16) == 0 &&
	      td->match_src_port   == match_src_port   &&
	      td->match_dst_port   == match_dst_port   &&
	      td->match_proto      == match_proto       &&
	      td->match_ip_version == match_ip_version  &&
	      td->tcp_flags_mask   == tcp_flags_mask)
	    break;
	}

      if (t == n_tables)
	{
	  /* New combo — check capacity first. */
	  if (n_tables >= GPU_CLASSIFY_MAX_TABLES)
	    {
	      res->n_hash_tables = 0; /* too many distinct mask combos */
	      return;
	    }
	  memset (&descs[t], 0, sizeof (descs[t]));
	  memcpy (descs[t].src_mask, r->src_mask, 16);
	  memcpy (descs[t].dst_mask, r->dst_mask, 16);
	  descs[t].match_src_port   = match_src_port;
	  descs[t].match_dst_port   = match_dst_port;
	  descs[t].match_proto      = match_proto;
	  descs[t].match_ip_version = match_ip_version;
	  descs[t].tcp_flags_mask   = tcp_flags_mask;
	  descs[t].n_slots          = 0; /* reused as rule count in Pass 1 */
	  descs[t].min_rule_idx     = i;
	  n_tables++;
	}
      else
	{
	  if (i < descs[t].min_rule_idx)
	    descs[t].min_rule_idx = i;
	}

      rule_table[i] = t;
      descs[t].n_slots++; /* count rules per table */
    }

  /* ---- Pass 2: compute n_slots (next power-of-2 ≥ 2×count) and
   *              entry_base (prefix-sum of slot counts).             */
  uint32_t total_slots = 0;
  for (int t = 0; t < n_tables; t++)
    {
      uint32_t cnt = descs[t].n_slots;
      uint32_t min_s = cnt * 2;
      if (min_s < 2)
	min_s = 2;
      uint32_t s = 1;
      while (s < min_s)
	s <<= 1;
      descs[t].n_slots    = s;
      descs[t].entry_base = total_slots;
      total_slots += s;
    }

  /* Guard: should never fire with MAX_RULES=1024 / capacity=4096. */
  if (total_slots > res->hash_entry_capacity)
    {
      res->n_hash_tables = 0;
      return;
    }

  /* ---- Pass 3: zero the entry region -------------------------------- */
  memset (res->hash_entries, 0, total_slots * sizeof (gpu_hash_entry_t));

  /* ---- Pass 4: insert rules into hash tables ------------------------ */
  for (int i = 0; i < n_rules; i++)
    {
      const gpu_classify_rule_t *r    = &rules[i];
      int			 t    = rule_table[i];
      const gpu_hash_table_desc_t *td = &descs[t];

      /* Compute the normalised masked key for this rule. */
      const uint32_t *sa = (const uint32_t *) r->src_addr;
      const uint32_t *sm = (const uint32_t *) td->src_mask;
      const uint32_t *da = (const uint32_t *) r->dst_addr;
      const uint32_t *dm = (const uint32_t *) td->dst_mask;
      uint32_t msr[4], mdr[4];
      msr[0] = sa[0] & sm[0]; msr[1] = sa[1] & sm[1];
      msr[2] = sa[2] & sm[2]; msr[3] = sa[3] & sm[3];
      mdr[0] = da[0] & dm[0]; mdr[1] = da[1] & dm[1];
      mdr[2] = da[2] & dm[2]; mdr[3] = da[3] & dm[3];

      uint16_t msp    = td->match_src_port    ? r->src_port    : 0;
      uint16_t mdp    = td->match_dst_port    ? r->dst_port    : 0;
      uint8_t  mproto = td->match_proto       ? r->proto       : 0;
      uint8_t  mver   = td->match_ip_version  ? r->ip_version  : 0;
      uint8_t  mfl    = td->tcp_flags_mask    ? r->tcp_flags_val : 0;

      uint32_t h    = gpu_fnv1a_hash ((const uint8_t *) msr,
				      (const uint8_t *) mdr,
				      msp, mdp, mproto, mver, mfl);
      uint32_t slot = h & (td->n_slots - 1);
      uint32_t base = td->entry_base;

      /* Linear probe to find empty slot or existing duplicate key. */
      for (uint32_t probe = 0; probe < td->n_slots; probe++)
	{
	  gpu_hash_entry_t *e = &res->hash_entries[base + slot];
	  if (!e->valid)
	    {
	      /* Empty slot: insert new entry. */
	      memcpy (e->src_ip, msr, 16);
	      memcpy (e->dst_ip, mdr, 16);
	      e->src_port      = msp;
	      e->dst_port      = mdp;
	      e->proto         = mproto;
	      e->ip_version    = mver;
	      e->tcp_flags_val = mfl;
	      e->action        = r->action;
	      e->valid         = 1;
	      e->rule_idx      = i;
	      break;
	    }
	  if (gpu_hash_keys_eq (e, (const uint8_t *) msr,
				(const uint8_t *) mdr,
				msp, mdp, mproto, mver, mfl))
	    {
	      /* Duplicate masked key: keep lower rule_idx (higher priority). */
	      if (i < e->rule_idx)
		{
		  e->rule_idx = i;
		  e->action   = r->action;
		}
	      break;
	    }
	  slot = (slot + 1) & (td->n_slots - 1);
	}
    }

  /* ---- Pass 5: insertion-sort descs[] by min_rule_idx ascending ----- *
   * ≤ 64 elements; insertion sort is O(K²) = negligible.             */
  for (int i = 1; i < n_tables; i++)
    {
      gpu_hash_table_desc_t tmp = descs[i];
      int j = i - 1;
      while (j >= 0 && descs[j].min_rule_idx > tmp.min_rule_idx)
	{
	  descs[j + 1] = descs[j];
	  j--;
	}
      descs[j + 1] = tmp;
    }

  /* ---- Copy sorted descriptors to managed memory -------------------- */
  memcpy (res->hash_descs, descs, (size_t) n_tables * sizeof (descs[0]));
  res->n_hash_tables = n_tables;
}

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
  ctrl->kill	      = 0;
  ctrl->n_packets     = 0;
  ctrl->n_rules       = (int32_t) res->n_rules;
  ctrl->n_hash_tables = (int32_t) res->n_hash_tables;

  /* Full fence: all resets must be globally visible before the GPU
   * kernel executes its first polling iteration.                     */
  __atomic_thread_fence (__ATOMIC_SEQ_CST);

  /* GPU_CLASSIFY_PERSIST_SHMEM_BYTES = 80 KB (full rule table).
   * cudaFuncSetAttribute has already opted this kernel in at init time.
   * Shmem is reused across frames; reloaded only when rule_version changes. */
  gpu_classify_persistent<<<dim3 (1), dim3 (GPU_CLASSIFY_MAX_FRAME),
			     GPU_CLASSIFY_PERSIST_SHMEM_BYTES,
			     stream>>> (ctrl, res->descs, res->results,
					res->rules,
					res->hash_descs, res->hash_entries);

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

    /* Unlock > 48 KB of dynamic shared memory for the persistent kernel.
     *
     * GPU_CLASSIFY_PERSIST_SHMEM_BYTES = 80 KB (full rule table).
     * Blackwell supports up to 256 KB per block with this opt-in.
     * The on-demand kernel uses only 20 KB (tiled) and needs no opt-in.
     *
     * Failure means the persistent kernel cannot cache the full rule
     * table; treat as fatal since it would fall back to an oversized
     * shmem request at launch time.                                      */
    err = cudaFuncSetAttribute (
      (const void *) gpu_classify_persistent,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int) GPU_CLASSIFY_PERSIST_SHMEM_BYTES);
    if (err != cudaSuccess)
      {
	fprintf (stderr,
		 "gpu_classify: cudaFuncSetAttribute (persistent): %s\n",
		 cudaGetErrorString (err));
	goto fail_rules;
      }

    /* Allocate hash-table entry array (flat, 4096 × 48 B = 192 KB).
     * CPU writes at rule-update time; GPU reads per-frame via L1 cache. */
    err = cudaMallocManaged (reinterpret_cast<void **> (&res->hash_entries),
			     4096u * sizeof (gpu_hash_entry_t));
    if (err != cudaSuccess)
      {
	fprintf (stderr,
		 "gpu_classify: cudaMallocManaged (hash_entries): %s\n",
		 cudaGetErrorString (err));
	goto fail_rules; /* rules already allocated; free it in chain */
      }
    memset (res->hash_entries, 0, 4096u * sizeof (gpu_hash_entry_t));
    res->hash_entry_capacity = 4096u;

    /* Allocate hash-table descriptor array (MAX_TABLES × 64 B = 4 KB). */
    err = cudaMallocManaged (reinterpret_cast<void **> (&res->hash_descs),
			     GPU_CLASSIFY_MAX_TABLES * sizeof (gpu_hash_table_desc_t));
    if (err != cudaSuccess)
      {
	fprintf (stderr,
		 "gpu_classify: cudaMallocManaged (hash_descs): %s\n",
		 cudaGetErrorString (err));
	goto fail_hash_entries;
      }
    memset (res->hash_descs, 0,
	    GPU_CLASSIFY_MAX_TABLES * sizeof (gpu_hash_table_desc_t));
    res->n_hash_tables = 0;

    /* Hint preferred locations:
     *   descs        → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   rules        → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   hash_entries → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   hash_descs   → CPU writes, GPU reads  → prefer CPU-side DRAM.
     *   results      → GPU writes, CPU reads  → prefer GPU-side DRAM.
     *   ctrl         → mixed read/write       → no preference hint.   */
    cudaMemAdvise (res->descs,
		   GPU_CLASSIFY_MAX_FRAME * sizeof (gpu_pkt_desc_t),
		   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise (res->rules,
		   GPU_CLASSIFY_MAX_RULES * sizeof (gpu_classify_rule_t),
		   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise (res->hash_entries,
		   4096u * sizeof (gpu_hash_entry_t),
		   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise (res->hash_descs,
		   GPU_CLASSIFY_MAX_TABLES * sizeof (gpu_hash_table_desc_t),
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

  fail_hash_entries:
    cudaFree (res->hash_entries);
    res->hash_entries = nullptr;
  fail_rules:
    cudaFree (res->rules);
    res->rules = nullptr;
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

    if (res->hash_descs)
      {
	cudaFree (res->hash_descs);
	res->hash_descs = nullptr;
      }
    if (res->hash_entries)
      {
	cudaFree (res->hash_entries);
	res->hash_entries = nullptr;
      }
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

    /* Build hash tables from the new rule set.  Populates res->hash_entries,
     * res->hash_descs, and res->n_hash_tables.  Falls back to n_hash_tables=0
     * (linear scan) if there are too many distinct mask combos.           */
    gpu_classify_build_hash_tables (res, rules, (int) n_rules);

    /* If the persistent kernel is live, propagate the new counts and bump
     * rule_version so the kernel reloads its shmem cache on the next batch.
     *
     * Ordering: data (rules/hash_entries/hash_descs) and counts (n_rules,
     * n_hash_tables) must all be visible before rule_version so the kernel
     * always classifies with a fully consistent state.  Stores use RELEASE
     * so the preceding memcpy / build_hash_tables writes are also ordered. */
    if (res->persist_active)
      {
	__atomic_store_n (&res->ctrl->n_rules,
			  (int32_t) n_rules, __ATOMIC_RELEASE);
	__atomic_store_n (&res->ctrl->n_hash_tables,
			  (int32_t) res->n_hash_tables, __ATOMIC_RELEASE);
	__atomic_fetch_add (&res->ctrl->rule_version, 1u, __ATOMIC_RELEASE);
      }

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
	 *
	 * Dynamic shmem = GPU_CLASSIFY_SHMEM_BYTES (one tile + vote flag
	 * = 20 484 bytes); always allocated even when n_rules == 0
	 * (the tile loop is a no-op in that case).
	 */
	cudaStream_t stream = reinterpret_cast<cudaStream_t> (res->stream);
	dim3	     block (GPU_CLASSIFY_MAX_FRAME);
	dim3	     grid (1);

	gpu_classify_kernel<<<grid, block, GPU_CLASSIFY_SHMEM_BYTES,
			      stream>>> (
	  res->descs, res->results, static_cast<int> (n_packets),
	  res->rules, res->n_rules,
	  res->n_hash_tables, res->hash_descs, res->hash_entries);

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
