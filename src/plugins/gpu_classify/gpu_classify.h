/* SPDX-License-Identifier: Apache-2.0
 * GPU Packet Classifier plugin — VPP-side header.
 *
 * Includes VPP headers and the shared type definitions.  Not included
 * from gpu_classify_kernel.cu (CUDA device-code path uses only
 * gpu_classify_types.h).
 */

#ifndef __included_gpu_classify_h__
#define __included_gpu_classify_h__

#include <vnet/vnet.h>
#include <vnet/ip/ip.h>
#include <vnet/ethernet/ethernet.h>
#include <vnet/feature/feature.h>
#include <vlib/vlib.h>
#include <vppinfra/error.h>
#include <vppinfra/vec.h>
#include <vppinfra/lock.h>

#include <gpu_classify/gpu_classify_types.h>

/* ------------------------------------------------------------------ */
/* Buffer flags                                                        */
/* ------------------------------------------------------------------ */

/** vlib_buffer flag set on packets whose GPU action is MARK.
 *  Downstream nodes may inspect this flag. */
#define GPU_CLASSIFY_BUFFER_FLAG_MARKED VLIB_BUFFER_FLAG_USER (1)

/* ------------------------------------------------------------------ */
/* Per-interface state                                                 */
/* ------------------------------------------------------------------ */

typedef struct
{
  u8 ip4_enabled; /**< Non-zero when the ip4-unicast feature is active */
  u8 ip6_enabled; /**< Non-zero when the ip6-unicast feature is active */
} gpu_classify_if_state_t;

/* ------------------------------------------------------------------ */
/* Plugin main structure                                               */
/* ------------------------------------------------------------------ */

typedef struct
{
  /* API message-ID base (reserved for future API extension) */
  u16 msg_id_base;

  /* Classification rules — CPU master copy.
   * Written by CLI/API handlers, pushed to GPU constant memory via
   * gpu_classify_update_rules() whenever they change.           */
  gpu_classify_rule_t rules[GPU_CLASSIFY_MAX_RULES];
  u32		      n_rules;

  /* Spinlock protecting rules and n_rules when workers are active */
  clib_spinlock_t rules_lock;

  /* Per-interface enable flags (vec, indexed by sw_if_index) */
  gpu_classify_if_state_t *if_state;

  /* CUDA resource bundle (stream + managed memory buffers) */
  gpu_classify_cuda_res_t cuda_res;

  /** Set to 1 iff gpu_classify_cuda_init() succeeded.
   *  When 0 the node passes all packets through without GPU inspection
   *  so VPP keeps running on hardware without a CUDA-capable GPU.    */
  u8 cuda_ready;

  /* Log class — registered once at init, reused by all log calls */
  vlib_log_class_t log_class;

  /* Simple packet counters (updated only on the main thread for now;
   * accurate enough for CLI display, not relied on for correctness). */
  u64 n_pass;
  u64 n_drop;
  u64 n_mark;

  vlib_main_t *vlib_main;
  vnet_main_t *vnet_main;
} gpu_classify_main_t;

extern gpu_classify_main_t gpu_classify_main;

/* ------------------------------------------------------------------ */
/* CUDA interface (implemented in gpu_classify_kernel.cu)             */
/* Declared extern "C" so VPP's plain-C compilation units can call.  */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief Allocate CUDA resources (stream + managed memory buffers).
   *        Call once at plugin init time.
   * @return 0 on success, -1 on CUDA error.
   */
  int gpu_classify_cuda_init (gpu_classify_cuda_res_t *res);

  /**
   * @brief Release all CUDA resources.  Safe to call with zeroed res.
   */
  void gpu_classify_cuda_cleanup (gpu_classify_cuda_res_t *res);

  /**
   * @brief Copy @a n_rules rules into GPU constant memory.
   * @return 0 on success, -1 on error (e.g. n_rules > MAX_RULES).
   */
  int gpu_classify_update_rules (gpu_classify_cuda_res_t *res,
				 gpu_classify_rule_t *rules, u32 n_rules);

  /**
   * @brief Launch the classification kernel on @a n_packets packets,
   *        then synchronise the stream before returning.
   *
   * Precondition:  res->descs[0..n_packets-1] have been filled by the
   *                CPU node function.
   * Postcondition: res->results[0..n_packets-1] contain GPU_CLASSIFY_ACTION_*.
   *
   * @return 0 on success, -1 on CUDA error.
   */
  int gpu_classify_launch_kernel (gpu_classify_cuda_res_t *res,
				  u32 n_packets);

  /**
   * @brief Query static properties of the active CUDA device.
   * @return 0 on success, -1 if no CUDA device is available.
   */
  int gpu_classify_get_device_info (gpu_classify_device_info_t *info);

#ifdef __cplusplus
}
#endif

#endif /* __included_gpu_classify_h__ */
