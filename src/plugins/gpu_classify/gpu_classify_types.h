/* SPDX-License-Identifier: Apache-2.0
 * GPU Packet Classifier — shared types (C and CUDA compatible).
 *
 * This header uses only <stdint.h> so it can be compiled by both
 * the host C compiler (gcc/clang) and nvcc without pulling in VPP
 * headers on the device-code path.
 *
 * Target hardware: NVIDIA GB10 Blackwell (DGX Spark, NVLink-C2C).
 */

#ifndef __included_gpu_classify_types_h__
#define __included_gpu_classify_types_h__

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/** Maximum number of classification rules held in GPU constant memory.
 *  64 rules × 24 bytes = 1 536 bytes (well within the 64 KB limit).  */
#define GPU_CLASSIFY_MAX_RULES  64

/** Maximum packets per VPP frame (== VLIB_FRAME_SIZE).               */
#define GPU_CLASSIFY_MAX_FRAME  256

/* Packet actions written to the result buffer by the GPU kernel. */
#define GPU_CLASSIFY_ACTION_PASS  0   /**< forward to next feature     */
#define GPU_CLASSIFY_ACTION_DROP  1   /**< send to error-drop          */
#define GPU_CLASSIFY_ACTION_MARK  2   /**< set flag and forward        */

/* ------------------------------------------------------------------ */
/* Packet descriptor — 32 bytes, cache-line–friendly                  */
/* ------------------------------------------------------------------ */

/**
 * @brief Compact per-packet descriptor filled by the CPU and read by
 *        the GPU kernel.
 *
 * 32 bytes total: one naturally-aligned 32-byte block per thread.
 * All multi-byte fields are in **network byte order**, consistent with
 * the raw packet bytes the CPU reads out of vlib_buffer_t.
 */
typedef struct
{
  uint32_t src_ip4;     /**< Source IPv4 address (network byte order)      */
  uint32_t dst_ip4;     /**< Destination IPv4 address (network byte order) */
  uint16_t src_port;    /**< Source L4 port (network byte order)            */
  uint16_t dst_port;    /**< Destination L4 port (network byte order)       */
  uint8_t  ip_proto;    /**< IP protocol number (TCP=6, UDP=17, …)         */
  uint8_t  tcp_flags;   /**< TCP flags byte; 0 for non-TCP packets          */
  uint8_t  ip_version;  /**< IP version: 4 (IPv6 reserved for future)       */
  uint8_t  valid;       /**< 1 = slot is populated; 0 = padding             */
  uint8_t  payload[16]; /**< First 16 bytes after the L4 header             */
  /*                         ───────────────────────────────────────── 32 B */
} gpu_pkt_desc_t;

/* ------------------------------------------------------------------ */
/* Classification rule — 24 bytes                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief One classification rule stored in GPU constant memory.
 *
 * Field semantics:
 *  - A zero value in src_mask/dst_mask means "any source/dest IP".
 *  - A zero value in src_port/dst_port means "any port".
 *  - A zero value in proto means "any protocol".
 *  - tcp_flags_mask == 0 means "don't check TCP flags".
 * All address/port fields are in **network byte order**.
 */
typedef struct
{
  uint32_t src_addr;       /**< Source IP prefix (net byte order)           */
  uint32_t src_mask;       /**< Source prefix mask  (0 = wildcard)          */
  uint32_t dst_addr;       /**< Destination IP prefix (net byte order)      */
  uint32_t dst_mask;       /**< Destination prefix mask (0 = wildcard)      */
  uint16_t src_port;       /**< Source port to match    (0 = wildcard)      */
  uint16_t dst_port;       /**< Destination port to match (0 = wildcard)    */
  uint8_t  proto;          /**< IP protocol to match    (0 = wildcard)      */
  uint8_t  action;         /**< Action on match: GPU_CLASSIFY_ACTION_*      */
  uint8_t  tcp_flags_mask; /**< Bits to check in TCP flags (0 = skip)       */
  uint8_t  tcp_flags_val;  /**< Expected value after masking                */
  /*                         ───────────────────────────────────────── 24 B */
} gpu_classify_rule_t;

/* ------------------------------------------------------------------ */
/* CUDA resource bundle                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief CUDA resources owned by the plugin.
 *
 * The stream handle is kept as void* so this struct can be included
 * in VPP C files without pulling in <cuda_runtime.h>.
 */
typedef struct
{
  void	           *stream;  /**< cudaStream_t (opaque to host C code)      */
  gpu_pkt_desc_t   *descs;   /**< cudaMallocManaged: [MAX_FRAME] descriptors */
  uint8_t          *results; /**< cudaMallocManaged: [MAX_FRAME] result bytes */
} gpu_classify_cuda_res_t;

#endif /* __included_gpu_classify_types_h__ */
