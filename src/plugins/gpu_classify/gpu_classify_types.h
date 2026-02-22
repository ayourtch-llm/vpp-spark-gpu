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
 *  64 rules × 80 bytes = 5 120 bytes (well within the 64 KB limit).  */
#define GPU_CLASSIFY_MAX_RULES  64

/** Maximum packets per VPP frame (== VLIB_FRAME_SIZE).               */
#define GPU_CLASSIFY_MAX_FRAME  256

/**
 * Number of buckets in the kernel-latency histogram.
 *
 * Log2-microsecond scale:
 *   bucket  0 : [    0,    1) us
 *   bucket  1 : [    1,    2) us
 *   bucket  2 : [    2,    4) us
 *   ...
 *   bucket k  : [ 2^(k-1), 2^k ) us   for k = 1 … 10
 *   bucket 11 : [ 1024,  ∞ ) us   (overflow)
 */
#define GPU_CLASSIFY_LAT_BUCKETS 12

/* Packet actions written to the result buffer by the GPU kernel. */
#define GPU_CLASSIFY_ACTION_PASS  0   /**< forward to next feature     */
#define GPU_CLASSIFY_ACTION_DROP  1   /**< send to error-drop          */
#define GPU_CLASSIFY_ACTION_MARK  2   /**< set flag and forward        */

/* ------------------------------------------------------------------ */
/* Packet descriptor — 64 bytes, one full cache line                  */
/* ------------------------------------------------------------------ */

/**
 * @brief Compact per-packet descriptor filled by the CPU and read by
 *        the GPU kernel.
 *
 * 64 bytes total (one cache line).
 * All multi-byte fields are in **network byte order**, consistent with
 * the raw packet bytes the CPU reads out of vlib_buffer_t.
 *
 * IPv4 packets store the 4-byte address in src_ip[0..3]; bytes [4..15]
 * are zeroed.  IPv6 packets use all 16 bytes of src_ip / dst_ip.
 */
typedef struct
{
  uint8_t  src_ip[16];  /**< Source IP address (4 or 16 bytes, NBO)      */
  uint8_t  dst_ip[16];  /**< Destination IP address (4 or 16 bytes, NBO) */
  uint16_t src_port;    /**< Source L4 port (network byte order)          */
  uint16_t dst_port;    /**< Destination L4 port (network byte order)     */
  uint8_t  ip_proto;    /**< IP protocol number (TCP=6, UDP=17, …)       */
  uint8_t  tcp_flags;   /**< TCP flags byte; 0 for non-TCP packets        */
  uint8_t  ip_version;  /**< IP version: 4 or 6                          */
  uint8_t  valid;       /**< 1 = slot is populated; 0 = padding           */
  uint8_t  payload[16]; /**< First 16 bytes after the L4 header           */
  uint8_t  _pad[8];     /**< Explicit padding to reach 64 bytes           */
  /*                         ───────────────────────────────────────── 64 B */
} gpu_pkt_desc_t;

/* ------------------------------------------------------------------ */
/* Classification rule — 80 bytes                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief One classification rule stored in GPU constant memory.
 *
 * Field semantics:
 *  - ip_version: 4, 6, or 0 (0 = match any IP version).
 *  - Zero src_mask/dst_mask bytes mean "any source/dest IP" for those
 *    bytes; a fully-zero mask matches any address (wildcard).
 *  - A zero value in src_port/dst_port means "any port".
 *  - A zero value in proto means "any protocol".
 *  - tcp_flags_mask == 0 means "don't check TCP flags".
 *
 * IPv4 rules store a 4-byte address in addr[0..3]; bytes [4..15] are
 * zeroed.  IPv6 rules use all 16 bytes.  The ip_version field prevents
 * cross-version false matches when the ip_matches() helper is used.
 *
 * All address/port fields are in **network byte order**.
 */
typedef struct
{
  uint8_t  src_addr[16];     /**< Source IP prefix (net byte order)       */
  uint8_t  src_mask[16];     /**< Source prefix mask  (0 = wildcard)      */
  uint8_t  dst_addr[16];     /**< Destination IP prefix (net byte order)  */
  uint8_t  dst_mask[16];     /**< Destination prefix mask (0 = wildcard)  */
  uint16_t src_port;         /**< Source port to match    (0 = wildcard)  */
  uint16_t dst_port;         /**< Destination port to match (0 = wildcard)*/
  uint8_t  proto;            /**< IP protocol to match    (0 = wildcard)  */
  uint8_t  action;           /**< Action on match: GPU_CLASSIFY_ACTION_*  */
  uint8_t  tcp_flags_mask;   /**< Bits to check in TCP flags (0 = skip)   */
  uint8_t  tcp_flags_val;    /**< Expected value after masking             */
  uint8_t  ip_version;       /**< IP version to match: 4, 6, or 0 (any)  */
  uint8_t  _pad[7];          /**< Explicit padding to reach 80 bytes      */
  /*                         ───────────────────────────────────────── 80 B */
} gpu_classify_rule_t;

/* ------------------------------------------------------------------ */
/* CUDA resource bundle                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief CUDA resources owned by the plugin.
 *
 * Stream and event handles are kept as void* so this struct can be
 * included in VPP C files without pulling in <cuda_runtime.h>.
 */
typedef struct
{
  void	           *stream;    /**< cudaStream_t (opaque to host C code)     */
  void             *ev_start;  /**< cudaEvent_t — kernel start timestamp     */
  void             *ev_stop;   /**< cudaEvent_t — kernel stop  timestamp     */
  gpu_pkt_desc_t   *descs;     /**< cudaMallocManaged: [MAX_FRAME] descs     */
  uint8_t          *results;   /**< cudaMallocManaged: [MAX_FRAME] results   */

  /* Rolling statistics updated by gpu_classify_launch_kernel(). */
  uint64_t n_kernel_calls;     /**< Total kernel invocations                 */
  uint64_t n_gpu_packets;      /**< Total packets submitted to the GPU       */
  float    total_kernel_ms;    /**< Cumulative GPU kernel time (ms)          */
  float    min_kernel_ms;      /**< Shortest single-frame kernel time (ms)   */
  float    max_kernel_ms;      /**< Longest  single-frame kernel time (ms)   */

  /** Log2-us latency histogram — see GPU_CLASSIFY_LAT_BUCKETS above. */
  uint64_t lat_hist[GPU_CLASSIFY_LAT_BUCKETS];
} gpu_classify_cuda_res_t;

/* ------------------------------------------------------------------ */
/* GPU device information (filled by gpu_classify_get_device_info())  */
/* ------------------------------------------------------------------ */

/**
 * @brief Static properties of the CUDA device queried at show time.
 *        Uses only stdint.h types so it can live in this shared header.
 */
typedef struct
{
  char     name[256];          /**< Human-readable device name               */
  int      compute_major;      /**< SM compute-capability major version      */
  int      compute_minor;      /**< SM compute-capability minor version      */
  uint64_t total_mem_bytes;    /**< Total device global memory in bytes      */
  int      sm_count;           /**< Number of streaming multiprocessors      */
  int      clock_rate_khz;     /**< GPU core clock rate (kHz)                */
  int      mem_clock_rate_khz; /**< Memory clock rate (kHz)                  */
  int      mem_bus_width_bits; /**< Global memory bus width in bits          */
} gpu_classify_device_info_t;

#endif /* __included_gpu_classify_types_h__ */
