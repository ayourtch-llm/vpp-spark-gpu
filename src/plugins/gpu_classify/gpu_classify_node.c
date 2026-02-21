/* SPDX-License-Identifier: Apache-2.0
 * GPU Packet Classifier — VPP graph node.
 *
 * Hooks onto the ip4-unicast feature arc.  For each frame of packets:
 *
 *   1. CPU pass:  extract IPv4 + TCP/UDP header fields into the
 *                 CUDA-managed descriptor buffer.
 *   2. GPU pass:  launch the CUDA kernel (256 threads, 1 block) and
 *                 wait for results.
 *   3. Route:     build the nexts[] array from the per-packet results
 *                 and hand the frame to vlib_buffer_enqueue_to_next().
 */

#include <vlib/vlib.h>
#include <vnet/vnet.h>
#include <vnet/ip/ip4_packet.h>
#include <vnet/tcp/tcp_packet.h>
#include <vnet/udp/udp_packet.h>
#include <vnet/feature/feature.h>
#include <vppinfra/error.h>

#include <gpu_classify/gpu_classify.h>

/* ------------------------------------------------------------------ */
/* Next-node indices                                                   */
/* ------------------------------------------------------------------ */

typedef enum
{
  GPU_CLASSIFY_NEXT_FEATURE, /**< Continue along the feature arc (PASS/MARK) */
  GPU_CLASSIFY_NEXT_DROP,    /**< error-drop                                 */
  GPU_CLASSIFY_N_NEXT,
} gpu_classify_next_t;

/* ------------------------------------------------------------------ */
/* Error counters                                                      */
/* ------------------------------------------------------------------ */

#define foreach_gpu_classify_error                                            \
  _ (PROCESSED, "Packets processed by GPU classifier")                       \
  _ (DROPPED, "Packets dropped by GPU classifier")                           \
  _ (MARKED, "Packets marked by GPU classifier")                             \
  _ (GPU_ERROR, "Frames skipped due to CUDA error")

typedef enum
{
#define _ (sym, str) GPU_CLASSIFY_ERROR_##sym,
  foreach_gpu_classify_error
#undef _
    GPU_CLASSIFY_N_ERROR,
} gpu_classify_error_t;

static char *gpu_classify_error_strings[] = {
#define _ (sym, str) str,
  foreach_gpu_classify_error
#undef _
};

/* ------------------------------------------------------------------ */
/* Packet trace record                                                 */
/* ------------------------------------------------------------------ */

typedef struct
{
  u32 sw_if_index;
  u32 src_ip4;
  u32 dst_ip4;
  u16 src_port;
  u16 dst_port;
  u8  ip_proto;
  u8  action;
} gpu_classify_trace_t;

static u8 *
format_gpu_classify_trace (u8 *s, va_list *args)
{
  CLIB_UNUSED (vlib_main_t * vm) = va_arg (*args, vlib_main_t *);
  CLIB_UNUSED (vlib_node_t * node) = va_arg (*args, vlib_node_t *);
  gpu_classify_trace_t *t = va_arg (*args, gpu_classify_trace_t *);

  static const char *action_names[] = { "PASS", "DROP", "MARK" };
  const char *action =
    (t->action < 3) ? action_names[t->action] : "UNKNOWN";

  s = format (s,
	      "GPU-CLASSIFY: sw_if_index %d action %s\n"
	      "  src %U:%d -> dst %U:%d proto %d",
	      t->sw_if_index, action, format_ip4_address, &t->src_ip4,
	      ntohs (t->src_port), format_ip4_address, &t->dst_ip4,
	      ntohs (t->dst_port), t->ip_proto);
  return s;
}

/* ------------------------------------------------------------------ */
/* Node function                                                       */
/* ------------------------------------------------------------------ */

VLIB_NODE_FN (gpu_classify_ip4_node)
(vlib_main_t *vm, vlib_node_runtime_t *node, vlib_frame_t *frame)
{
  gpu_classify_main_t *gcm = &gpu_classify_main;
  u32		      *from  = vlib_frame_vector_args (frame);
  u32		       n_left = frame->n_vectors;
  vlib_buffer_t	      *bufs[VLIB_FRAME_SIZE];
  u16		       nexts[VLIB_FRAME_SIZE];

  vlib_get_buffers (vm, from, bufs, n_left);

  /* ============================================================
   * Safety guard: if CUDA did not initialise (no GPU, no driver,
   * or toolkit not installed at build time), pass every packet
   * straight to the next feature without any GPU involvement.
   * This keeps VPP functional on non-GPU hardware even when the
   * feature arc is (mistakenly) enabled on an interface.
   * ============================================================ */
  if (PREDICT_FALSE (!gcm->cuda_ready))
    {
      for (u32 i = 0; i < n_left; i++)
	vnet_feature_next_u16 (&nexts[i], bufs[i]);
      vlib_buffer_enqueue_to_next (vm, node, from, nexts, n_left);
      return frame->n_vectors;
    }

  /* ============================================================
   * Pass 1 — CPU: extract packet headers into the managed buffer.
   * ============================================================ */
  for (u32 i = 0; i < n_left; i++)
    {
      vlib_buffer_t   *b	= bufs[i];
      gpu_pkt_desc_t *desc = &gcm->cuda_res.descs[i];

      /* In the ip4-unicast arc, current_data points at the IP header. */
      ip4_header_t *ip4 = vlib_buffer_get_current (b);
      u32 ip4_hdr_len = (ip4->ip_version_and_header_length & 0x0f) << 2;
      u32 pkt_len     = b->current_length;

      desc->src_ip4    = ip4->src_address.as_u32;
      desc->dst_ip4    = ip4->dst_address.as_u32;
      desc->ip_proto   = ip4->protocol;
      desc->ip_version = 4;
      desc->valid      = 1;
      desc->tcp_flags  = 0;
      desc->src_port   = 0;
      desc->dst_port   = 0;
      clib_memset (desc->payload, 0, sizeof (desc->payload));

      /* Sanity: need at least the IP header inside the buffer. */
      if (PREDICT_FALSE (ip4_hdr_len > pkt_len))
	continue;

      u8 *l4 = (u8 *) ip4 + ip4_hdr_len;
      u32 l4_remaining = pkt_len - ip4_hdr_len;

      if (ip4->protocol == IP_PROTOCOL_TCP &&
	  l4_remaining >= sizeof (tcp_header_t))
	{
	  tcp_header_t *tcp = (tcp_header_t *) l4;
	  desc->src_port  = tcp->src_port;
	  desc->dst_port  = tcp->dst_port;
	  desc->tcp_flags = tcp->flags;

	  u32 tcp_hdr_len = (tcp->data_offset_and_reserved >> 4) << 2;
	  u8 *payload     = l4 + tcp_hdr_len;
	  u32 payload_len =
	    (l4_remaining > tcp_hdr_len) ? l4_remaining - tcp_hdr_len : 0;
	  clib_memcpy_fast (desc->payload, payload,
			    clib_min (payload_len,
				      (u32) sizeof (desc->payload)));
	}
      else if (ip4->protocol == IP_PROTOCOL_UDP &&
	       l4_remaining >= sizeof (udp_header_t))
	{
	  udp_header_t *udp = (udp_header_t *) l4;
	  desc->src_port = udp->src_port;
	  desc->dst_port = udp->dst_port;

	  u8 *payload     = l4 + sizeof (udp_header_t);
	  u32 payload_len = (l4_remaining > sizeof (udp_header_t)) ?
			      l4_remaining - sizeof (udp_header_t) :
			      0;
	  clib_memcpy_fast (desc->payload, payload,
			    clib_min (payload_len,
				      (u32) sizeof (desc->payload)));
	}
    }

  /* ============================================================
   * Pass 2 — GPU: launch kernel, wait for results.
   * ============================================================ */
  if (PREDICT_FALSE (gpu_classify_launch_kernel (&gcm->cuda_res, n_left) !=
		     0))
    {
      /* CUDA error: pass all packets and count the frame as an error. */
      vlib_node_increment_counter (vm, node->node_index,
				   GPU_CLASSIFY_ERROR_GPU_ERROR, 1);
      for (u32 i = 0; i < n_left; i++)
	vnet_feature_next_u16 (&nexts[i], bufs[i]);
      vlib_buffer_enqueue_to_next (vm, node, from, nexts, n_left);
      return frame->n_vectors;
    }

  /* ============================================================
   * Pass 3 — CPU: interpret results and build nexts[] array.
   * ============================================================ */
  u32 n_pass = 0, n_drop = 0, n_mark = 0;

  for (u32 i = 0; i < n_left; i++)
    {
      u8 action = gcm->cuda_res.results[i];

      /* Advance the feature arc for every packet (even ones we drop —
       * modifying current_config_index on a buffer that goes to
       * error-drop is harmless as that buffer is freed immediately). */
      vnet_feature_next_u16 (&nexts[i], bufs[i]);

      if (action == GPU_CLASSIFY_ACTION_DROP)
	{
	  nexts[i]	    = GPU_CLASSIFY_NEXT_DROP;
	  bufs[i]->error = node->errors[GPU_CLASSIFY_ERROR_DROPPED];
	  n_drop++;
	}
      else if (action == GPU_CLASSIFY_ACTION_MARK)
	{
	  bufs[i]->flags |= GPU_CLASSIFY_BUFFER_FLAG_MARKED;
	  n_mark++;
	}
      else
	{
	  n_pass++;
	}

      if (PREDICT_FALSE ((node->flags & VLIB_NODE_FLAG_TRACE) &&
			  (bufs[i]->flags & VLIB_BUFFER_IS_TRACED)))
	{
	  gpu_classify_trace_t *t =
	    vlib_add_trace (vm, node, bufs[i], sizeof (*t));
	  ip4_header_t *ip4 = vlib_buffer_get_current (bufs[i]);
	  t->sw_if_index   = vnet_buffer (bufs[i])->sw_if_index[VLIB_RX];
	  t->src_ip4	   = ip4->src_address.as_u32;
	  t->dst_ip4	   = ip4->dst_address.as_u32;
	  t->src_port	   = gcm->cuda_res.descs[i].src_port;
	  t->dst_port	   = gcm->cuda_res.descs[i].dst_port;
	  t->ip_proto	   = ip4->protocol;
	  t->action	   = action;
	}
    }

  /* Update counters in the main struct (single-threaded for now). */
  gcm->n_pass += n_pass;
  gcm->n_drop += n_drop;
  gcm->n_mark += n_mark;

  vlib_node_increment_counter (vm, node->node_index,
			       GPU_CLASSIFY_ERROR_PROCESSED, n_left);
  vlib_node_increment_counter (vm, node->node_index,
			       GPU_CLASSIFY_ERROR_DROPPED, n_drop);
  vlib_node_increment_counter (vm, node->node_index,
			       GPU_CLASSIFY_ERROR_MARKED, n_mark);

  vlib_buffer_enqueue_to_next (vm, node, from, nexts, n_left);
  return frame->n_vectors;
}

/* ------------------------------------------------------------------ */
/* Node registration                                                   */
/* ------------------------------------------------------------------ */

VLIB_REGISTER_NODE (gpu_classify_ip4_node) = {
  .name	       = "gpu-classify-ip4",
  .vector_size = sizeof (u32),
  .format_trace = format_gpu_classify_trace,
  .type	       = VLIB_NODE_TYPE_INTERNAL,

  .n_errors	    = ARRAY_LEN (gpu_classify_error_strings),
  .error_strings = gpu_classify_error_strings,

  .n_next_nodes = GPU_CLASSIFY_N_NEXT,
  .next_nodes	= {
    /* NEXT_FEATURE is a placeholder; the real next is determined at
     * runtime by vnet_feature_next_u16() from the feature arc config. */
    [GPU_CLASSIFY_NEXT_FEATURE] = "ip4-lookup",
    [GPU_CLASSIFY_NEXT_DROP]    = "error-drop",
  },
};

/* Register as a feature on the ip4-unicast arc, running before
 * ip4-flow-classify (and therefore before ip4-lookup).              */
VNET_FEATURE_INIT (gpu_classify_ip4_feature, static) = {
  .arc_name    = "ip4-unicast",
  .node_name   = "gpu-classify-ip4",
  .runs_before = VNET_FEATURES ("ip4-flow-classify"),
};
