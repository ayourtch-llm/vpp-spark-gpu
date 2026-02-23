/* SPDX-License-Identifier: Apache-2.0
 * GPU Packet Classifier — plugin init, CLI commands.
 *
 * Provides:
 *   gpu-classify enable <interface> [ip4] [ip6] [disable]
 *   gpu-classify rule add [proto <p>] [src <addr/len>] [dst <addr/len>]
 *                         [sport <port>] [dport <port>]
 *                         [tcpflags <mask> <val>]
 *                         action {pass|drop|mark}
 *   gpu-classify rule clear
 *   show gpu-classify
 *
 * src/dst addresses are auto-detected: dotted-decimal → IPv4,
 * colon-hex → IPv6.  Rules with no address fields set ip_version=0
 * (match any IP version).
 */

#include <vlib/vlib.h>
#include <vnet/vnet.h>
#include <vnet/plugin/plugin.h>
#include <vnet/feature/feature.h>
#include <vnet/ip/ip4_packet.h>
#include <vnet/ip/ip6_packet.h>
#include <vppinfra/error.h>
#include <vppinfra/format.h>
#include <vpp/app/version.h>

#include <gpu_classify/gpu_classify.h>

gpu_classify_main_t gpu_classify_main;

/* ------------------------------------------------------------------ */
/* Plugin registration                                                 */
/* ------------------------------------------------------------------ */

VLIB_PLUGIN_REGISTER () = {
  .version     = VPP_BUILD_VER,
  .description = "GPU Packet Classifier (NVIDIA CUDA / Blackwell GB10)",
};

/* ------------------------------------------------------------------ */
/* Init / exit                                                         */
/* ------------------------------------------------------------------ */

static clib_error_t *
gpu_classify_init (vlib_main_t *vm)
{
  gpu_classify_main_t *gcm = &gpu_classify_main;

  clib_memset (gcm, 0, sizeof (*gcm));
  gcm->vlib_main  = vm;
  gcm->vnet_main  = vnet_get_main ();
  gcm->log_class  = vlib_log_register_class ("gpu_classify", 0);
  clib_spinlock_init (&gcm->rules_lock);

  if (gpu_classify_cuda_init (&gcm->cuda_res) != 0)
    {
      /* Non-fatal: log a warning and continue without GPU acceleration.
       * The node will pass all packets through unchanged until a GPU
       * becomes available (requires VPP restart).                       */
      vlib_log_warn (gcm->log_class,
		     "CUDA initialisation failed — no NVIDIA GPU or driver? "
		     "Plugin loaded but GPU classification is disabled. "
		     "VPP will continue normally.");
      return 0; /* NOT an error — VPP keeps running */
    }

  gcm->cuda_ready = 1;
  vlib_log_notice (gcm->log_class,
		   "GPU classify initialised (GB10 Blackwell, NVLink-C2C)");
  return 0;
}

VLIB_INIT_FUNCTION (gpu_classify_init);

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/**
 * Compute an approximate percentile from the log2-us latency histogram.
 *
 * Uses linear interpolation within the matched bucket.
 * For the overflow bucket (>= 1024 us) returns the lower bound (1024 us).
 *
 * @param hist   GPU_CLASSIFY_LAT_BUCKETS-element histogram array.
 * @param total  Sum of all bucket counts (== n_kernel_calls).
 * @param pct    Target percentile in [0.0, 1.0].
 * @return       Approximate latency in microseconds.
 */
static double
lat_hist_percentile (u64 *hist, u64 total, double pct)
{
  /* Lower / upper bounds of each bucket in microseconds. */
  static const double lo[GPU_CLASSIFY_LAT_BUCKETS] = {
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  };
  static const double hi[GPU_CLASSIFY_LAT_BUCKETS] = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
  };

  if (total == 0)
    return 0.0;

  u64 target = (u64) (pct * (double) total);
  u64 cum    = 0;

  for (int b = 0; b < GPU_CLASSIFY_LAT_BUCKETS; b++)
    {
      if (cum + hist[b] > target)
	{
	  /* Overflow bucket: just report the lower bound. */
	  if (b == GPU_CLASSIFY_LAT_BUCKETS - 1)
	    return lo[b];
	  double frac = (double) (target - cum) / hist[b];
	  return lo[b] + frac * (hi[b] - lo[b]);
	}
      cum += hist[b];
    }
  return lo[GPU_CLASSIFY_LAT_BUCKETS - 1]; /* unreachable */
}

/** Convert a prefix length (0-32) to a network-byte-order IPv4 mask. */
static u32
prefixlen_to_mask (u32 prefixlen)
{
  if (prefixlen == 0)
    return 0;
  return clib_host_to_net_u32 (~0u << (32 - prefixlen));
}

/**
 * Fill a 16-byte buffer with the IPv6 network mask for @a plen (0–128).
 * Full bytes are set to 0xff; the partial byte (if any) has leading 1s.
 */
static void
prefixlen6_to_mask (u32 plen, u8 *mask)
{
  clib_memset (mask, 0, 16);
  u32 full_bytes = plen / 8;
  u32 rem_bits   = plen % 8;

  for (u32 i = 0; i < full_bytes; i++)
    mask[i] = 0xff;
  if (rem_bits)
    mask[full_bytes] = (u8) (0xff << (8 - rem_bits));
}

/**
 * Count the number of leading 1-bits in an @a nbytes mask.
 * Used to convert a network mask back to a prefix length for display.
 */
static u32
mask_to_plen (const u8 *mask, int nbytes)
{
  u32 plen = 0;
  for (int i = 0; i < nbytes; i++)
    {
      if (mask[i] == 0xff)
	{
	  plen += 8;
	  continue;
	}
      /* Count leading 1-bits in the partial byte. */
      u8 b = mask[i];
      while (b & 0x80)
	{
	  plen++;
	  b <<= 1;
	}
      break;
    }
  return plen;
}

/** Push the current CPU rule set to GPU constant memory. */
static int
sync_rules_to_gpu (gpu_classify_main_t *gcm)
{
  return gpu_classify_update_rules (&gcm->cuda_res, gcm->rules, gcm->n_rules);
}

/* ------------------------------------------------------------------ */
/* CLI: gpu-classify enable <if> [ip4] [ip6] [disable]               */
/* ------------------------------------------------------------------ */

static clib_error_t *
gpu_classify_enable_command_fn (vlib_main_t *vm, unformat_input_t *input,
				vlib_cli_command_t *cmd)
{
  gpu_classify_main_t *gcm = &gpu_classify_main;
  unformat_input_t     _line_input, *line_input = &_line_input;
  u32		       sw_if_index = ~0;
  u8		       enable	   = 1;
  u8		       do_ip4	   = 0;
  u8		       do_ip6	   = 0;
  clib_error_t	      *error	   = 0;

  if (!unformat_user (input, unformat_line_input, line_input))
    return 0;

  while (unformat_check_input (line_input) != UNFORMAT_END_OF_INPUT)
    {
      if (unformat (line_input, "%U", unformat_vnet_sw_interface,
		    gcm->vnet_main, &sw_if_index))
	;
      else if (unformat (line_input, "disable"))
	enable = 0;
      else if (unformat (line_input, "ip4"))
	do_ip4 = 1;
      else if (unformat (line_input, "ip6"))
	do_ip6 = 1;
      else
	{
	  error = clib_error_return (0, "unknown input: `%U'",
				     format_unformat_error, line_input);
	  goto done;
	}
    }

  /* If neither ip4 nor ip6 specified, affect both. */
  if (!do_ip4 && !do_ip6)
    do_ip4 = do_ip6 = 1;

  if (sw_if_index == ~0)
    {
      error = clib_error_return (0, "please specify an interface");
      goto done;
    }

  if (enable && !gcm->cuda_ready)
    {
      error = clib_error_return (
	0, "gpu_classify: CUDA not initialised — "
	   "cannot enable on hardware without an NVIDIA GPU / CUDA driver");
      goto done;
    }

  vec_validate_init_empty (gcm->if_state, sw_if_index,
			   (gpu_classify_if_state_t){ 0 });

  if (do_ip4)
    {
      gcm->if_state[sw_if_index].ip4_enabled = enable;
      vnet_feature_enable_disable ("ip4-unicast", "gpu-classify-ip4",
				   sw_if_index, enable, 0, 0);
    }
  if (do_ip6)
    {
      gcm->if_state[sw_if_index].ip6_enabled = enable;
      vnet_feature_enable_disable ("ip6-unicast", "gpu-classify-ip6",
				   sw_if_index, enable, 0, 0);
    }

  vlib_cli_output (vm, "gpu-classify %s on %U (%s%s%s)",
		   enable ? "enabled" : "disabled",
		   format_vnet_sw_if_index_name, gcm->vnet_main, sw_if_index,
		   do_ip4 ? "ip4" : "",
		   (do_ip4 && do_ip6) ? "+" : "",
		   do_ip6 ? "ip6" : "");
done:
  unformat_free (line_input);
  return error;
}

VLIB_CLI_COMMAND (gpu_classify_enable_command, static) = {
  .path	      = "gpu-classify enable",
  .short_help = "gpu-classify enable <interface> [ip4] [ip6] [disable]",
  .function   = gpu_classify_enable_command_fn,
};

/* ------------------------------------------------------------------ */
/* CLI: gpu-classify rule add …                                        */
/* ------------------------------------------------------------------ */

static clib_error_t *
gpu_classify_rule_command_fn (vlib_main_t *vm, unformat_input_t *input,
			      vlib_cli_command_t *cmd)
{
  gpu_classify_main_t *gcm = &gpu_classify_main;
  unformat_input_t     _line_input, *line_input = &_line_input;
  clib_error_t	      *error = 0;

  if (!unformat_user (input, unformat_line_input, line_input))
    return 0;

  /* Subcommand dispatch */
  if (unformat (line_input, "clear"))
    {
      clib_spinlock_lock (&gcm->rules_lock);
      gcm->n_rules = 0;
      sync_rules_to_gpu (gcm);
      clib_spinlock_unlock (&gcm->rules_lock);
      vlib_cli_output (vm, "gpu-classify: all rules cleared");
      goto done;
    }

  if (!unformat (line_input, "add"))
    {
      error = clib_error_return (0, "expected 'add' or 'clear'");
      goto done;
    }

  if (!gcm->cuda_ready)
    {
      error = clib_error_return (
	0, "gpu_classify: CUDA not initialised — "
	   "rules are accepted only when a CUDA-capable GPU is present");
      goto done;
    }

  if (gcm->n_rules >= GPU_CLASSIFY_MAX_RULES)
    {
      error = clib_error_return (0, "rule table full (%d rules max)",
				 GPU_CLASSIFY_MAX_RULES);
      goto done;
    }

  gpu_classify_rule_t rule;
  clib_memset (&rule, 0, sizeof (rule));

  ip4_address_t addr4;
  ip6_address_t addr6;
  u32	       prefix_len;
  u32	       val32, val32b;
  u8	       action_set  = 0;
  u8	       src_ip_ver  = 0;
  u8	       dst_ip_ver  = 0;

  while (unformat_check_input (line_input) != UNFORMAT_END_OF_INPUT)
    {
      if (unformat (line_input, "proto %d", &val32))
	rule.proto = (u8) val32;

      /* ---- Source address: try IPv4 first, then IPv6 -------------- */
      else if (unformat (line_input, "src %U/%d", unformat_ip4_address,
			 &addr4, &prefix_len))
	{
	  clib_memset (rule.src_addr, 0, 16);
	  clib_memcpy (rule.src_addr, &addr4.as_u32, 4);
	  clib_memset (rule.src_mask, 0, 16);
	  u32 mask4 = prefixlen_to_mask (prefix_len);
	  clib_memcpy (rule.src_mask, &mask4, 4);
	  src_ip_ver = 4;
	}
      else if (unformat (line_input, "src %U/%d", unformat_ip6_address,
			 &addr6, &prefix_len))
	{
	  clib_memcpy (rule.src_addr, addr6.as_u8, 16);
	  prefixlen6_to_mask (prefix_len, rule.src_mask);
	  src_ip_ver = 6;
	}

      /* ---- Destination address: try IPv4 first, then IPv6 --------- */
      else if (unformat (line_input, "dst %U/%d", unformat_ip4_address,
			 &addr4, &prefix_len))
	{
	  clib_memset (rule.dst_addr, 0, 16);
	  clib_memcpy (rule.dst_addr, &addr4.as_u32, 4);
	  clib_memset (rule.dst_mask, 0, 16);
	  u32 mask4 = prefixlen_to_mask (prefix_len);
	  clib_memcpy (rule.dst_mask, &mask4, 4);
	  dst_ip_ver = 4;
	}
      else if (unformat (line_input, "dst %U/%d", unformat_ip6_address,
			 &addr6, &prefix_len))
	{
	  clib_memcpy (rule.dst_addr, addr6.as_u8, 16);
	  prefixlen6_to_mask (prefix_len, rule.dst_mask);
	  dst_ip_ver = 6;
	}

      else if (unformat (line_input, "sport %d", &val32))
	rule.src_port = clib_host_to_net_u16 ((u16) val32);
      else if (unformat (line_input, "dport %d", &val32))
	rule.dst_port = clib_host_to_net_u16 ((u16) val32);
      else if (unformat (line_input, "tcpflags %d %d", &val32, &val32b))
	{
	  rule.tcp_flags_mask = (u8) val32;
	  rule.tcp_flags_val  = (u8) val32b;
	}
      else if (unformat (line_input, "action pass"))
	{
	  rule.action = GPU_CLASSIFY_ACTION_PASS;
	  action_set  = 1;
	}
      else if (unformat (line_input, "action drop"))
	{
	  rule.action = GPU_CLASSIFY_ACTION_DROP;
	  action_set  = 1;
	}
      else if (unformat (line_input, "action mark"))
	{
	  rule.action = GPU_CLASSIFY_ACTION_MARK;
	  action_set  = 1;
	}
      else
	{
	  error = clib_error_return (0, "unknown parameter: `%U'",
				     format_unformat_error, line_input);
	  goto done;
	}
    }

  /* Validate: cannot mix IPv4 src with IPv6 dst (or vice versa). */
  if (src_ip_ver && dst_ip_ver && src_ip_ver != dst_ip_ver)
    {
      error = clib_error_return (
	0, "cannot mix IPv4 src with IPv6 dst (or vice versa)");
      goto done;
    }

  /* ip_version = 0 means "any version" (no address constraints). */
  rule.ip_version = src_ip_ver ? src_ip_ver : dst_ip_ver;

  if (!action_set)
    {
      error = clib_error_return (0, "action {pass|drop|mark} is required");
      goto done;
    }

  clib_spinlock_lock (&gcm->rules_lock);
  gcm->rules[gcm->n_rules++] = rule;
  int rc			= sync_rules_to_gpu (gcm);
  clib_spinlock_unlock (&gcm->rules_lock);

  if (rc != 0)
    error = clib_error_return (0, "failed to push rules to GPU");
  else
    vlib_cli_output (vm, "gpu-classify: rule %d added (total %d)",
		     gcm->n_rules - 1, gcm->n_rules);

done:
  unformat_free (line_input);
  return error;
}

VLIB_CLI_COMMAND (gpu_classify_rule_command, static) = {
  .path	      = "gpu-classify rule",
  .short_help = "gpu-classify rule add [proto <N>] "
		"[src <addr/len>] [dst <addr/len>] "
		"[sport <port>] [dport <port>] "
		"[tcpflags <mask> <val>] "
		"action {pass|drop|mark}  |  clear",
  .function = gpu_classify_rule_command_fn,
};

/* ------------------------------------------------------------------ */
/* CLI: gpu-classify show                                              */
/* ------------------------------------------------------------------ */

static clib_error_t *
gpu_classify_show_command_fn (vlib_main_t *vm, unformat_input_t *input,
			      vlib_cli_command_t *cmd)
{
  gpu_classify_main_t      *gcm = &gpu_classify_main;
  gpu_classify_cuda_res_t  *res = &gcm->cuda_res;

  static const char *action_names[] = { "pass", "drop", "mark" };

  vlib_cli_output (vm,
		   "GPU Packet Classifier\n"
		   "  Backend : NVIDIA Blackwell GB10 (sm_100, NVLink-C2C)\n"
		   "  CUDA    : %s\n"
		   "  Rules   : %d / %d\n"
		   "  Counters: pass=%llu  drop=%llu  mark=%llu",
		   gcm->cuda_ready ? "ready" : "NOT available (no GPU or driver)",
		   gcm->n_rules, GPU_CLASSIFY_MAX_RULES,
		   (unsigned long long) gcm->n_pass,
		   (unsigned long long) gcm->n_drop,
		   (unsigned long long) gcm->n_mark);

  /* Hash-table summary line (after CUDA init only). */
  if (gcm->cuda_ready)
    {
      if (res->n_hash_tables > 0)
	{
	  u32 total_hash_slots = 0;
	  for (int t = 0; t < res->n_hash_tables; t++)
	    total_hash_slots += res->hash_descs[t].n_slots;
	  vlib_cli_output (vm, "  Hash    : %d tables, %u total slots",
			   res->n_hash_tables, total_hash_slots);
	}
      else
	{
	  vlib_cli_output (vm, "  Hash    : 0 tables (linear scan fallback)");
	}
    }

  /* ---- GPU device properties ---- */
  if (gcm->cuda_ready)
    {
      gpu_classify_device_info_t info;
      if (gpu_classify_get_device_info (&info) == 0)
	{
	  /* Express total memory in GiB (rounded to one decimal). */
	  double mem_gib = (double) info.total_mem_bytes / (1024.0 * 1024.0 * 1024.0);
	  vlib_cli_output (vm,
			   "\nGPU Device:\n"
			   "  Name    : %s\n"
			   "  Compute : sm_%d.%d\n"
			   "  Memory  : %.1f GiB\n"
			   "  SMs     : %d\n"
			   "  Clocks  : core %.2f GHz  |  %d-bit memory @ %.2f GHz",
			   info.name,
			   info.compute_major, info.compute_minor,
			   mem_gib,
			   info.sm_count,
			   info.clock_rate_khz     / 1e6,
			   info.mem_bus_width_bits,
			   info.mem_clock_rate_khz / 1e6);
	}
    }

  /* ---- Kernel statistics ---- */
  vlib_cli_output (vm, "\nKernel Statistics:");
  if (res->n_kernel_calls == 0)
    {
      vlib_cli_output (vm, "  No frames processed yet.");
    }
  else
    {
      double avg_pkt = (double) res->n_gpu_packets / res->n_kernel_calls;
      double avg_us  = (double) res->total_kernel_ms / res->n_kernel_calls
		       * 1000.0;
      double p50   = lat_hist_percentile (res->lat_hist, res->n_kernel_calls,
					  0.500);
      double p99   = lat_hist_percentile (res->lat_hist, res->n_kernel_calls,
					  0.990);
      double p999  = lat_hist_percentile (res->lat_hist, res->n_kernel_calls,
					  0.999);
      /* All times in microseconds; measured as CPU-observed round-trip. */
      vlib_cli_output (vm,
		       "  Dispatch: %s (busy_frames=%u  idle_frames=%u)\n"
		       "  Frames  : %llu\n"
		       "  Packets : %llu  (avg %.1f / frame)\n"
		       "  Latency : avg %.1f us  min %.1f us  max %.1f us"
		       "  (GPU round-trip)\n"
		       "  Pctiles : p50 %.1f us  p99 %.1f us  p99.9 %.1f us",
		       res->persist_active ? "persistent" : "on-demand",
		       res->busy_frames, res->idle_frames,
		       (unsigned long long) res->n_kernel_calls,
		       (unsigned long long) res->n_gpu_packets,
		       avg_pkt,
		       avg_us,
		       (double) res->min_kernel_ms * 1000.0,
		       (double) res->max_kernel_ms * 1000.0,
		       p50, p99, p999);
    }

  /* ---- Per-interface enable state ---- */
  vlib_cli_output (vm, "\nEnabled interfaces:");
  for (u32 i = 0; i < vec_len (gcm->if_state); i++)
    {
      u8 v4 = gcm->if_state[i].ip4_enabled;
      u8 v6 = gcm->if_state[i].ip6_enabled;
      if (v4 || v6)
	vlib_cli_output (vm, "  %U (%s%s%s)",
			 format_vnet_sw_if_index_name, gcm->vnet_main, i,
			 v4 ? "ip4" : "",
			 (v4 && v6) ? "+" : "",
			 v6 ? "ip6" : "");
    }

  /* ---- Rules ---- */
  vlib_cli_output (vm, "\nRules:");
  for (u32 i = 0; i < gcm->n_rules; i++)
    {
      gpu_classify_rule_t *r = &gcm->rules[i];
      const char *act =
	(r->action < 3) ? action_names[r->action] : "?";

      if (r->ip_version == 6)
	vlib_cli_output (
	  vm,
	  "  [%2d] v6  proto=%-3d  src=%U/%d  dst=%U/%d  "
	  "sport=%-5d dport=%-5d  tcpflags=%02x/%02x  action=%s",
	  i, r->proto,
	  format_ip6_address, (ip6_address_t *) r->src_addr,
	  mask_to_plen (r->src_mask, 16),
	  format_ip6_address, (ip6_address_t *) r->dst_addr,
	  mask_to_plen (r->dst_mask, 16),
	  clib_net_to_host_u16 (r->src_port),
	  clib_net_to_host_u16 (r->dst_port),
	  r->tcp_flags_mask, r->tcp_flags_val, act);
      else
	vlib_cli_output (
	  vm,
	  "  [%2d] v%d  proto=%-3d  src=%U/%d  dst=%U/%d  "
	  "sport=%-5d dport=%-5d  tcpflags=%02x/%02x  action=%s",
	  i, r->ip_version, r->proto,
	  format_ip4_address, (ip4_address_t *) r->src_addr,
	  mask_to_plen (r->src_mask, 4),
	  format_ip4_address, (ip4_address_t *) r->dst_addr,
	  mask_to_plen (r->dst_mask, 4),
	  clib_net_to_host_u16 (r->src_port),
	  clib_net_to_host_u16 (r->dst_port),
	  r->tcp_flags_mask, r->tcp_flags_val, act);
    }

  return 0;
}

VLIB_CLI_COMMAND (gpu_classify_show_command, static) = {
  .path	      = "show gpu-classify",
  .short_help = "show gpu-classify",
  .function   = gpu_classify_show_command_fn,
};
