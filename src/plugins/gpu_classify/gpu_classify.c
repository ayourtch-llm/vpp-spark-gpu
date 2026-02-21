/* SPDX-License-Identifier: Apache-2.0
 * GPU Packet Classifier — plugin init, CLI commands.
 *
 * Provides:
 *   gpu-classify enable <interface> [disable]
 *   gpu-classify rule add [proto <p>] [src <addr/len>] [dst <addr/len>]
 *                         [sport <port>] [dport <port>]
 *                         [tcpflags <mask> <val>]
 *                         action {pass|drop|mark}
 *   gpu-classify rule clear
 *   gpu-classify show
 */

#include <vlib/vlib.h>
#include <vnet/vnet.h>
#include <vnet/plugin/plugin.h>
#include <vnet/feature/feature.h>
#include <vnet/ip/ip4_packet.h>
#include <vppinfra/error.h>
#include <vppinfra/format.h>

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

/** Convert a prefix length (0-32) to a network-byte-order mask. */
static u32
prefixlen_to_mask (u32 prefixlen)
{
  if (prefixlen == 0)
    return 0;
  return clib_host_to_net_u32 (~0u << (32 - prefixlen));
}

/** Push the current CPU rule set to GPU constant memory. */
static int
sync_rules_to_gpu (gpu_classify_main_t *gcm)
{
  return gpu_classify_update_rules (gcm->rules, gcm->n_rules);
}

/* ------------------------------------------------------------------ */
/* CLI: gpu-classify enable <if> [disable]                            */
/* ------------------------------------------------------------------ */

static clib_error_t *
gpu_classify_enable_command_fn (vlib_main_t *vm, unformat_input_t *input,
				vlib_cli_command_t *cmd)
{
  gpu_classify_main_t *gcm = &gpu_classify_main;
  unformat_input_t     _line_input, *line_input = &_line_input;
  u32		       sw_if_index = ~0;
  u8		       enable	   = 1;
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
      else
	{
	  error = clib_error_return (0, "unknown input: `%U'",
				     format_unformat_error, line_input);
	  goto done;
	}
    }

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
  gcm->if_state[sw_if_index].ip4_enabled = enable;

  vnet_feature_enable_disable ("ip4-unicast", "gpu-classify-ip4",
				sw_if_index, enable, 0, 0);

  vlib_cli_output (vm, "gpu-classify %s on %U (ip4-unicast)",
		   enable ? "enabled" : "disabled", format_vnet_sw_if_index_name,
		   gcm->vnet_main, sw_if_index);
done:
  unformat_free (line_input);
  return error;
}

VLIB_CLI_COMMAND (gpu_classify_enable_command, static) = {
  .path	      = "gpu-classify enable",
  .short_help = "gpu-classify enable <interface> [disable]",
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

  ip4_address_t addr;
  u32	       prefix_len;
  u32	       val32, val32b;
  u8	       action_set = 0;

  while (unformat_check_input (line_input) != UNFORMAT_END_OF_INPUT)
    {
      if (unformat (line_input, "proto %d", &val32))
	rule.proto = (u8) val32;
      else if (unformat (line_input, "src %U/%d", unformat_ip4_address, &addr,
			 &prefix_len))
	{
	  rule.src_addr = addr.as_u32;
	  rule.src_mask = prefixlen_to_mask (prefix_len);
	}
      else if (unformat (line_input, "dst %U/%d", unformat_ip4_address, &addr,
			 &prefix_len))
	{
	  rule.dst_addr = addr.as_u32;
	  rule.dst_mask = prefixlen_to_mask (prefix_len);
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
      float  avg_ms  = res->total_kernel_ms / res->n_kernel_calls;
      /* Convert ms → us for display (kernel is typically sub-millisecond). */
      vlib_cli_output (vm,
		       "  Frames  : %llu\n"
		       "  Packets : %llu  (avg %.1f / frame)\n"
		       "  Latency : avg %.1f us  min %.1f us  max %.1f us"
		       "  (GPU kernel only)",
		       (unsigned long long) res->n_kernel_calls,
		       (unsigned long long) res->n_gpu_packets,
		       avg_pkt,
		       (double) avg_ms          * 1000.0,
		       (double) res->min_kernel_ms * 1000.0,
		       (double) res->max_kernel_ms * 1000.0);
    }

  /* Print per-interface enable state */
  vlib_cli_output (vm, "\nEnabled interfaces:");
  for (u32 i = 0; i < vec_len (gcm->if_state); i++)
    {
      if (gcm->if_state[i].ip4_enabled)
	vlib_cli_output (vm, "  %U (ip4)",
			 format_vnet_sw_if_index_name, gcm->vnet_main, i);
    }

  /* Print rules */
  vlib_cli_output (vm, "\nRules:");
  for (u32 i = 0; i < gcm->n_rules; i++)
    {
      gpu_classify_rule_t *r = &gcm->rules[i];
      const char *act =
	(r->action < 3) ? action_names[r->action] : "?";

      vlib_cli_output (
	vm,
	"  [%2d] proto=%-3d  src=%U/%d  dst=%U/%d  "
	"sport=%-5d dport=%-5d  tcpflags=%02x/%02x  action=%s",
	i, r->proto, format_ip4_address, &r->src_addr,
	/* mask → prefix length: /24 host-order = 0xFFFFFF00, ctz = 8, 32-8 = 24 */
	(r->src_mask == 0) ?
	  0 :
	  32 - __builtin_ctz (clib_net_to_host_u32 (r->src_mask)),
	format_ip4_address, &r->dst_addr,
	(r->dst_mask == 0) ?
	  0 :
	  32 - __builtin_ctz (clib_net_to_host_u32 (r->dst_mask)),
	clib_net_to_host_u16 (r->src_port),
	clib_net_to_host_u16 (r->dst_port), r->tcp_flags_mask,
	r->tcp_flags_val, act);
    }

  return 0;
}

VLIB_CLI_COMMAND (gpu_classify_show_command, static) = {
  .path	      = "show gpu-classify",
  .short_help = "show gpu-classify",
  .function   = gpu_classify_show_command_fn,
};
