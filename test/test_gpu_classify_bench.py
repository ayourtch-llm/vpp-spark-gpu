#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
gpu_classify benchmark — GPU kernel vs CPU ACL plugin.

Measures true packet-processing throughput (Mpps) and latency per
256-packet vlib frame (μs/frame) by injecting packets via VPP's built-in
packet generator in sustained generator mode.  No Python round-trips occur
between frames during the timed phase — VPP's pg runs N_REPS × BATCH
packets continuously and we measure wall-clock elapsed time.

The "kern μs" column reports the GPU's own internal measurement of the
dispatch round-trip (cpu → gpu → cpu), obtained from "show gpu-classify"
via clock_gettime() inside gpu_classify_launch_kernel().  It excludes all
VPP feature-arc and buffer-management overhead.  The difference between
"GPU μs/fr" (wall-clock frame time) and "kern μs" is VPP overhead.

Four scenarios, each run at rule counts {1, 8, 64, 256, 1024}:

  no-match      N deny rules; traffic matches none     → full linear scan
  first-match   N rules;      traffic matches rule 0   → exits after 1 compare
  last-match    N rules;      traffic matches rule N-1 → full linear scan + match
  diverse-pfx   64 rules with 1→2→4→8→16 distinct (src_mask, dst_mask) combos;
                traffic matches none.  ACL plugin builds one hash table per
                unique combo; GPU still scans linearly regardless.

GPU  action for "matching" rules  → MARK  (packet forwarded)
ACL  action for "matching" rules  → PERMIT (packet forwarded)

Run with::

    make test TEST=test_gpu_classify_bench

"""

import os
import re
import time
import unittest
from ipaddress import IPv4Network

from config import config
from framework import VppTestCase
from asfframework import VppTestRunner

from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP
from scapy.utils import wrpcap

from vpp_acl import AclRule, VppAcl, VppAclInterface


# ---------------------------------------------------------------------------
# Skip the entire module if gpu_classify was explicitly excluded from the build.
# ---------------------------------------------------------------------------
@unittest.skipIf(
    "gpu_classify" in config.excluded_plugins,
    "gpu_classify plugin excluded from build — skipping benchmark",
)
class TestGpuClassifyBench(VppTestCase):
    """GPU Packet Classifier — benchmark vs CPU ACL plugin (ip4-unicast arc)"""

    # ------------------------------------------------------------------
    # Tuning knobs
    # ------------------------------------------------------------------

    BATCH  = 256       # packets per vlib frame (one pg stream replay)
    N_REPS = 50000     # timed replays of the 256-packet template
    #   → total packets per measurement = BATCH × N_REPS = 12.8 M
    #   → at 10 Mpps ≈ 1.28 s; at 50 Mpps ≈ 256 ms; at 100 Mpps ≈ 128 ms

    POLL_S = 0.05      # seconds between "show packet-generator" polls

    # Rule counts to sweep (gpu_classify maximum is now 1024)
    RULE_COUNTS = [1, 8, 64, 256, 1024]

    # Port assignments — chosen so there is no accidental overlap:
    #   DENY_PORT_BASE  ports 5000-5063  →  deny rules that never match traffic
    #   MATCH_PORT_BASE ports 9000-9063  →  the "match" port per rule count
    #   PASS_PORT       port 1234        →  no-match traffic (hits none of the above)
    DENY_PORT_BASE  = 5000
    MATCH_PORT_BASE = 9000
    PASS_PORT       = 1234

    # Scenario 4: diverse src+dst prefix mask combinations.
    #
    # N_RULES_DIVERSE rules are always installed; what varies is how many
    # *distinct (src_mask, dst_mask) combinations* those rules span.
    # The ACL plugin builds one hash table per unique (src_mask, dst_mask)
    # pair, so its lookup cost grows with mask diversity.  The GPU scans
    # all N rules linearly — its cost is independent of mask diversity.
    #
    # SRC_LENGTHS = DST_LENGTHS = [8, 16, 24, 32] → 4×4 = 16 unique combos.
    #
    # MASK_COUNTS = [1, 2, 4, 8, 16]:
    #   mask_count=1  →  64 rules all (src/8,  dst/8)   →  1 ACL table
    #   mask_count=2  →  32 rules each (s/8,d/8) and (s/24,d/32) → 2 tables
    #   mask_count=4  →  16 rules each across 4 combos  →  4 ACL tables
    #   mask_count=8  →   8 rules each across 8 combos  →  8 ACL tables
    #   mask_count=16 →   4 rules each across all 16 combos → 16 ACL tables
    #
    # Src prefix pool (avoids 172.16.x.x — test traffic src is pg0.remote_ip4):
    #   /8 : 1.0.0.0/8 … 64.0.0.0/8
    #   /16: 2.0.0.0/16 … 2.63.0.0/16
    #   /24: 3.0.0.0/24 … 3.0.63.0/24
    #   /32: 4.0.0.1/32 … 4.0.0.64/32
    #
    # Dst prefix pool (avoids 172.16.x.x — test traffic dst is pg1.remote_ip4):
    #   /8 : 20.0.0.0/8 … 83.0.0.0/8   (first octets 20–83)
    #   /16: 100.0.0.0/16 … 100.63.0.0/16
    #   /24: 102.0.0.0/24 … 102.0.63.0/24
    #   /32: 105.0.0.1/32 … 105.0.0.64/32
    N_RULES_DIVERSE = 64
    MASK_COUNTS     = [1, 2, 4, 8, 16]
    SRC_LENGTHS     = [8, 16, 24, 32]
    DST_LENGTHS     = [8, 16, 24, 32]
    # All 16 (src_plen, dst_plen) combinations, row-major over SRC×DST.
    # (Must inline literals — Python 3 list comprehensions cannot see class-
    #  level names from the enclosing class body scope.)
    ALL_MASK_COMBOS = [(sl, dl) for sl in [8, 16, 24, 32]
                       for dl in [8, 16, 24, 32]]
    _SRC_PFX = {
        8:  [f"{1+i}.0.0.0/8"    for i in range(64)],   # 1/8 … 64/8
        16: [f"2.{i}.0.0/16"     for i in range(64)],   # 2.0/16 … 2.63/16
        24: [f"3.0.{i}.0/24"     for i in range(64)],   # 3.0.0/24 … 3.0.63/24
        32: [f"4.0.0.{i+1}/32"   for i in range(64)],   # 4.0.0.1/32 … 4.0.0.64/32
    }
    _DST_PFX = {
        8:  [f"{20+i}.0.0.0/8"   for i in range(64)],   # 20/8 … 83/8
        16: [f"100.{i}.0.0/16"   for i in range(64)],   # 100.0/16 … 100.63/16
        24: [f"102.0.{i}.0/24"   for i in range(64)],   # 102.0.0/24 … 102.0.63/24
        32: [f"105.0.0.{i+1}/32" for i in range(64)],   # 105.0.0.1/32 … 105.0.0.64/32
    }

    # ------------------------------------------------------------------
    # Class-level setup
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.create_pg_interfaces(range(2))
        for iface in cls.pg_interfaces:
            iface.admin_up()
            iface.config_ip4()   # assigns 172.16.x.y/24, creates connected route
            iface.resolve_arp()  # teaches VPP the remote MAC for forwarding

        cls.gpu_available = cls._probe_gpu()
        cls.acl_available = cls._probe_acl()

        if not cls.gpu_available and not cls.acl_available:
            raise unittest.SkipTest(
                "Neither gpu_classify (CUDA ready) nor ACL plugin is available"
            )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @classmethod
    def _probe_gpu(cls):
        """Return True if gpu-classify-ip4 is registered AND CUDA is ready."""
        try:
            cls.vapi.feature_enable_disable(
                enable=1,
                arc_name="ip4-unicast",
                feature_name="gpu-classify-ip4",
                sw_if_index=cls.pg0.sw_if_index,
            )
            cls.vapi.feature_enable_disable(
                enable=0,
                arc_name="ip4-unicast",
                feature_name="gpu-classify-ip4",
                sw_if_index=cls.pg0.sw_if_index,
            )
            out = cls.vapi.cli("show gpu-classify")
            return "CUDA    : ready" in out
        except Exception:
            return False

    @classmethod
    def _probe_acl(cls):
        """Return True if the ACL plugin is loaded and functional.

        Uses raw vapi calls rather than VppAcl/VppAclInterface because
        those helpers call self._test.registry.register() which only
        exists on test *instances*, not on the class in setUpClass.
        """
        try:
            r = cls.vapi.acl_add_replace(
                acl_index=0xFFFFFFFF,
                tag="probe",
                count=1,
                r=[AclRule(is_permit=1).encode()],
            )
            cls.vapi.acl_del(acl_index=r.acl_index)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Per-test setup / teardown
    # ------------------------------------------------------------------

    def setUp(self):
        super().setUp()
        self._gpu_enabled = False
        self._acl = None
        self._acl_if = None

    def tearDown(self):
        # Best-effort cleanup of any leftover pg streams.
        try:
            self.vapi.cli("packet-generator disable")
        except Exception:
            pass
        try:
            self.vapi.cli("packet-generator delete bench-stream")
        except Exception:
            pass

        # GPU cleanup.
        try:
            if self._gpu_enabled:
                self.vapi.cli("gpu-classify rule clear")
                self.vapi.feature_enable_disable(
                    enable=0,
                    arc_name="ip4-unicast",
                    feature_name="gpu-classify-ip4",
                    sw_if_index=self.pg0.sw_if_index,
                )
        except Exception:
            pass

        # ACL cleanup.
        try:
            if self._acl_if is not None:
                self._acl_if.remove_vpp_config()
            if self._acl is not None:
                self._acl.remove_vpp_config()
        except Exception:
            pass

        super().tearDown()

    # ------------------------------------------------------------------
    # Packet factory
    # ------------------------------------------------------------------

    def _pkts(self, dport):
        """Return BATCH TCP packets from pg0 → pg1 addressed to *dport*."""
        return [
            Ether(dst=self.pg0.local_mac, src=self.pg0.remote_mac)
            / IP(src=self.pg0.remote_ip4, dst=self.pg1.remote_ip4)
            / TCP(sport=12345, dport=dport)
            for _ in range(self.BATCH)
        ]

    # ------------------------------------------------------------------
    # pg generator timing core
    # ------------------------------------------------------------------

    def _pg_wait(self, stream_name, timeout=120):
        """Poll show packet-generator until the named stream is no longer
        running (Enabled column no longer shows 'Yes').
        """
        deadline = time.time() + timeout
        while True:
            status = self.vapi.cli("show packet-generator")
            still_running = any(
                stream_name in line and "Yes" in line
                for line in status.splitlines()
            )
            if not still_running:
                return
            if time.time() > deadline:
                raise TimeoutError(
                    f"pg stream '{stream_name}' did not finish within {timeout}s"
                )
            time.sleep(self.POLL_S)

    def _time_pg(self, pkts):
        """
        Inject BATCH × N_REPS packets through the active VPP data path using
        the packet generator in sustained generator mode.

        A warm-up pass (one 256-packet frame) primes the data path before the
        timed run starts.  During the timed window VPP's pg node injects all
        N_REPS × BATCH packets without any Python involvement — no round-trips,
        no synchronisation calls.  Wall-clock elapsed time is measured from
        immediately before 'packet-generator enable' to immediately after
        _pg_wait() returns, giving at most ±POLL_S timing error (~50 ms).

        pg output (packets forwarded to pg1) is deliberately NOT captured;
        the pg output node simply frees forwarded buffers, so there is no
        buffer accumulation or I/O overhead on the output side.

        Returns (mpps, us_per_frame):
          mpps         — millions of packets per second (wall-clock)
          us_per_frame — microseconds to process one 256-packet vlib frame
        """
        n_total  = len(pkts) * self.N_REPS
        pcap_in  = os.path.join(self.tempdir, "bench_in.pcap")
        wrpcap(pcap_in, pkts)

        stream_def = (
            f"packet-generator new pcap {pcap_in} "
            f"source pg0 name bench-stream"
        )

        # ---- Warm-up: one 256-packet frame ----
        self.vapi.cli(f"{stream_def} limit {len(pkts)}")
        self.vapi.cli("packet-generator enable")
        self._pg_wait("bench-stream")
        self.vapi.cli("packet-generator delete bench-stream")

        # ---- Timed run ----
        self.vapi.cli(f"{stream_def} limit {n_total}")
        t0 = time.perf_counter()
        self.vapi.cli("packet-generator enable")
        self._pg_wait("bench-stream")
        elapsed = time.perf_counter() - t0

        self.vapi.cli("packet-generator delete bench-stream")

        mpps         = n_total / elapsed / 1e6
        us_per_frame = elapsed / self.N_REPS * 1e6
        return mpps, us_per_frame

    # ------------------------------------------------------------------
    # GPU helpers
    # ------------------------------------------------------------------

    def _gpu_enable(self):
        self.vapi.feature_enable_disable(
            enable=1,
            arc_name="ip4-unicast",
            feature_name="gpu-classify-ip4",
            sw_if_index=self.pg0.sw_if_index,
        )
        self._gpu_enabled = True

    def _gpu_disable(self):
        self.vapi.cli("gpu-classify rule clear")
        self.vapi.feature_enable_disable(
            enable=0,
            arc_name="ip4-unicast",
            feature_name="gpu-classify-ip4",
            sw_if_index=self.pg0.sw_if_index,
        )
        self._gpu_enabled = False

    def _gpu_rule(self, dport, action="drop"):
        self.vapi.cli(
            f"gpu-classify rule add proto 6 dport {dport} action {action}"
        )

    def _gpu_rule_prefix(self, prefix, action="drop"):
        """Add a GPU deny rule matching *only* a dst IP prefix (any proto/port)."""
        self.vapi.cli(f"gpu-classify rule add dst {prefix} action {action}")

    def _gpu_rule_src_dst_prefix(self, src_prefix, dst_prefix, action="drop"):
        """Add a GPU rule matching both a src and dst IP prefix."""
        self.vapi.cli(
            f"gpu-classify rule add src {src_prefix} dst {dst_prefix} action {action}"
        )

    def _gpu_snapshot(self):
        """Parse 'show gpu-classify' and return kernel timing stats.

        Returns a dict with keys:
          calls   — cumulative frame count (n_kernel_calls)
          avg_us  — cumulative average kernel round-trip (µs)
          p50_us  — cumulative p50 latency (µs)
          p99_us  — cumulative p99 latency (µs)

        All fields are 0.0 / 0 if no frames have been processed yet.
        """
        out = self.vapi.cli("show gpu-classify")
        snap = {"calls": 0, "avg_us": 0.0, "p50_us": 0.0, "p99_us": 0.0}
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Frames"):
                m = re.search(r":\s*(\d+)", line)
                if m:
                    snap["calls"] = int(m.group(1))
            elif line.startswith("Latency"):
                m = re.search(r"avg\s+([\d.]+)\s+us", line)
                if m:
                    snap["avg_us"] = float(m.group(1))
            elif line.startswith("Pctiles"):
                m = re.search(
                    r"p50\s+([\d.]+)\s+us.*p99\s+([\d.]+)\s+us", line
                )
                if m:
                    snap["p50_us"] = float(m.group(1))
                    snap["p99_us"] = float(m.group(2))
        return snap

    def _gpu_delta_avg(self, before, after):
        """Return average kernel latency (µs) for frames between two snapshots.

        Uses the cumulative (calls × avg) totals to compute an exact weighted
        average for only the frames that happened between *before* and *after*.
        Returns None if no frames were processed in the interval.
        """
        delta_calls = after["calls"] - before["calls"]
        if delta_calls <= 0:
            return None
        # total_ms = avg_us * calls / 1000 (exact inverse of how avg is stored)
        total_ms_before = before["avg_us"] * before["calls"] / 1000.0
        total_ms_after  = after["avg_us"]  * after["calls"]  / 1000.0
        return (total_ms_after - total_ms_before) / delta_calls * 1000.0

    # ------------------------------------------------------------------
    # ACL helpers
    # ------------------------------------------------------------------

    def _acl_install(self, rules):
        self._acl = VppAcl(self, rules=rules, tag="bench")
        self._acl.add_vpp_config()
        self._acl_if = VppAclInterface(
            self,
            sw_if_index=self.pg0.sw_if_index,
            acls=[self._acl],
            n_input=1,
        )
        self._acl_if.add_vpp_config()

    def _acl_remove(self):
        if self._acl_if is not None:
            self._acl_if.remove_vpp_config()
            self._acl_if = None
        if self._acl is not None:
            self._acl.remove_vpp_config()
            self._acl = None

    def _acl_deny(self, dport):
        """Return a stateless deny rule for TCP dport == *dport*."""
        return AclRule(
            is_permit=0, proto=6, dport_from=dport, dport_to=dport
        )

    def _acl_permit_port(self, dport):
        """Return a stateless permit rule for TCP dport == *dport*."""
        return AclRule(
            is_permit=1, proto=6, dport_from=dport, dport_to=dport
        )

    def _acl_deny_prefix(self, prefix):
        """Return a stateless deny rule matching a dst IP prefix (any proto/port)."""
        return AclRule(is_permit=0, dst_prefix=IPv4Network(prefix))

    def _acl_deny_src_dst_prefix(self, src_prefix, dst_prefix):
        """Return a stateless deny rule matching both src and dst IP prefixes."""
        return AclRule(
            is_permit=0,
            src_prefix=IPv4Network(src_prefix),
            dst_prefix=IPv4Network(dst_prefix),
        )

    # ------------------------------------------------------------------
    # Results table helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(title):
        w = 88
        sep = "=" * w
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        print(
            f"  {'Rules':>6}  "
            f"{'GPU Mpps':>10}  {'GPU μs/fr':>10}  {'kern μs':>8}  "
            f"{'ACL Mpps':>10}  {'ACL μs/fr':>10}  "
            f"{'GPU/ACL':>8}"
        )
        print(
            f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  "
            f"{'-'*10}  {'-'*10}  {'-'*8}"
        )

    @staticmethod
    def _print_row(n_rules, gpu_mpps, gpu_us, kern_us, acl_mpps, acl_us):
        def fmt_mpps(v):
            return f"{v:10.3f}" if v is not None else f"{'n/a':>10}"

        def fmt_us(v):
            return f"{v:10.1f}" if v is not None else f"{'n/a':>10}"

        def fmt_kern(v):
            return f"{v:8.1f}" if v is not None else f"{'n/a':>8}"

        if gpu_mpps is not None and acl_mpps is not None:
            ratio = f"{gpu_mpps / acl_mpps:7.2f}x"
        else:
            ratio = f"{'n/a':>8}"

        print(
            f"  {n_rules:>6}  "
            f"{fmt_mpps(gpu_mpps)}  {fmt_us(gpu_us)}  {fmt_kern(kern_us)}  "
            f"{fmt_mpps(acl_mpps)}  {fmt_us(acl_us)}  "
            f"{ratio:>8}"
        )

    @staticmethod
    def _print_kern_note():
        print(
            "  (kern μs = GPU dispatch round-trip only, "
            "from clock_gettime inside gpu_classify_launch_kernel;\n"
            "   GPU μs/fr − kern μs = VPP feature-arc + buffer overhead)"
        )

    # ==================================================================
    # Benchmark 1 — no-match (worst-case linear scan for all rules)
    # ==================================================================

    def test_bench_01_no_match(self):
        """Benchmark: N deny rules — traffic matches NONE (full scan of all rules)."""
        # Traffic uses PASS_PORT which matches no deny rule.
        # GPU: scans all N rules, no match, packet passes.
        # ACL: scans all N deny rules, no match, hits the trailing permit-all.
        pkts = self._pkts(self.PASS_PORT)

        n_total_k = self.BATCH * self.N_REPS // 1000
        self._print_header(
            f"Scenario 1 — no-match  "
            f"(traffic hits no deny rule; {self.BATCH}-pkt frames, "
            f"{self.N_REPS} reps = {n_total_k}k pkts per measurement)"
        )

        for n in self.RULE_COUNTS:
            gpu_mpps = gpu_us = kern_us = acl_mpps = acl_us = None

            # ---- GPU ----
            if self.gpu_available:
                self._gpu_enable()
                for i in range(n):
                    self._gpu_rule(self.DENY_PORT_BASE + i, "drop")
                snap0 = self._gpu_snapshot()
                gpu_mpps, gpu_us = self._time_pg(pkts)
                snap1 = self._gpu_snapshot()
                kern_us = self._gpu_delta_avg(snap0, snap1)
                self._gpu_disable()

            # ---- ACL ----
            if self.acl_available:
                rules = (
                    [self._acl_deny(self.DENY_PORT_BASE + i) for i in range(n)]
                    + [AclRule(is_permit=1)]   # permit-all fallthrough
                )
                self._acl_install(rules)
                acl_mpps, acl_us = self._time_pg(pkts)
                self._acl_remove()

            self._print_row(n, gpu_mpps, gpu_us, kern_us, acl_mpps, acl_us)

        self._print_kern_note()

    # ==================================================================
    # Benchmark 2 — first-rule match (best case for linear scan)
    # ==================================================================

    def test_bench_02_first_match(self):
        """Benchmark: N rules — traffic always matches rule 0 (1 comparison)."""
        # Rule 0:    GPU=MARK / ACL=PERMIT for MATCH_PORT_BASE  → packet forwarded
        # Rules 1…N-1: GPU=DROP / ACL=DENY  for DENY_PORT_BASE+i → never reached
        #
        # Both plugins exit after exactly 1 rule comparison.
        match_port = self.MATCH_PORT_BASE
        pkts = self._pkts(match_port)

        n_total_k = self.BATCH * self.N_REPS // 1000
        self._print_header(
            f"Scenario 2 — first-match  "
            f"(rule 0 always fires; {self.BATCH}-pkt frames, "
            f"{self.N_REPS} reps = {n_total_k}k pkts per measurement)"
        )

        for n in self.RULE_COUNTS:
            gpu_mpps = gpu_us = kern_us = acl_mpps = acl_us = None

            # ---- GPU ----
            if self.gpu_available:
                self._gpu_enable()
                self._gpu_rule(match_port, "mark")   # rule 0: MARK → forwarded
                for i in range(1, n):
                    self._gpu_rule(self.DENY_PORT_BASE + i, "drop")
                snap0 = self._gpu_snapshot()
                gpu_mpps, gpu_us = self._time_pg(pkts)
                snap1 = self._gpu_snapshot()
                kern_us = self._gpu_delta_avg(snap0, snap1)
                self._gpu_disable()

            # ---- ACL ----
            if self.acl_available:
                rules = (
                    [self._acl_permit_port(match_port)]   # rule 0: PERMIT → forwarded
                    + [self._acl_deny(self.DENY_PORT_BASE + i) for i in range(1, n)]
                    + [AclRule(is_permit=1)]               # permit-all fallthrough
                )
                self._acl_install(rules)
                acl_mpps, acl_us = self._time_pg(pkts)
                self._acl_remove()

            self._print_row(n, gpu_mpps, gpu_us, kern_us, acl_mpps, acl_us)

        self._print_kern_note()

    # ==================================================================
    # Benchmark 3 — last-rule match (full scan before finding a match)
    # ==================================================================

    def test_bench_03_last_match(self):
        """Benchmark: N rules — traffic matches rule N-1 only (full scan + match)."""
        # Rules 0…N-2: GPU=DROP / ACL=DENY  for DENY_PORT_BASE+i → never match traffic
        # Rule N-1:    GPU=MARK / ACL=PERMIT for MATCH_PORT_BASE+N-1 → packet forwarded
        #
        # Both plugins scan N rules before finding a match, so this is the
        # worst case for match throughput (same rule-visit count as no-match
        # but with a successful match at the end).

        n_total_k = self.BATCH * self.N_REPS // 1000
        self._print_header(
            f"Scenario 3 — last-match  "
            f"(rule N-1 fires after full scan; {self.BATCH}-pkt frames, "
            f"{self.N_REPS} reps = {n_total_k}k pkts per measurement)"
        )

        for n in self.RULE_COUNTS:
            gpu_mpps = gpu_us = kern_us = acl_mpps = acl_us = None
            match_port = self.MATCH_PORT_BASE + n - 1
            pkts = self._pkts(match_port)

            # ---- GPU ----
            if self.gpu_available:
                self._gpu_enable()
                for i in range(n - 1):
                    self._gpu_rule(self.DENY_PORT_BASE + i, "drop")
                self._gpu_rule(match_port, "mark")   # rule N-1: MARK → forwarded
                snap0 = self._gpu_snapshot()
                gpu_mpps, gpu_us = self._time_pg(pkts)
                snap1 = self._gpu_snapshot()
                kern_us = self._gpu_delta_avg(snap0, snap1)
                self._gpu_disable()

            # ---- ACL ----
            if self.acl_available:
                rules = (
                    [self._acl_deny(self.DENY_PORT_BASE + i) for i in range(n - 1)]
                    + [self._acl_permit_port(match_port)]   # rule N-1: PERMIT → forwarded
                    + [AclRule(is_permit=1)]                 # permit-all fallthrough
                )
                self._acl_install(rules)
                acl_mpps, acl_us = self._time_pg(pkts)
                self._acl_remove()

            self._print_row(n, gpu_mpps, gpu_us, kern_us, acl_mpps, acl_us)

        self._print_kern_note()

    # ==================================================================
    # Benchmark 4 — fixed N rules, varying dst prefix length diversity
    # ==================================================================

    def test_bench_04_diverse_prefix(self):
        """Benchmark: 64 rules, sweep distinct src+dst prefix mask combos 1→16.

        Total rules are always N_RULES_DIVERSE=64, split evenly across the
        chosen number of distinct (src_mask, dst_mask) combinations.
        Traffic never matches any rule (no-match / pass).

        The ACL plugin builds one hash table per unique (src_mask, dst_mask)
        pair, so lookup cost grows with mask-combo diversity while rule count
        stays constant.  The GPU always scans all 64 rules linearly; its
        cost is independent of mask diversity.

        Expected trend:
          ACL µs/fr  grows with mask_count  (more hash-table probes per pkt)
          GPU µs/fr  stays flat             (same 64 rules regardless)
        """
        pkts = self._pkts(self.PASS_PORT)

        n_total_k = self.BATCH * self.N_REPS // 1000
        w = 104
        print(f"\n{'=' * w}")
        print(
            f"  Scenario 4 — diverse src+dst prefix mask combinations  "
            f"({self.N_RULES_DIVERSE} rules total, no-match traffic;\n"
            f"  {self.BATCH}-pkt frames, "
            f"{self.N_REPS} reps = {n_total_k}k pkts per measurement)"
        )
        print("=" * w)
        print(
            f"  ACL builds 1 hash table per unique (src_mask, dst_mask) pair;\n"
            f"  its per-packet cost grows with combo count.\n"
            f"  GPU always scans {self.N_RULES_DIVERSE} rules; cost is independent of mask diversity."
        )
        print(
            f"\n  {'Combos':>6}  "
            f"{'GPU Mpps':>10}  {'GPU μs/fr':>10}  {'kern μs':>8}  "
            f"{'ACL Mpps':>10}  {'ACL μs/fr':>10}  "
            f"{'GPU/ACL':>8}  "
            f"{'(src, dst) lengths used'}"
        )
        print(
            f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  "
            f"{'-'*10}  {'-'*10}  {'-'*8}  "
            f"{'-'*30}"
        )

        for mask_count in self.MASK_COUNTS:
            gpu_mpps = gpu_us = kern_us = acl_mpps = acl_us = None

            # Use the first mask_count (src_plen, dst_plen) combos;
            # split N_RULES_DIVERSE rules evenly across them.
            combos    = self.ALL_MASK_COMBOS[:mask_count]
            n_per_cbo = self.N_RULES_DIVERSE // mask_count
            rule_pairs = []   # list of (src_pfx, dst_pfx) strings
            for j, (splen, dplen) in enumerate(combos):
                for k in range(n_per_cbo):
                    rule_pairs.append(
                        (self._SRC_PFX[splen][j * n_per_cbo + k],
                         self._DST_PFX[dplen][j * n_per_cbo + k])
                    )

            combos_str = " ".join(f"(/{sl},/{dl})" for sl, dl in combos[:4])
            if len(combos) > 4:
                combos_str += f" …+{len(combos)-4}more"

            # ---- GPU ----
            if self.gpu_available:
                self._gpu_enable()
                for src_pfx, dst_pfx in rule_pairs:
                    self._gpu_rule_src_dst_prefix(src_pfx, dst_pfx, "drop")
                snap0 = self._gpu_snapshot()
                gpu_mpps, gpu_us = self._time_pg(pkts)
                snap1 = self._gpu_snapshot()
                kern_us = self._gpu_delta_avg(snap0, snap1)
                self._gpu_disable()

            # ---- ACL ----
            if self.acl_available:
                acl_rules = (
                    [self._acl_deny_src_dst_prefix(sp, dp) for sp, dp in rule_pairs]
                    + [AclRule(is_permit=1)]   # permit-all fallthrough
                )
                self._acl_install(acl_rules)
                acl_mpps, acl_us = self._time_pg(pkts)
                self._acl_remove()

            def fmt_mpps(v):
                return f"{v:10.3f}" if v is not None else f"{'n/a':>10}"
            def fmt_us(v):
                return f"{v:10.1f}" if v is not None else f"{'n/a':>10}"
            def fmt_kern(v):
                return f"{v:8.1f}" if v is not None else f"{'n/a':>8}"

            if gpu_mpps is not None and acl_mpps is not None:
                ratio = f"{gpu_mpps / acl_mpps:7.2f}x"
            else:
                ratio = f"{'n/a':>8}"

            print(
                f"  {mask_count:>6}  "
                f"{fmt_mpps(gpu_mpps)}  {fmt_us(gpu_us)}  {fmt_kern(kern_us)}  "
                f"{fmt_mpps(acl_mpps)}  {fmt_us(acl_us)}  "
                f"{ratio:>8}  "
                f"{combos_str}"
            )

        self._print_kern_note()
        print(
            f"  Note: 'Combos' = number of distinct (src_mask, dst_mask) pairs.\n"
            f"  All rows have exactly {self.N_RULES_DIVERSE} rules; only mask diversity varies.\n"
            f"  Src prefixes: 1-64/8, 2.0-63/16, 3.0.0-63/24, 4.0.0.1-64/32\n"
            f"  Dst prefixes: 20-83/8, 100.0-63/16, 102.0.0-63/24, 105.0.0.1-64/32"
        )


if __name__ == "__main__":
    unittest.main(testRunner=VppTestRunner)
