#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VPP GPU Packet Classifier Plugin — test suite.

Tests the ``gpu-classify-ip4`` graph node on the ``ip4-unicast`` feature arc.
Exercises DROP, PASS, MARK actions and various rule predicates using the VPP
packet-generator interface and the standard VppTestCase infrastructure.

Hardware target: NVIDIA Blackwell GB10 (DGX Spark, NVLink-C2C).
The plugin must be compiled with CUDA support (nvcc, sm_100) and the NVIDIA
driver must be loaded for the plugin to initialise correctly.

Run with::

    make test TEST=test_gpu_classify.py                  # all tests
    make test TEST=test_gpu_classify.TestGpuClassify     # whole class
    make test TEST=test_gpu_classify.TestGpuClassify.test_drop_by_dst_port

"""

import unittest

from config import config
from framework import VppTestCase
from asfframework import VppTestRunner

from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw


# ---------------------------------------------------------------------------
# Skip the entire class if the plugin was explicitly excluded from the build.
# On hardware without an NVIDIA GPU the plugin will fail to initialise and
# the tests will error out with a "feature not found" message — that is the
# expected outcome on non-GPU machines.
# ---------------------------------------------------------------------------
@unittest.skipIf(
    "gpu_classify" in config.excluded_plugins,
    "gpu_classify plugin excluded from build — skipping GPU tests",
)
class TestGpuClassify(VppTestCase):
    """GPU Packet Classifier — functional tests (ip4-unicast feature arc)"""

    # ------------------------------------------------------------------
    # Class-level setup: VPP interfaces configured once for all tests.
    #
    #   pg0  — ingress interface  (gpu-classify-ip4 feature attached here)
    #   pg1  — egress  interface  (we capture packets forwarded by VPP)
    #
    # The setup performs two availability probes that skip the whole
    # class on non-GPU hardware without leaving VPP in a broken state:
    #
    #   Probe 1 — try feature_enable_disable().
    #     Fails when the plugin .so was not built (no CUDA toolkit at
    #     compile time), because the feature node is never registered.
    #
    #   Probe 2 — inspect "show gpu-classify" CLI output.
    #     Fails when the plugin was built but CUDA init failed at
    #     runtime (e.g. no GPU driver, or incompatible hardware).
    #     In this case the node passes all traffic through, so tests
    #     that expect DROP/MARK behaviour would give false failures.
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            cls.create_pg_interfaces(range(2))
            for iface in cls.pg_interfaces:
                iface.admin_up()
                iface.config_ip4()   # assigns 172.16.x.y/24, creates connected route
                iface.resolve_arp()  # teaches VPP the remote MAC for forwarding
        except Exception:
            cls.tearDownClass()
            raise

        # ── Probe 1: is the feature node registered? ──────────────────
        # feature_enable_disable raises if "gpu-classify-ip4" is unknown
        # (plugin .so not present because nvcc was absent at build time).
        try:
            cls.vapi.feature_enable_disable(
                enable=1,
                arc_name="ip4-unicast",
                feature_name="gpu-classify-ip4",
                sw_if_index=cls.pg0.sw_if_index,
            )
            # Immediately disable; setUp() will re-enable before each test.
            cls.vapi.feature_enable_disable(
                enable=0,
                arc_name="ip4-unicast",
                feature_name="gpu-classify-ip4",
                sw_if_index=cls.pg0.sw_if_index,
            )
        except Exception as e:
            cls.tearDownClass()
            raise unittest.SkipTest(
                f"gpu-classify-ip4 feature not registered — "
                f"plugin requires CUDA toolkit at build time ({e})"
            )

        # ── Probe 2: did CUDA initialise successfully at runtime? ─────
        # The node function passes traffic through without GPU inspection
        # when cuda_ready == 0, so DROP/MARK tests would give wrong results.
        show_out = cls.vapi.cli("show gpu-classify")
        if "CUDA    : ready" not in show_out:
            cls.tearDownClass()
            raise unittest.SkipTest(
                "gpu_classify plugin loaded but CUDA not ready — "
                "NVIDIA GPU with a compatible driver required. "
                f"(show gpu-classify reported: "
                f"{show_out.splitlines()[2] if show_out else 'no output'})"
            )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    # ------------------------------------------------------------------
    # Per-test setup / teardown
    # ------------------------------------------------------------------

    def setUp(self):
        super().setUp()

        # Enable the GPU classifier feature on pg0's ip4-unicast arc.
        # Using the generic feature API so we don't depend on a vpp-papi
        # binding for our plugin's own API (CLI-only plugin for now).
        self.vapi.feature_enable_disable(
            enable=1,
            arc_name="ip4-unicast",
            feature_name="gpu-classify-ip4",
            sw_if_index=self.pg0.sw_if_index,
        )

    def tearDown(self):
        # Disable the feature and flush all rules between tests so each
        # test starts from a clean state.
        self.vapi.feature_enable_disable(
            enable=0,
            arc_name="ip4-unicast",
            feature_name="gpu-classify-ip4",
            sw_if_index=self.pg0.sw_if_index,
        )
        self.vapi.cli("gpu-classify rule clear")
        super().tearDown()

    # ------------------------------------------------------------------
    # Packet helpers
    # ------------------------------------------------------------------

    def _pkt(self, proto="udp", src_ip=None, dst_ip=None,
             sport=1234, dport=5000, tcp_flags="S", payload_size=64):
        """Build an Ethernet/IP/TCP-or-UDP packet for the pg0→pg1 path."""
        src = src_ip if src_ip is not None else self.pg0.remote_ip4
        dst = dst_ip if dst_ip is not None else self.pg1.remote_ip4

        eth = Ether(src=self.pg0.remote_mac, dst=self.pg0.local_mac)
        ip  = IP(src=src, dst=dst, ttl=64)

        if proto == "tcp":
            l4 = TCP(sport=sport, dport=dport, flags=tcp_flags)
        else:
            l4 = UDP(sport=sport, dport=dport)

        return eth / ip / l4 / Raw(b"\xab" * payload_size)

    def _pkts(self, n, **kw):
        """Return a list of *n* identical packets built with _pkt(**kw)."""
        return [self._pkt(**kw) for _ in range(n)]

    # ------------------------------------------------------------------
    # Helper: read a gpu-classify-ip4 error counter from the stats segment.
    # ------------------------------------------------------------------

    def _err(self, counter_name):
        """Return the current absolute value of an error counter."""
        return self.statistics.get_err_counter(
            f"/err/gpu-classify-ip4/{counter_name}"
        )

    # ==================================================================
    # Test 1 — Baseline pass-through (no rules)
    # ==================================================================

    def test_01_pass_no_rules(self):
        """With zero rules every packet should pass to the next feature."""
        N   = 16
        rx  = self.send_and_expect(self.pg0, self._pkts(N), self.pg1)
        self.assertEqual(
            len(rx), N, f"Expected {N} packets to pass; got {len(rx)}"
        )

    # ==================================================================
    # Test 2 — DROP by destination port (TCP)
    # ==================================================================

    def test_02_drop_by_dst_port_tcp(self):
        """Packets matching 'proto 6 dport 8080 → drop' are discarded.
        Packets to port 80 (no matching rule) pass through."""
        self.vapi.cli(
            "gpu-classify rule add proto 6 dport 8080 action drop"
        )

        N_drop = 8
        N_pass = 5

        # Dropped packets: nothing should arrive on pg1.
        self.send_and_assert_no_replies(
            self.pg0,
            self._pkts(N_drop, proto="tcp", dport=8080),
            "TCP/8080 should be dropped",
        )

        # Non-matching packets: all should arrive on pg1.
        rx = self.send_and_expect(
            self.pg0, self._pkts(N_pass, proto="tcp", dport=80), self.pg1
        )
        self.assertEqual(len(rx), N_pass)

    # ==================================================================
    # Test 3 — DROP by source IP prefix
    # ==================================================================

    def test_03_drop_by_src_prefix(self):
        """Packets originating from 192.0.2.0/24 are dropped.
        Packets from the pg0 remote address (different /24) pass through."""
        self.vapi.cli(
            "gpu-classify rule add src 192.0.2.0/24 action drop"
        )

        N_drop = 6
        N_pass = 4

        self.send_and_assert_no_replies(
            self.pg0,
            self._pkts(N_drop, src_ip="192.0.2.55"),
            "Src 192.0.2.x should be dropped",
        )

        rx = self.send_and_expect(
            self.pg0,
            self._pkts(N_pass, src_ip=self.pg0.remote_ip4),
            self.pg1,
        )
        self.assertEqual(len(rx), N_pass)

    # ==================================================================
    # Test 4 — MARK: packets still forwarded, mark counter increments
    # ==================================================================

    def test_04_mark_forwards_packets_and_increments_counter(self):
        """Packets from 10.0.0.0/8 are MARKed but still forwarded.
        The 'Packets marked by GPU classifier' error counter must rise."""
        self.vapi.cli(
            "gpu-classify rule add src 10.0.0.0/8 action mark"
        )

        N = 12

        before_mark = self._err("Packets marked by GPU classifier")

        rx = self.send_and_expect(
            self.pg0, self._pkts(N, src_ip="10.1.2.3"), self.pg1
        )
        self.assertEqual(len(rx), N, "MARK packets must still be forwarded")

        after_mark = self._err("Packets marked by GPU classifier")
        self.assertEqual(
            after_mark - before_mark,
            N,
            f"Mark counter should rise by {N}; rose by {after_mark - before_mark}",
        )

    # ==================================================================
    # Test 5 — DROP counter increments correctly
    # ==================================================================

    def test_05_drop_counter(self):
        """The 'Packets dropped by GPU classifier' error counter must rise
        by exactly the number of packets that hit the DROP rule."""
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 7777 action drop"
        )

        N_drop = 10
        N_pass = 5

        before_drop = self._err("Packets dropped by GPU classifier")
        before_proc = self._err("Packets processed by GPU classifier")

        self.send_and_assert_no_replies(
            self.pg0, self._pkts(N_drop, proto="udp", dport=7777)
        )
        rx = self.send_and_expect(
            self.pg0, self._pkts(N_pass, proto="udp", dport=9999), self.pg1
        )
        self.assertEqual(len(rx), N_pass)

        after_drop = self._err("Packets dropped by GPU classifier")
        after_proc = self._err("Packets processed by GPU classifier")

        self.assertEqual(after_drop - before_drop, N_drop, "Drop counter mismatch")
        self.assertEqual(
            after_proc - before_proc,
            N_drop + N_pass,
            "Processed counter should count all packets",
        )

    # ==================================================================
    # Test 6 — Rule priority: first match wins
    # ==================================================================

    def test_06_first_rule_wins(self):
        """Rule 0 (DROP dport 9090) must fire before Rule 1 (MARK dport 9090).
        Only Rule 2 (MARK dport 7070) should fire for those packets."""
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 9090 action drop"
        )
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 9090 action mark"  # shadowed
        )
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 7070 action mark"
        )

        N = 4

        before_mark = self._err("Packets marked by GPU classifier")

        # dport 9090 → rule 0 (drop) fires, rule 1 (mark) must NOT fire.
        self.send_and_assert_no_replies(
            self.pg0,
            self._pkts(N, proto="udp", dport=9090),
            "Rule 0 (drop) must shadow Rule 1 (mark) for dport 9090",
        )

        # dport 7070 → only rule 2 (mark) fires.
        rx = self.send_and_expect(
            self.pg0, self._pkts(N, proto="udp", dport=7070), self.pg1
        )
        self.assertEqual(len(rx), N)

        after_mark = self._err("Packets marked by GPU classifier")
        self.assertEqual(
            after_mark - before_mark,
            N,
            "Only dport-7070 packets (rule 2) should be counted as marked",
        )

    # ==================================================================
    # Test 7 — Wildcard protocol (proto=0 matches TCP and UDP)
    # ==================================================================

    def test_07_wildcard_protocol(self):
        """A rule without 'proto' (proto=0 → wildcard) matches both
        TCP and UDP packets to the same destination port."""
        self.vapi.cli(
            "gpu-classify rule add dport 4444 action drop"
        )

        tcp_drop = self._pkts(4, proto="tcp", dport=4444)
        udp_drop = self._pkts(4, proto="udp", dport=4444)
        tcp_pass = self._pkts(4, proto="tcp", dport=4445)

        self.send_and_assert_no_replies(
            self.pg0, tcp_drop, "TCP dport 4444 should be dropped (wildcard proto)"
        )
        self.send_and_assert_no_replies(
            self.pg0, udp_drop, "UDP dport 4444 should be dropped (wildcard proto)"
        )
        rx = self.send_and_expect(self.pg0, tcp_pass, self.pg1)
        self.assertEqual(len(rx), 4, "TCP dport 4445 should pass")

    # ==================================================================
    # Test 8 — TCP flags matching (SYN detection)
    # ==================================================================

    def test_08_tcp_flags_syn_drop(self):
        """Rule: proto=6 dport=443 tcpflags 0x02/0x02 → DROP (SYN only).
        ACK packets to the same port must pass (SYN bit clear)."""
        # tcp_flags_mask=0x02 (SYN bit), tcp_flags_val=0x02
        self.vapi.cli(
            "gpu-classify rule add proto 6 dport 443 tcpflags 2 2 action drop"
        )

        N = 6
        syn_pkts  = self._pkts(N, proto="tcp", dport=443, tcp_flags="S")   # 0x02
        ack_pkts  = self._pkts(N, proto="tcp", dport=443, tcp_flags="A")   # 0x10
        other_pkts = self._pkts(N, proto="tcp", dport=80,  tcp_flags="S")  # wrong port

        self.send_and_assert_no_replies(
            self.pg0, syn_pkts, "TCP SYN to port 443 must be dropped"
        )

        rx = self.send_and_expect(self.pg0, ack_pkts, self.pg1)
        self.assertEqual(
            len(rx), N, "TCP ACK to port 443 must pass (SYN flag is clear)"
        )

        rx = self.send_and_expect(self.pg0, other_pkts, self.pg1)
        self.assertEqual(
            len(rx), N, "TCP SYN to port 80 must pass (port doesn't match rule)"
        )

    # ==================================================================
    # Test 9 — Multiple rules, mixed actions in one batch
    # ==================================================================

    def test_09_mixed_actions_in_one_frame(self):
        """One VPP frame containing pass, drop, and mark packets is
        processed correctly by the GPU kernel in a single launch."""
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 1111 action drop"
        )
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 2222 action mark"
        )
        # dport 3333 → no matching rule → implicit pass

        N = 20

        drop_pkts = self._pkts(N, proto="udp", dport=1111)
        mark_pkts = self._pkts(N, proto="udp", dport=2222)
        pass_pkts = self._pkts(N, proto="udp", dport=3333)

        before_drop = self._err("Packets dropped by GPU classifier")
        before_mark = self._err("Packets marked by GPU classifier")

        self.send_and_assert_no_replies(self.pg0, drop_pkts)
        rx_mark = self.send_and_expect(self.pg0, mark_pkts, self.pg1)
        rx_pass = self.send_and_expect(self.pg0, pass_pkts, self.pg1)

        self.assertEqual(len(rx_mark), N, "Marked packets must be forwarded")
        self.assertEqual(len(rx_pass), N, "Pass packets must be forwarded")

        after_drop = self._err("Packets dropped by GPU classifier")
        after_mark = self._err("Packets marked by GPU classifier")

        self.assertEqual(after_drop - before_drop, N, "Drop counter mismatch")
        self.assertEqual(after_mark - before_mark, N, "Mark counter mismatch")

    # ==================================================================
    # Test 10 — Full 256-packet frame (== VLIB_FRAME_SIZE)
    # ==================================================================

    def test_10_full_vlib_frame(self):
        """Send the maximum frame size (256 packets) through the GPU
        kernel in a single launch and verify all packets are handled."""
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 5555 action drop"
        )

        # 128 drop + 128 pass = 256 total, matching VLIB_FRAME_SIZE.
        N = 128
        drop_pkts = self._pkts(N, proto="udp", dport=5555)
        pass_pkts = self._pkts(N, proto="udp", dport=6666)

        self.send_and_assert_no_replies(self.pg0, drop_pkts)
        rx = self.send_and_expect(self.pg0, pass_pkts, self.pg1)
        self.assertEqual(len(rx), N)

    # ==================================================================
    # Test 11 — src prefix + proto + port: all fields must match
    # ==================================================================

    def test_11_combined_rule_all_fields_must_match(self):
        """A rule with src-prefix, proto, and dport must only fire when
        ALL three predicates are satisfied simultaneously.

        All test packets use dst=pg1.remote_ip4 so they are routeable
        and will appear on pg1 when they pass the GPU classifier.
        """
        # Rule: src 198.51.100.0/24, proto UDP (17), dport 53 → drop.
        self.vapi.cli(
            "gpu-classify rule add src 198.51.100.0/24 proto 17 dport 53 "
            "action drop"
        )

        dst = self.pg1.remote_ip4   # always routeable

        # All three predicates match → DROP.
        full_match = self._pkts(
            5, proto="udp", src_ip="198.51.100.10", dst_ip=dst, dport=53
        )
        # Wrong proto (TCP ≠ UDP) → PASS.
        wrong_proto = self._pkts(
            5, proto="tcp", src_ip="198.51.100.10", dst_ip=dst, dport=53
        )
        # Wrong src prefix (not in 198.51.100.0/24) → PASS.
        wrong_src = self._pkts(
            5, proto="udp", src_ip=self.pg0.remote_ip4, dst_ip=dst, dport=53
        )
        # Wrong port (54 ≠ 53) → PASS.
        wrong_port = self._pkts(
            5, proto="udp", src_ip="198.51.100.10", dst_ip=dst, dport=54
        )

        self.send_and_assert_no_replies(
            self.pg0, full_match, "Full match must drop"
        )
        rx = self.send_and_expect(self.pg0, wrong_proto, self.pg1)
        self.assertEqual(len(rx), 5, "Wrong proto: must not drop")

        rx = self.send_and_expect(self.pg0, wrong_src, self.pg1)
        self.assertEqual(len(rx), 5, "Wrong src prefix: must not drop")

        rx = self.send_and_expect(self.pg0, wrong_port, self.pg1)
        self.assertEqual(len(rx), 5, "Wrong port: must not drop")

    # ==================================================================
    # Test 12 — show gpu-classify CLI output is well-formed
    # ==================================================================

    def test_12_show_command(self):
        """'show gpu-classify' must run without error and include
        key strings: plugin name, GPU name, and any added rule."""
        self.vapi.cli("gpu-classify rule add proto 6 dport 22 action drop")

        out = self.vapi.cli("show gpu-classify")

        self.assertIn("GPU Packet Classifier", out, "Missing plugin header")
        self.assertIn("Blackwell", out, "Missing GPU name")
        self.assertIn("drop", out, "Missing action in rule dump")

    # ==================================================================
    # Test 13 — Feature disabled: GPU node must not intercept traffic
    # ==================================================================

    def test_13_feature_disabled_passes_all(self):
        """After disabling the feature the GPU node is bypassed entirely.
        Packets destined for the DROP port must pass through."""
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 9999 action drop"
        )

        # Disable the feature (tearDown will also disable it, that is fine).
        self.vapi.feature_enable_disable(
            enable=0,
            arc_name="ip4-unicast",
            feature_name="gpu-classify-ip4",
            sw_if_index=self.pg0.sw_if_index,
        )

        # Packets to dport 9999 should now pass — the node is not in the arc.
        rx = self.send_and_expect(
            self.pg0,
            self._pkts(8, proto="udp", dport=9999),
            self.pg1,
        )
        self.assertEqual(len(rx), 8, "All packets must pass when feature is disabled")

        # Re-enable so tearDown's disable call succeeds cleanly.
        self.vapi.feature_enable_disable(
            enable=1,
            arc_name="ip4-unicast",
            feature_name="gpu-classify-ip4",
            sw_if_index=self.pg0.sw_if_index,
        )

    # ==================================================================
    # Test 14 — Stress: many rules, last rule fires
    # ==================================================================

    def test_14_many_rules_last_fires(self):
        """Add GPU_CLASSIFY_MAX_RULES-1 non-matching rules then one that
        fires.  This exercises the full rule-walking path on the GPU."""
        MAX_RULES = 64

        # Add 63 rules that will not match our test packet.
        for i in range(MAX_RULES - 1):
            self.vapi.cli(
                f"gpu-classify rule add proto 6 dport {10000 + i} action drop"
            )

        # Rule 63 (index 63): match the actual test packet.
        self.vapi.cli(
            "gpu-classify rule add proto 17 dport 55555 action drop"
        )

        # The test packet matches only the last rule.
        self.send_and_assert_no_replies(
            self.pg0,
            self._pkts(4, proto="udp", dport=55555),
            "Last rule (index 63) must fire",
        )

        # A packet that matches no rule must pass.
        rx = self.send_and_expect(
            self.pg0, self._pkts(4, proto="udp", dport=99), self.pg1
        )
        self.assertEqual(len(rx), 4)


if __name__ == "__main__":
    unittest.main(testRunner=VppTestRunner)
