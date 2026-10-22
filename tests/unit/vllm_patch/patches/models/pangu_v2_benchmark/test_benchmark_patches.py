# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only behavior checks; can run directly without importing vLLM."""

import importlib.util
import logging
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class BenchmarkPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = {}

        def stub(name, **attributes):
            module = ModuleType(name)
            module.__path__ = []
            module.__dict__.update(attributes)
            cls.modules[name] = module
            if "." in name:
                parent, child = name.rsplit(".", 1)
                if parent not in cls.modules:
                    stub(parent)
                setattr(cls.modules[parent], child, module)
            return module

        class Scheduler:
            def __init__(self):
                self.max_num_running_reqs = 4

            def schedule(self, throttle_prefills=False):
                self.seen = (list(self.running), throttle_prefills)
                self.seen_capacity = self.max_num_running_reqs
                if getattr(self, "raise_on_schedule", False):
                    raise RuntimeError("schedule failed")
                if getattr(self, "admitted", None) is not None:
                    self.running.append(self.admitted)
                return self.seen

        class DPEngineCoreProc:
            def _has_global_unfinished_reqs(self, local_unfinished):
                self.original_calls += 1
                return local_unfinished

        class DPLBAsyncMPClient:
            def get_core_engine_for_request(self, request):
                self.original_calls += 1
                return "upstream"

        cls.envs = SimpleNamespace(
            OMNI_HYBRID_ALIGNED_DECODE=True,
            OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD=2,
            OMNI_DP_ROUND_ROBIN=True,
            OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD=2,
            is_set=lambda name: name in os.environ,
        )
        stub("vllm.logger", init_logger=logging.getLogger)
        stub("vllm.v1.core.sched.scheduler", Scheduler=Scheduler)
        stub("vllm.v1.engine.core", DPEngineCoreProc=DPEngineCoreProc)
        stub(
            "vllm.v1.engine.core_client",
            DPLBAsyncMPClient=DPLBAsyncMPClient,
            get_late_interaction_engine_index=lambda params, size: (
                params.get("pinned") if params else None
            ),
        )
        stub("omni_npu.envs")
        cls.modules["omni_npu"].envs = cls.envs
        core = stub("omni_npu.vllm_patches.core")
        cls.modules["omni_npu.vllm_patches"].PatchManager = SimpleNamespace(
            register=lambda *_args: None
        )
        core_path = (Path(__file__).resolve().parents[6] / "omni"
                     / "vllm_patches" / "core.py")
        stub("torch", int32="int32")
        stub("torch.distributed", ReduceOp=SimpleNamespace(MIN="min"))

        cls.module_patch = patch.dict(sys.modules, cls.modules)
        cls.module_patch.start()
        cls.addClassCleanup(cls.module_patch.stop)
        exec(compile(core_path.read_text(), str(core_path), "exec"), core.__dict__)
        source = (Path(__file__).resolve().parents[6] / "omni" / "vllm_patches"
                  / "patches" / "models" / "pangu_v2_benchmark")

        def load(name):
            spec = importlib.util.spec_from_file_location(name, source / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        cls.gate = load("patch_hybrid_aligned_decode")
        cls.routing = load("patch_dp_round_robin")
        cls.pd_gate = load("patch_bench_aligned_decode")
        cls.Scheduler = Scheduler
        cls.DPEngineCoreProc = DPEngineCoreProc
        cls.DPLBAsyncMPClient = DPLBAsyncMPClient
        cls.originals = {
            Scheduler: {"schedule": Scheduler.schedule},
            DPEngineCoreProc: {
                "_has_global_unfinished_reqs": DPEngineCoreProc._has_global_unfinished_reqs
            },
            DPLBAsyncMPClient: {
                "get_core_engine_for_request": DPLBAsyncMPClient.get_core_engine_for_request
            },
        }

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()

    def _reset_targets(self):
        for target, attrs in self.originals.items():
            for name, original in attrs.items():
                setattr(target, name, original)
            if "_omni_npu_applied_patches" in target.__dict__:
                delattr(target, "_omni_npu_applied_patches")

    def _apply_all(self):
        self.pd_gate.BenchAlignedDecodeEngineCorePatch.apply()
        self.routing.DPRoundRobinClientPatch.apply()
        self.gate.HybridAlignedDecodeSchedulerPatch.apply()
        self.gate.HybridAlignedDecodeEngineCorePatch.apply()

    def setUp(self):
        self._reset_targets()
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD", None)
        os.environ.pop("OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD", None)
        os.environ["OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD_FILE"] = ""
        self.envs.OMNI_HYBRID_ALIGNED_DECODE = True
        self.envs.OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD = 2
        self.envs.OMNI_DP_ROUND_ROBIN = True

    @staticmethod
    def request(computed, prompt):
        return SimpleNamespace(num_computed_tokens=computed, num_prompt_tokens=prompt)

    def test_gate_holds_decode_and_preserves_scheduler_arguments_and_order(self):
        self.gate.HybridAlignedDecodeSchedulerPatch.apply()
        ready = self.request(8, 8)
        prefill = self.request(3, 8)
        admitted = self.request(0, 8)
        scheduler = self.Scheduler()
        scheduler.running = [ready, prefill]
        scheduler.admitted = admitted

        seen = scheduler.schedule(throttle_prefills=True)

        self.assertEqual(seen, ([prefill], True))
        self.assertEqual(scheduler.running, [ready, prefill, admitted])
        self.assertEqual(scheduler.seen_capacity, 3)
        self.assertEqual(scheduler.max_num_running_reqs, 4)

    def test_gate_restores_requests_when_scheduler_raises(self):
        self.gate.HybridAlignedDecodeSchedulerPatch.apply()
        ready = self.request(8, 8)
        scheduler = self.Scheduler()
        scheduler.running = [ready]
        scheduler.raise_on_schedule = True

        with self.assertRaisesRegex(RuntimeError, "schedule failed"):
            scheduler.schedule()
        self.assertEqual(scheduler.running, [ready])
        self.assertEqual(scheduler.seen_capacity, 3)
        self.assertEqual(scheduler.max_num_running_reqs, 4)

    def test_cross_rank_release_is_one_shot_and_preserves_upstream_sync(self):
        scheduler = self.Scheduler()
        scheduler.running = [self.request(8, 8), self.request(8, 8)]
        engine = self.DPEngineCoreProc()
        engine.scheduler = scheduler
        engine.dp_group = object()
        engine.original_calls = 0
        sync = self.gate.HybridAlignedDecodeEngineCorePatch._has_global_unfinished_reqs

        with patch.object(self.gate, "_all_ranks_ready", side_effect=[False, True]) as barrier:
            self.assertTrue(sync(engine, True))
            self.assertFalse(getattr(scheduler, "hybrid_aligned_decode_released", False))
            self.assertTrue(sync(engine, True))
            self.assertTrue(scheduler.hybrid_aligned_decode_released)
            self.assertFalse(sync(engine, False))
        self.assertEqual(barrier.call_count, 2)
        self.assertEqual(engine.original_calls, 3)

    def test_round_robin_and_upstream_affinity(self):
        client = self.DPLBAsyncMPClient()
        client.core_engines = ["dp0", "dp1", "dp2"]
        client.eng_start_index = 1
        client.lb_engines = [[0, 0] for _ in range(3)]
        client.client_count = 1
        client.reqs_in_flight = {}
        client.original_calls = 0
        route = self.routing.DPRoundRobinClientPatch.get_core_engine_for_request

        def req(index, rank=None, pooling=None):
            return SimpleNamespace(request_id=str(index), data_parallel_rank=rank,
                                   pooling_params=pooling)

        self.assertEqual([route(client, req(i)) for i in range(4)],
                         ["dp1", "dp2", "dp0", "dp1"])
        self.assertEqual([row[0] for row in client.lb_engines], [1, 2, 1])
        self.assertEqual(client.reqs_in_flight["3"], "dp1")
        self.assertEqual(route(client, req(4, rank=2)), "upstream")
        self.assertEqual(route(client, req(5, pooling={"pinned": 0})), "upstream")
        self.envs.OMNI_DP_ROUND_ROBIN = False
        self.assertEqual(route(client, req(6)), "upstream")
        self.assertEqual(client.original_calls, 3)


    def test_unset_switches_leave_all_target_methods_unchanged(self):
        self.envs.OMNI_HYBRID_ALIGNED_DECODE = False
        self.envs.OMNI_DP_ROUND_ROBIN = False
        self._apply_all()
        for target, attrs in self.originals.items():
            for name, original in attrs.items():
                self.assertIs(getattr(target, name), original)
            self.assertNotIn("_omni_npu_applied_patches", target.__dict__)

    def test_hybrid_applies_both_halves_without_pd_gate(self):
        self.envs.OMNI_DP_ROUND_ROBIN = False
        self._apply_all()
        self.assertIs(
            self.DPEngineCoreProc._has_global_unfinished_reqs,
            self.gate.HybridAlignedDecodeEngineCorePatch._has_global_unfinished_reqs,
        )
        self.assertIsNot(self.Scheduler.schedule, self.originals[self.Scheduler]["schedule"])
        with self.assertRaisesRegex(ValueError, "already"):
            self.gate.HybridAlignedDecodeEngineCorePatch.apply()

    def test_explicit_pd_threshold_or_legacy_alias_selects_pd_only(self):
        self.envs.OMNI_HYBRID_ALIGNED_DECODE = False
        self.envs.OMNI_DP_ROUND_ROBIN = False
        for name in ("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD",
                     "OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD"):
            for value in ("0", "2"):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                    self._reset_targets()
                    self._apply_all()
                    self.assertIs(
                        self.DPEngineCoreProc._has_global_unfinished_reqs,
                        self.pd_gate.BenchAlignedDecodeEngineCorePatch._has_global_unfinished_reqs,
                    )
                    self.assertIs(self.Scheduler.schedule, self.originals[self.Scheduler]["schedule"])

    def test_conflicting_switches_do_not_modify_either_target(self):
        for name in ("OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD",
                     "OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "2"}):
                for patch_class in (
                    self.pd_gate.BenchAlignedDecodeEngineCorePatch,
                    self.gate.HybridAlignedDecodeSchedulerPatch,
                    self.gate.HybridAlignedDecodeEngineCorePatch,
                ):
                    with self.assertRaisesRegex(ValueError, "cannot be enabled together"):
                        patch_class.apply()
                self.assertIs(self.Scheduler.schedule, self.originals[self.Scheduler]["schedule"])
                self.assertIs(
                    self.DPEngineCoreProc._has_global_unfinished_reqs,
                    self.originals[self.DPEngineCoreProc]["_has_global_unfinished_reqs"],
                )

    def test_round_robin_can_be_applied_independently(self):
        self.envs.OMNI_HYBRID_ALIGNED_DECODE = False
        self._apply_all()
        self.assertIs(
            self.DPLBAsyncMPClient.get_core_engine_for_request,
            self.routing.DPRoundRobinClientPatch.get_core_engine_for_request,
        )
        self.assertIs(self.Scheduler.schedule, self.originals[self.Scheduler]["schedule"])

    def test_pd_gate_holds_remote_kv_and_releases_once(self):
        self.envs.OMNI_HYBRID_ALIGNED_DECODE = False
        os.environ["OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD"] = "2"
        self._apply_all()
        scheduler = self.Scheduler()
        scheduler.finished_recving_kv_req_ids = {"first"}
        engine = self.DPEngineCoreProc()
        engine.scheduler = scheduler
        engine.dp_group = object()
        engine.original_calls = 0
        with patch.object(self.pd_gate, "_all_ranks_ready", side_effect=[False, True]) as barrier:
            self.assertTrue(engine._has_global_unfinished_reqs(True))
            self.assertEqual(scheduler.finished_recving_kv_req_ids, set())
            scheduler.finished_recving_kv_req_ids.add("second")
            self.assertTrue(engine._has_global_unfinished_reqs(True))
            self.assertEqual(scheduler.finished_recving_kv_req_ids, {"first", "second"})
            self.assertTrue(scheduler.bench_aligned_decode_released)
            self.assertFalse(engine._has_global_unfinished_reqs(False))
        self.assertEqual([call.args[1] for call in barrier.call_args_list], [False, True])
        self.assertEqual(engine.original_calls, 3)


if __name__ == "__main__":
    unittest.main()
