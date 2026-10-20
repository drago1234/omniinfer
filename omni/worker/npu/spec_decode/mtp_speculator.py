# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Separate MTP compilation sizes from per-phase capture sizes."""

from copy import copy

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    DecodeSpeculatorCudaGraphManager,
    PrefillSpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


class NPUMTPSpeculator(MTPSpeculator):
    def __init__(self, vllm_config, device):
        self._dummy_request_inputs = False
        spec = vllm_config.speculative_config
        compilation = vllm_config.compilation_config
        decode_capture_sizes = compilation.cudagraph_capture_sizes
        n_predict = getattr(spec.draft_model_config.hf_config, "n_predict", 1)
        self.prefill_single_step = (
            n_predict == 1 and getattr(vllm_config.kv_transfer_config, "kv_role", None) == "kv_producer"
        )

        if n_predict == 1 and not self.prefill_single_step and decode_capture_sizes:
            # Single-head MTP: later steps use one token per request.
            query_len = 1 + spec.num_speculative_tokens
            decode_capture_sizes = sorted({(size + query_len - 1) // query_len for size in decode_capture_sizes})

            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(compilation)
            # Copy the sizes as well: never mutate the target configuration.
            compile_sizes = set(compilation.compile_sizes or [])
            compile_sizes.discard("cudagraph_capture_sizes")
            vllm_config.compilation_config.compile_sizes = sorted(
                compile_sizes | set(compilation.cudagraph_capture_sizes) | set(decode_capture_sizes)
            )
        super().__init__(vllm_config, device)

        self.decode_config = copy(self.vllm_config)
        self.decode_config.compilation_config = copy(self.vllm_config.compilation_config)
        self.decode_config.compilation_config.cudagraph_capture_sizes = decode_capture_sizes
        self.decode_config.compilation_config.max_cudagraph_capture_size = (
            max(decode_capture_sizes) if decode_capture_sizes else None
        )

    def _copy_request_inputs(self, num_reqs, idx_mapping, temperature, seeds):
        # Backport PR #56734 to the older propose signature: the upstream
        # caller does not yet pass dummy_run to this method. propose updates
        # the flag on every call; mark the persistent device indices before
        # any later draft (including graph replay) can consume them.
        super()._copy_request_inputs(num_reqs, idx_mapping, temperature, seeds)
        if self._dummy_request_inputs:
            self.idx_mapping.fill_(-1)
        else:
            self.idx_mapping[num_reqs:].fill_(-1)

    def propose(self, *args, dummy_run=False, **kwargs):
        self._dummy_request_inputs = dummy_run
        return super().propose(*args, dummy_run=dummy_run, **kwargs)

    def init_cudagraph_manager(self, cudagraph_mode):
        self.prefill_cudagraph_manager = PrefillSpeculatorCudaGraphManager(
            self.vllm_config, self.device, cudagraph_mode,
            self.num_speculative_steps + 1,
        )
        decode_mode = (CUDAGraphMode.FULL_DECODE_ONLY
                       if not self.prefill_single_step and cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                       else CUDAGraphMode.NONE)
        self.decode_cudagraph_manager = DecodeSpeculatorCudaGraphManager(
            self.decode_config, self.device, decode_mode, decode_query_len=1,
        )

    def _multi_step_decode(self, num_reqs, skip_attn, batch_desc, num_tokens_across_dp):
        if self.prefill_single_step:
            # P only needs draft0 KV. Preserve the runner's K-column contract;
            # the unused proposals are placeholders, not additional MTP steps.
            self.draft_tokens[:num_reqs, 1:] = self.draft_tokens[:num_reqs, :1]
            return
        super()._multi_step_decode(num_reqs, skip_attn, batch_desc, num_tokens_across_dp)
