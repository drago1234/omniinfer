# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Vendored from vllm-ascend (vllm_ascend/distributed/kv_transfer/utils/
# mooncake_transfer_engine.py). Owns the process-wide singleton TransferEngine
# used by every mooncake consumer (KV pool store connector + future P2P
# connectors) so that buffer registration and transport land in the same
# engine instance.

import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the "
                            "instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/"
                            "doc/en/build.md to run vLLM with "
                            "MooncakeStoreConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(
                        hostname, "P2PHANDSHAKE", "ascend", device_name
                    )
                    if ret_value != 0:
                        raise RuntimeError(
                            "TransferEngine initialization failed with "
                            f"ret_value: {ret_value}"
                        )
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            if self.transfer_engine is None:
                raise RuntimeError("Transfer engine must be initialized")
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True


global_te = GlobalTE()
