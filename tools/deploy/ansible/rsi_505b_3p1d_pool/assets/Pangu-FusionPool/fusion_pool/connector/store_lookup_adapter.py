# SPDX-License-Identifier: Apache-2.0
"""Translate Ascend's exists() call onto the installed Mooncake store."""

import time


class AscendLookupStore:
    """batch_is_exist plus the worker's existing lookup statistics."""

    def __init__(self, worker):
        self.worker = worker

    def exists(self, keys):
        start = time.perf_counter()
        try:
            result = self.worker.store.batch_is_exist(keys)
            if len(result) != len(keys):
                raise ValueError("Mooncake lookup returned an incomplete result")
        except Exception:
            self.worker._record_kv_connector_operation(
                "lookup_exists", time.perf_counter() - start, len(keys),
                status="error", num_failed_keys=len(keys),
            )
            raise
        self.worker._record_kv_connector_operation(
            "lookup_exists", time.perf_counter() - start, len(keys),
        )
        return result


class AscendLookupInterfaces:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ascend_lookup_store = AscendLookupStore(self)

    @property
    def m_store(self):
        return self._ascend_lookup_store
