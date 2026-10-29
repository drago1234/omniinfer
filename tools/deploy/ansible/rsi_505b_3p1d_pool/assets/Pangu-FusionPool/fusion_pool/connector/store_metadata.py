# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Store jobs use connector metadata, never request send-completion votes."""

from dataclasses import dataclass, field
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeStoreConnectorMetadata,
)


class StoreMetadata(MooncakeStoreConnectorMetadata):
    def __init__(self, original, jobs, flush, retired, finished_contexts):
        super().__init__(set(original.unfinished_request_ids), set(original.preempted_req_ids))
        self.requests = original.requests
        self.store_jobs = jobs  # job ID -> (request metadata, save incarnation)
        self.jobs_to_flush = flush
        self.retired_jobs = retired
        self.finished_contexts = finished_contexts


@dataclass
class StoreWorkerMetadata(KVConnectorWorkerMetadata):
    outcomes: dict[str, dict[int, str | None]] = field(default_factory=dict)

    def aggregate(self, other):
        if not isinstance(other, StoreWorkerMetadata):
            raise TypeError("Incompatible Mooncake worker metadata")
        merged = {job: dict(ranks) for job, ranks in self.outcomes.items()}
        for job, ranks in other.outcomes.items():
            target = merged.setdefault(job, {})
            for rank, outcome in ranks.items():
                if rank in target and target[rank] != outcome:
                    raise RuntimeError(f"Conflicting store outcome: {job}/{rank}")
                target[rank] = outcome
        return StoreWorkerMetadata(merged)
