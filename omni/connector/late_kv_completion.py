# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-only completion polling after an already-required dummy batch.

Used only by the TP1/PP1 D control-batch path. Does not submit a new
transfer, launch a model, synchronize a device or alter speculative depth.
"""
from copy import copy


def poll_llmdatadist_completion(vllm_config):
    """Return LLMD completion + errors together; leave all other children alone."""
    kv = vllm_config.kv_transfer_config
    if kv is None or kv.kv_role != 'kv_consumer':
        return None
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        return None
    from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
    from omni_npu.connector.llmdatadist_connector_v1 import LLMDataDistConnector
    if not has_kv_transfer_group():
        return None

    def leaves(connector):
        if isinstance(connector, LLMDataDistConnector):
            yield connector
        else:
            for child in getattr(connector, '_connectors', ()):
                yield from leaves(child)

    connectors = list(leaves(get_kv_transfer_group()))
    if not connectors:
        return None
    if len(connectors) != 1:
        raise RuntimeError('Late LLMD completion polling requires one LLMD child')
    connector = connectors[0]
    # The normal get_finished() requires currently bound step metadata, which
    # the preceding control batch has already cleared. Use the explicit D-only
    # completion API; never rebind stale metadata or resubmit a transfer.
    receiving, invalid = connector.poll_decode_completions()
    if not receiving and not invalid:
        return None
    return dict(finished_sending=None, finished_recving=receiving,
                invalid_block_ids=invalid)


def merge_llmdatadist_reply(model_output, reply):
    """Merge before the normal scheduler update, including load-error recovery.

    Never mutate EMPTY_MODEL_RUNNER_OUTPUT or the original connector result:
    empty control batches can use process-wide shared singleton objects.
    """
    if isinstance(reply, (list, tuple)):
        if len(reply) != 1:
            raise RuntimeError('Late LLMD poll expected exactly one TP1 worker reply')
        reply = reply[0]
    if reply is None:
        return model_output
    allowed = {'finished_sending', 'finished_recving', 'invalid_block_ids'}
    if not isinstance(reply, dict) or set(reply) != allowed:
        raise RuntimeError('Unexpected late LLMD completion reply')
    if not any(reply.values()):
        return model_output
    from vllm.v1.outputs import KVConnectorOutput
    old = model_output.kv_connector_output
    kv = copy(old) if old is not None else KVConnectorOutput()
    for name in allowed:
        combined = set(getattr(kv, name) or ()) | set(reply[name] or ())
        setattr(kv, name, combined if name == 'invalid_block_ids' else combined or None)
    output = copy(model_output)
    output.kv_connector_output = kv
    return output
