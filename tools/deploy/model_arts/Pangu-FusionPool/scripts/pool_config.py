# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Create the pool config used by the 1019 _fusion_pool_runtime.sh.j2."""
import json
import os
import socket
import sys
import time
from pathlib import Path


def build_transfer(role, rank, parallel_size, port, store_events=False, load_events=False):
    kv_role = "kv_producer" if role == "prefill" else "kv_consumer"
    primary = dict(kv_connector="LLMDataDistConnector", kv_role=kv_role,
                   kv_rank=rank, kv_parallel_size=parallel_size, kv_port=port)
    extra = ({"enable_store_events": store_events, "enable_load_events": load_events}
             if role == "prefill" else {"enable_lookup": False, "enable_load_events": False})
    pool = dict(kv_connector="NPUMooncakeStoreConnector",
                kv_role="kv_both" if role == "prefill" else "kv_consumer",
                kv_connector_extra_config=extra)
    return dict(primary, kv_connector="MultiConnector",
                kv_connector_module_path="fusion_pool.connector.pool_event_multi_connector",
                kv_connector_extra_config={"connectors": [primary, pool]})


def main():
    role = sys.argv[1]
    env = os.environ
    if "--dry-run" in sys.argv:
        print(json.dumps(build_transfer(role, int(env["prefill_instance_rank"]), 5,
                         int(env.get("OMNI_LLMDATADIST_ZMQ_PORT", "5568")),
                         env.get("FUSION_POOL_ENABLE_STORE_EVENTS", "false") == "true",
                         env.get("FUSION_POOL_ENABLE_LOAD_EVENTS", "false") == "true"), separators=(",", ":")))
        return
    master = env["MOONCAKE_MASTER_IP"]
    metadata = int(env.get("MOONCAKE_METADATA_PORT", "8081"))
    rpc = int(env.get("MOONCAKE_RPC_PORT", "16752"))
    segment = int(env.get("FUSION_POOL_GLOBAL_SEGMENT_SIZE_GB", "64"))
    buffer = int(env.get("FUSION_POOL_LOCAL_BUFFER_SIZE_GB", "8"))
    if segment <= 0 or buffer < 0 or not all(0 < port < 65536 for port in (metadata, rpc)):
        raise ValueError("invalid pool size or Mooncake port")
    config = dict(metadata_server=f"http://{master}:{metadata}/metadata",
                  master_server_address=f"{master}:{rpc}", protocol="ascend", device_name="",
                  global_segment_size=f"{segment}GB", local_buffer_size=f"{buffer}GB",
                  tenant_id=env.get("FUSION_POOL_TENANT_ID", "default"))
    Path(env["MOONCAKE_CONFIG_PATH"]).write_text(json.dumps(config), encoding="utf-8")
    deadline = time.monotonic() + int(env.get("FUSION_POOL_MASTER_WAIT_SECONDS", "600"))
    while True:
        try:
            for port in (metadata, rpc):
                with socket.create_connection((master, port), timeout=2):
                    pass
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Mooncake master not ready: {master}:{metadata}/{rpc}")
            time.sleep(2)
    transfer = build_transfer(role, int(env["prefill_instance_rank"]), 5,
                              int(env.get("OMNI_LLMDATADIST_ZMQ_PORT", "5568")),
                              env.get("FUSION_POOL_ENABLE_STORE_EVENTS", "false") == "true",
                              env.get("FUSION_POOL_ENABLE_LOAD_EVENTS", "false") == "true")
    print(json.dumps(transfer, separators=(",", ":")))


if __name__ == "__main__":
    main()
