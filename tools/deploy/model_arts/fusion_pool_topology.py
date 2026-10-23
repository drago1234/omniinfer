# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Resolve FusionPool topology: single-node P instances and one D group."""
import argparse
import ipaddress
import json
import os
import shlex
import sys
import time


def ordered_groups(data, env):
    groups = data["server_group_list"]
    ids = [str(group["group_id"]) for group in groups]
    if not groups or len(set(ids)) != len(ids):
        raise ValueError("empty groups or duplicate group_id")
    selected = env.get("DECODE_GROUP_IDS", "").strip()
    if selected:
        if selected not in ids:
            raise ValueError("DECODE_GROUP_IDS must name exactly one existing D group")
        groups = sorted(groups, key=lambda group: str(group["group_id"]) == selected)
    result = [[server["server_ip"] for server in group["server_list"]] for group in groups]
    flat = [ip for group in result for ip in group]
    if any(not group for group in result) or len(set(flat)) != len(flat):
        raise ValueError("empty group or duplicate server_ip")
    if any(ipaddress.ip_address(ip).version != 4 for ip in flat):
        raise ValueError("IPv4 server_ip required")
    return result


def topology(groups, env, role):
    p_count = int(env.get("PREFILL_INSTANCE_NUM", "4"))
    if p_count <= 0 or int(env.get("DECODE_INSTANCE_NUM", "1")) != 1:
        raise ValueError("FusionPool requires positive P instances and exactly one D group")
    if (len(groups) != p_count + 1 or any(len(g) != 1 for g in groups[:-1])
            or not groups[-1]):
        raise ValueError("expected single-node P groups followed by one nonempty D group")
    npu = int(env.get("npu", "16"))
    if npu != 16:
        raise ValueError("this inventory requires 16 logical NPU devices per machine")
    p_ips = [group[0] for group in groups[:-1]]
    d_ips = groups[-1]
    ip = env.get("POD_IP", "")
    if ip not in p_ips + d_ips:
        raise ValueError("POD_IP not found in rank table")
    base = int(env.get("base_api_port", "9000"))
    master_base = int(env.get("GLOBAL_PORT_BASE", "8000"))
    result = {
        "all_ip_list": ";".join(",".join(group) for group in groups),
        "PREFILL_API_SERVER_LIST": ",".join(f"{ip}:{base + i}" for i, ip in enumerate(p_ips)),
        "DECODE_API_SERVER_LIST": ",".join(f"{ip}:{base + 100 + i + j}" for i, ip in enumerate(d_ips) for j in range(npu)),
        "P_NODE_LIST": ";".join(p_ips),
        "LOCAL_DECODE_SERVER_IP_LIST": ",".join(d_ips),
        "GLOBAL_DECODE_SERVER_IP_LIST": ",".join(d_ips),
        "PREFILL_POD_NUM": str(p_count), "DECODE_POD_NUM": "1",
        "OMNI_PD_PREFILL_POD_NUM": str(p_count), "OMNI_PD_DECODE_POD_NUM": "1",
        "MOONCAKE_MASTER_IP": p_ips[0] if env.get("MOONCAKE_MASTER_IP", "auto") == "auto" else env["MOONCAKE_MASTER_IP"],
        "cur_pod_ip": ip,
        "FUSION_POOL_PARALLEL_SIZE": str(p_count + 1),
    }
    if role == "proxy":
        if ip != p_ips[0]:
            raise ValueError("Proxy/Master must run on first P group node (p0)")
        return result
    nodes = p_ips if role == "prefill" else d_ips
    if ip not in nodes:
        raise ValueError(f"POD_IP does not belong to {role}")
    rank = nodes.index(ip)
    result.update({
        "MASTER_IP": ip if role == "prefill" else d_ips[0],
        "HOST_IP": ip if role == "prefill" else d_ips[0],
        "MASTER_PORT": str(master_base + rank if role == "prefill" else master_base + 100),
        "API_PORT": str(base + rank if role == "prefill" else base + 100 + rank),
        "NODE_RANK": str(0 if role == "prefill" else rank),
        "prefill_instance_rank": str(rank if role == "prefill" else p_count),
        "SERVER_OFFSET": str(0 if role == "prefill" else rank * npu),
        "VLLM_DP_SIZE": str(1 if role == "prefill" else npu * len(d_ips)),
        "num_servers": str(1 if role == "prefill" else npu),
        "tp": str(npu if role == "prefill" else 1),
    })
    if int(env.get("tp", result["tp"])) != int(result["tp"]):
        raise ValueError("tp conflicts with inventory (P=16, D=1)")
    if int(env.get("dp", result["VLLM_DP_SIZE"])) != int(result["VLLM_DP_SIZE"]):
        raise ValueError("dp conflicts with resolved topology")
    return result


def read_rank_table(path, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
            if data.get("status") == "completed":
                return data
        except (OSError, json.JSONDecodeError):
            pass
        if time.monotonic() >= deadline:
            raise ValueError(f"rank table not ready after {timeout}s: {path}")
        time.sleep(min(2, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all"], default="all")
    parser.add_argument("--role", choices=["prefill", "decode", "proxy"])
    args = parser.parse_args()
    try:
        path = os.environ.get("GLOBAL_RANK_TABLE_FILE_PATH")
        if not path:
            raise ValueError("GLOBAL_RANK_TABLE_FILE_PATH is required")
        groups = ordered_groups(read_rank_table(path, int(os.getenv("RANK_TABLE_WAIT_SECONDS", "600"))), os.environ)
        if args.role:
            for key, value in topology(groups, os.environ, args.role).items():
                print(f"export {key}={shlex.quote(value)}")
        else:
            print(";".join(",".join(group) for group in groups))
        return 0
    except (ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
