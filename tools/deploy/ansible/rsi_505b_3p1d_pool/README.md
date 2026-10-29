# 505B 3P1D Pool deployment candidate

This bundle preserves the Pool and P-side configuration. D uses concurrency 3, graph capture 24 and disables SuperKernel. The corresponding exploratory run produced TTFT 838.95 ms (10155 requests) and TPOT 12.57 ms (10154 defined values); one successful one-token response had no TPOT. This is a candidate, not a validated RSI best or a target-passing result. Revalidate hardware, output correctness, and formal performance on the new machines.

## Configuration

- P: TP16, max sequences 4, APC enabled, MTP7.
- D: DP16, max sequences 3, graph capture 24, MTP7, SuperKernel disabled.
- Pool: 4 GiB segment and 1 GiB buffer per rank on both P and D; huge pages disabled. D contributes capacity without lookup. Pool master defaults to the P0 host through a separate `M` inventory alias and container.
- The proxy settings remain those of the measured run.

Use with the matching RSI source snapshot and compatible image, CANN, Python, model weights and `mooncake_master`. The ansible working tree must include `roles/common`, and the synced source must contain `omniinfer/components/Pangu-FusionComm/Pangu-FusionPool` and `vllm/vllm`. The playbook syncs code to the specified remote workspace and replaces Python files in the task containers; select dedicated task paths and containers.

## Prepare

1. Copy `inventory.example.yml` and replace all documentation addresses (`192.0.2.x`) with the new P0, P1, P2 and D0 addresses. Set C and M to the intended proxy and pool-master hosts. Adapt device lists and ports if needed; changing topology requires revalidation. Provide SSH credentials separately, for example through SSH keys or Ansible Vault. Do not commit credentials.
2. Fill every `rsi_*` path and image in your inventory copy. `vars.example.yml` provides the same values as an alternative for direct `ansible-playbook -e @vars.yml`. `rsi_executor_code_path` and `rsi_remote_code_path` are workspace roots containing both `omniinfer` and `vllm`. Model and log paths must exist or be provisioned with appropriate access. The image must contain Mooncake and Ascend runtime dependencies.
3. Run from `omniinfer/tools/deploy/ansible` on the executor. The bundle uses its own Pool role and templates, and the repository's common role:

```bash
BUNDLE="$PWD/rsi_505b_3p1d_pool"
export ANSIBLE_ROLES_PATH="$BUNDLE/roles:$PWD/roles"
ansible-playbook -i /path/to/inventory.yml "$BUNDLE/server.yml" \
  -e @/path/to/vars.yml --syntax-check
ansible-playbook -i /path/to/inventory.yml "$BUNDLE/server.yml" \
  -e @/path/to/vars.yml --list-hosts
ansible-playbook -i /path/to/inventory.yml "$BUNDLE/server.yml" \
  -e @/path/to/vars.yml --list-tags
```

After resource and identity checks, an authorized deployment uses the same arguments with `--tags run_docker,sync_code,run_mooncake_master,run_server,run_proxy`. Include `run_mooncake_master`: the standard RSI start tags do not include it. For an RSI config based on this bundle, place the entire bundle on the executor under `ANSIBLE_CWD`, point `SERVER_TEMPLATE` to `rsi_505b_3p1d_pool/server.yml`, point `INVENTORY` to the filled inventory under that directory, and set `ANSIBLE_START_TAGS=run_docker,sync_code,run_mooncake_master,run_server,run_proxy`. The RSI deploy wrapper does not pass `vars.example.yml`, so put `rsi_*` values in the inventory. Set `BENCH_TOKENIZER` to the actual model path explicitly: `auto` cannot resolve the templated `model_path`. Do not change an active track silently. A complete deployment requires the supported matching common role and source snapshot.

The Master supervisor retains its container across regular redeploys. A Master restart loses old KV metadata; do not restart it during active requests. The inherited common stop/cleanup tags can affect services beyond this task, so inspect them and select only the intended tags before use. After deployment verify master ports 8081/8181/16752, all P/D APIs, runtime flags and pool connections before sending traffic.

## Evidence and limitations

The P and D pool profiles and the connector roles are preserved from the experiment. The bundle patches the copied install task to the source snapshot's `Pangu-FusionComm` layout and uses the measured corrected Master supervisor. It has not been deployed or benchmarked as a bundle. The prior raw data and launcher are under `state/claw_mtp7_restart_20261028_b/tasks/pool_preparation/`; no prior run is promoted by this bundle.
