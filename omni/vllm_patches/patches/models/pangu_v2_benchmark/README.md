# Pangu V2 benchmark patches

Optional patches for vLLM 0.25.1. Load them after normal model patches:

```bash
export OMNI_VLLM_PATCHES_DIR="high_throughout,pangu_v2_benchmark"
```

The existing environment variables select which patches are applied at startup.
No additional mode variable is needed. If neither gate is enabled, neither the
scheduler nor engine-core methods are changed. Restart the service after
changing the selection.

## Hybrid serving: local prefill followed by decode

```bash
unset OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD
unset OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD
export OMNI_HYBRID_ALIGNED_DECODE=1
export OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD=12
export OMNI_DP_ROUND_ROBIN=1
```

`patch_hybrid_aligned_decode.py` applies both its scheduler wrapper and engine
patch only when `OMNI_HYBRID_ALIGNED_DECODE` is true (`1`, `true`, or `all`). It
holds requests that finished local prefill until every DP rank has reached the
per-rank decode-ready threshold. Setting the variable to `0` or `false` does
not apply either patch.

Use `--api-server-count 1` for exact internal round-robin distribution and send
enough simultaneous requests for every DP rank to reach the threshold. For DP8
and threshold 12, send at least 96 requests. The threshold must fit each rank's
request and KV-cache capacity. If a rank never reaches it, decode stays held.
Prefill continues while decode is held.

## PD-disaggregated decode: wait for remote KV

On the decode node:

```bash
unset OMNI_HYBRID_ALIGNED_DECODE
export OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD=12
```

`patch_bench_aligned_decode.py` applies only when the threshold variable is
explicitly set. Its legacy alias `OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD` also
selects the PD gate. The registry's default threshold of 20 does not activate
the patch by itself. An explicitly set threshold of `0` applies the patch but
releases the gate on its first readiness check.

This gate holds completed remote KV receptions until all DP ranks have reached
the threshold. It does not wrap the scheduler. Existing threshold-file tuning
is preserved: while the gate is closed, the integer in
`OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD_FILE` (default `/tmp/omni_gate_threshold`)
overrides the threshold value. A threshold file alone does not select the PD
patch; explicitly set the threshold environment variable first.

## Independent routing and mutual exclusion

`patch_dp_round_robin.py` is applied only when `OMNI_DP_ROUND_ROBIN` is true.
It distributes requests across the internal DP engines in rotation, preserving
explicit DP-rank selection and late-interaction pooling affinity. Proxy-side
routing bypasses this client and must distribute requests itself.

Hybrid and PD gates both modify the engine's unfinished-request synchronization
method. If hybrid is enabled while either PD threshold name is explicitly set,
each gate's `apply()` rejects the configuration before modifying its target.
The patch manager logs errors and neither gate is applied; it does not abort
service startup. Choose exactly one gate. Do not combine these gates with
`decode_profile_sync` or another aligned-decode patch modifying the same methods.

Both gates are one-shot: once released they remain open for the engine's
lifetime. They deliberately increase TTFT to coordinate benchmark batches.
