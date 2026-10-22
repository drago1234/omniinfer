# Pangu UltraMoE accuracy alignment

This package migrates the complete Pangu UltraMoE training/inference
alignment workflow to the vLLM 0.25.1 runtime. Its physical source directory
is `omni/diagnostics/accuracy_alignment`, exposed through the normal
`omni_npu.diagnostics.accuracy_alignment` Python namespace. Both the
implementation and its thin `NPUModelRunner` patch live in this directory. The normal
`apply_patches()` entry point imports `patch_accuracy_alignment.py` only when
`OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True`; no accuracy-alignment module or
Patch class is loaded during ordinary inference.

## Supported scope

The selected local decoder layer is covered at the same seven boundaries as
the legacy tool:

1. model level: token embedding, rotary cos/sin, and LM head;
2. decoder/MHC boundaries, including models configured with MHC fusion;
3. MLA/DSA attention projections and MoME convolution outputs;
4. DSA lightning-indexer query, merge weight, and top-k indices;
5. dense MLP, routed MoE, and shared-expert layers;
6. MoE gate output and router top-k scores/indices;
7. `apply_experts`, including optional recomputation from training-side
   permuted inputs, expert counts, and weights.

Only `PanguUltraMoE` and eager execution are supported. Hooks are installed
only for modules owned by the current PP rank, are removed on model reload,
and are bypassed during dummy/profile runs and graph capture.

In `train_ref`, model-level MHC fusion is temporarily disabled for the debug
session. The fused kernel combines a layer's `mhc_post`, the next layer's
`mhc_pre`, and normalization, so it has no independent boundaries matching
the imported legacy `pre_out`/`post_out` tensors. Falling back to the model's
own non-fused 0.25.1 forward preserves those old semantic points; the original
fusion flag is restored when hooks are removed. `real` and `pure_real` retain
the configured fused production path.

The same rule applies to DSA decode's fused MLA prolog. Its
`npu_mla_prolog_v3` call combines q_a/q_b/kv_a projections and normalization,
so the imported per-module Megatron boundaries do not execute. `train_ref`
temporarily sets `use_mlaprolog=False` on selected local attention layers and
restores it when the session closes. `real` keeps the production fused path;
the fused operator can be observed as part of the attention result, but its
internal projection boundaries cannot be dumped independently.

## Three runtime modes

| Mode | Input behavior | Output purpose |
|---|---|---|
| `train_ref` | every supported boundary uses its training input by default; an explicit `OMNI_NPU_DEBUG_MOCK_MODULES` limits injection to matching boundaries | isolate one operator/module from cascade error |
| `real` | only `OMNI_NPU_DEBUG_MOCK_MODULES` matches use training input | inspect real cascade or align a selected prefix |
| `pure_real` | never reads or injects training data | compare two independent inference runs |

The explicit 0.25.1 interface is `OMNI_NPU_DEBUG_MODE`. The legacy interface
is also preserved: with the master switch enabled, an **unset**
`OMNI_NPU_DEBUG_MOCK_MODULES` means `train_ref`; a present value (including an
empty string) means `real`; `OMNI_NPU_DEBUG_PURE_REAL=1` means `pure_real`.

`real` and an explicitly filtered `train_ref` accept exact tags and trailing
`.*` prefixes, for example:

```bash
export OMNI_NPU_DEBUG_MODE=real
export OMNI_NPU_DEBUG_MOCK_MODULES='self_attention.*,mlp.router,lm_head'
```

In `real`, an empty whitelist runs a completely real cascade while retaining
the semantic dump points. In explicit `train_ref` mode, an unset whitelist
preserves the original all-boundary behavior; a present whitelist prevents
unselected reference files from being loaded. The same old tags are accepted
for MHC raw boundaries, DSA indexer inputs, MoE gate output and
`gmm_experts_output`.

## Required configuration

```bash
export OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True
export OMNI_NPU_DEBUG_MODE=train_ref
export OMNI_NPU_DEBUG_TRAIN_DUMP_DIR=/path/to/megatron_dump
export OMNI_NPU_DEBUG_INFER_DUMP_DIR=/path/to/infer_dump
export OMNI_NPU_DEBUG_LAYER_ID=0
export OMNI_NPU_MHC_DEBUG_ROLE=P        # P or D
export OMNI_NPU_DEBUG_MOCK_SHAPE=110    # full prompt position
```

`OMNI_NPU_DEBUG_LAYER_ID` keeps the first-stage single-layer behavior. For a
complete run, use `OMNI_NPU_DEBUG_LAYER_IDS=all`, a comma-separated list, or
inclusive ranges such as `0,3-5,12`. Each PP worker installs hooks only for
the selected layers it actually owns.

The legacy names below are accepted as aliases:

- `OMNI_NPU_MHC_DEBUG_TRAIN_DUMP_DIR`
- `OMNI_NPU_MHC_DEBUG_INFER_DUMP_DIR`
- `OMNI_NPU_MHC_DEBUG_MOCK_SHAPE`

The old model-source-only switches `OMNI_NPU_MHC_DEBUG_INLINE`,
`OMNI_NPU_ATTN_DEBUG_INLINE`, and `OMNI_NPU_MOE_DEBUG_INLINE` are intentionally
not required: the 0.25.1 implementation installs instance-local observers and
method wrappers after model loading instead of modifying model source files.

Optional training-layout settings:

```bash
export OMNI_NPU_MODEL_LEVEL_TRAIN_PREFIX=iter0_module.module
export OMNI_NPU_MHC_DEBUG_TRAIN_NAME_PREFIX=iter0_module.module.decoder.layers
export OMNI_NPU_TRAIN_TP_SIZE=1
export OMNI_NPU_TRAIN_EP_SIZE=4
```

`OMNI_NPU_TRAIN_EP_SIZE` is the preferred training EP size. If that complete
rank range is not present, contiguous expert ranks are auto-detected from
rank 0, matching the legacy tool.

## Automatic dump from Ansible `server.yml`

Add the accuracy-alignment environment variables to the shell command in
`run_vllm_server_prefill_cmd` for P and `run_vllm_server_decode_cmd` for D.
The following deployment variables keep paths and request length outside the
generated command:

| Variable | Meaning | Example |
|---|---|---|
| `MOCK_SHAPE` | token count after applying the serving chat template; it must match the request actually sent | `110` |
| `TRAIN_DUMP_DIR` | Megatron/training dump root | `/log_path/train_dump/module_hooks/tensors` |
| `PREFILL_DIR_LOCAL` | P-side inference dump root | `/log_path/prefill_dump` |
| `DECODE_DIR_LOCAL` | D-side inference dump root when P and D use different storage | `/log_path/decode_dump` |

Append this block to the P-side command:

```bash
export OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True
export OMNI_NPU_DEBUG_MODE=train_ref
export OMNI_NPU_DEBUG_MOCK_SHAPE="${MOCK_SHAPE}"
export OMNI_NPU_DEBUG_TRAIN_DUMP_DIR="${TRAIN_DUMP_DIR}"
export OMNI_NPU_DEBUG_INFER_DUMP_DIR="${PREFILL_DIR_LOCAL}"
export OMNI_NPU_MHC_DEBUG_ROLE=P
```

Append this block to the D-side command:

```bash
export OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True
export OMNI_NPU_DEBUG_MODE=train_ref
export OMNI_NPU_DEBUG_MOCK_SHAPE="${MOCK_SHAPE}"
export OMNI_NPU_DEBUG_TRAIN_DUMP_DIR="${TRAIN_DUMP_DIR}"
export OMNI_NPU_DEBUG_INFER_DUMP_DIR="${DECODE_DIR_LOCAL:-${PREFILL_DIR_LOCAL}}"
export OMNI_NPU_MHC_DEBUG_ROLE=D
```

For example:

```bash
export OMNI_NPU_ACCURACY_ALIGNMENT_ENABLE=True
export OMNI_NPU_DEBUG_MODE=train_ref
export OMNI_NPU_DEBUG_MOCK_SHAPE=110
export OMNI_NPU_DEBUG_TRAIN_DUMP_DIR=/mnt/train_dump/module_hooks/tensors
export OMNI_NPU_DEBUG_INFER_DUMP_DIR=/data/accuracy_alignment/prefill
export OMNI_NPU_MHC_DEBUG_ROLE=P
```

Use the same blocks for `real` and `pure_real`, changing only
`OMNI_NPU_DEBUG_MODE`. `real` also uses `OMNI_NPU_DEBUG_MOCK_MODULES` to select
the boundaries that receive training inputs. `pure_real` does not require a
training dump and uses `OMNI_NPU_DEBUG_RUN_ID` to distinguish two inference
runs.

When `OMNI_VLLM_PATCHES` explicitly selects patch names, include
`AccuracyAlignmentTrainRefPatch`. The name is retained for compatibility even
though the patch now serves all three modes. Normally no explicit patch-list
entry is needed: the master switch conditionally imports it after the common
and model-specific patch directories have registered.

## Megatron dump format and filenames

Imported Megatron training dumps retain the old payload layout:

```python
{"tensors": {"input.0": tensor, "output.0": tensor}}
```

Inference-side alignment dumps remain raw tensors, matching the legacy debug
tool's output files. The comparator accepts both raw tensors and Megatron
payload dictionaries.

Model-level files use:

```text
iter0_module.module.embedding.word_embeddings_forward_rank0_0.pt
iter0_module.module.rotary_pos_emb_forward_rank0_0.pt
iter0_module.module.output_layer_forward_rank0_0.pt
```

The embedding and LM-head module names can be overridden with
`OMNI_NPU_EMBEDDING_TRAIN_NAME` and `OMNI_NPU_LM_HEAD_TRAIN_NAME`.

The whole decoder-layer file is:

```text
iter0_module.module.decoder.layers.{layer_id}_forward_rank0_0.pt
```

Fine-grained layer files use:

```text
{layer_prefix}.{layer_id}.{module_name}_forward_rank{train_tp_rank}_0.pt
```

Relevant `module_name` values include:

- `input_layernorm`, `self_attention`, `post_self_attn_layernorm`;
- `pre_mlp_layernorm`, `mlp`, `post_mlp_layernorm`,
  `block_post_layernorm`;
- `self_attention.linear_qkv`, `self_attention.q_layernorm`,
  `self_attention.linear_qb`, `self_attention.linear_kvb`,
  `self_attention.linear_proj`;
- `self_attention.qa_conv`, `self_attention.compresskv_conv`;
- `self_attention.index_linear_qb`, `self_attention.index_linear_k`,
  `self_attention.index_k_layernorm`,
  `self_attention.linear_merge_weight`;
- `mlp.linear_fc1`, `mlp.linear_fc2`, `mlp.router`, `mlp.experts`,
  `mlp.experts.shared_expert.linear_fc1`, and
  `mlp.experts.shared_expert.linear_fc2`.

Row-parallel inputs prefer the matching training TP-rank file. If training
and inference TP differ, all available training shards are concatenated on
the hidden dimension and re-sliced for the inference rank. Legacy NCL
convolution input dumps are converted to token-major layout before P/D
slicing.

When 0.25.1 enables `split_q_up_in_multistream`, the released `q_b_proj`
module is not used at runtime. The tool hooks `q_b_nope_proj` and
`q_b_pe_proj` instead, injects the same legacy `linear_qb` `input.0` into
both, and reconstructs the old per-head `linear_qb` output layout before
dumping. This keeps the imported training filename and comparator unchanged.

On Ascend 950 with sequence parallelism, 0.25.1 can execute MoME through the
function-style `conv_sp` fast path, bypassing `MomeAttention` and its child
module hooks. The tool observes the corresponding tensors at the next stable
semantic boundaries (q-layernorm, KV norm/cache, and o-projection), preserving
the same `qa_conv`, `compresskv_conv`, and `pangu_transformer_o_conv` dump
names without changing the model forward.

## Custom raw references

Set the optional legacy directory when MHC, DSA, gate, or expert-level raw
training tensors are available:

```bash
export OMNI_NPU_MHC_CUSTOM_DUMP_DIR=/path/to/custom_dump
```

The tool recognizes these old raw-tensor names (`N = layer_id + 1`):

```text
pangu_transformer_hc_{pre_in|post_in|post_h_post|post_residual|post_h_res}_N_rank0.pt
pangu_transformer_mlp_hc_{pre_in|post_in|post_h_post|post_residual|post_h_res}_N_rank0.pt
dsa_attention_index_query_all_heads_N_rank0.pt
dsa_attention_merge_weight_all_heads_N_rank0.pt
moe_gate_output_N_rank0.pt
moe_experts_up_permuted_N_rank{ep}.pt
moe_experts_up_tokens_per_expert_N_rank{ep}.pt
moe_experts_up_weight1_N_rank{ep}.pt
moe_experts_down_weight2_N_rank{ep}.pt
```

MHC, DSA, and expert reference sets are discovered independently. A component
that is entirely absent is skipped, while a partially present component is
rejected with an explicit missing-file error. Expert
weights are loaded without forcing a CPU round-trip so saved NPU formats such
as `FRACTAL_NZ` are preserved. Training-side `apply_experts` recomputation
matches the legacy tool and is limited to the unquantized NPU MoE method;
quantized methods can still dump their native `apply_experts` output when the
expert raw reference files are omitted.

MHC output additionally keeps the legacy semantic basenames
`pangu_transformer_{hc|mlp_hc}_{pre_out|post_out}_N` and
`pangu_transformer_mlp_N`; the normal stage/rank suffix below is appended to
those basenames.

## Runtime behavior and outputs

For prefill, the first `mock_shape` rows are used. For decode, row
`mock_shape - 1` is used. Only the target's primary activation is replaced;
residuals, cos/sin call arguments, metadata, caches, and fused-MHC auxiliary
fp32 router input remain runtime values unless they have their own explicit
target boundary.

The original vLLM 0.25.1 forward always executes. Hooks save observed
inference tensors but never replace the module's final return value. Output
files are written under `prefill/` or `decode/` as:

```text
{module_name}_{stage}_train_ref_rank{world_rank}.pt
```

Missing required Megatron files, incompatible sequence lengths, or shape
mismatches raise a clear error instead of silently falling back to runtime
input.

`real` replaces the `train_ref` filename tag with `real`. `pure_real` installs
output-only instance wrappers over every selected `named_modules()` entry and
writes:

```text
decorator_{qualname}_run{run_id}_call{call_index}_rank{world_rank}.pt
```

Wrapping each instance's `forward` (instead of relying on PyTorch post-hooks)
also captures the direct `.forward(...)` calls used by 0.25.1 projection and
MoME fast paths. The original bound method is restored when the session closes.

Use `OMNI_NPU_PURE_REAL_DUMP_INCLUDE` and
`OMNI_NPU_PURE_REAL_DUMP_EXCLUDE` for comma-separated regular expressions and
`OMNI_NPU_DEBUG_RUN_ID` to identify the run. Prefill writes only world rank 0,
matching the old tool; decode writes each participating rank.

## Comparator

The comparator directly reads the imported Megatron payload format and also
supports raw MHC/DSA/MoE tensors, training TP shard concatenation, legacy NCL
convolution layout, P/D sequence selection, rotary output.0/1, LM-head naming,
real-vs-train-ref gate/expert comparison, and pure-real call-by-call comparison.
It accepts both the new stage-tagged filenames and old tool filenames.

```bash
# P-side train_ref comparison
python -m omni_npu.diagnostics.accuracy_alignment.compare \
  --train-root /path/to/megatron_dump \
  --custom-train-root /path/to/custom_dump \
  --infer-root /path/to/prefill_dump \
  --role P --compare-mode train_ref --mock-seq-len 110 \
  --module-set all --report-prefix mhc_mock_vs_infer \
  --report-dir ./reports

# D-side train_ref comparison
python -m omni_npu.diagnostics.accuracy_alignment.compare \
  --train-root /path/to/megatron_dump \
  --custom-train-root /path/to/custom_dump \
  --infer-root /path/to/decode_dump \
  --role D --compare-mode train_ref --mock-seq-len 110 \
  --module-set all --report-prefix mhc_mock_vs_infer \
  --report-dir ./reports

# Compare two pure-real inference runs
python -m omni_npu.diagnostics.accuracy_alignment.compare \
  --infer-root /path/to/infer_dump \
  --role P --compare-mode pure_real --run-ids 0 1 \
  --mock-seq-len 110 --report-dir ./reports
```

Every run creates Markdown, CSV and JSON summaries. The report retains the
legacy fields `train_path`, `infer_path`, `seq_len`, `shapes_match`,
`all_close`, absolute/relative error statistics, correlation and error
distribution; `left_path`, `right_path`, `left_shape` and `right_shape` are
additional aliases/diagnostics. Relative-error statistics also retain the old
rule that reference values with `abs(value) <= atol` contribute zero. A
nonzero exit status means at least one `FAIL` or `ERROR`; missing optional
files are `SKIP`.

For `train_ref`, P compares the first `mock-seq-len` rows of the training
reference, while D selects the row at `mock-seq-len - 1`. Therefore the
comparison value must be identical to `OMNI_NPU_DEBUG_MOCK_SHAPE` used by the
service.

The summary reports `Total`, `PASS`, `FAIL`, `SKIP`, and `ERROR`. Use the
detailed Markdown table or the equivalent CSV/JSON fields as follows:

| Field | Meaning |
|---|---|
| `train_path` / `infer_path` | exact tensor files used in the comparison |
| `shapes_match` | whether reference and inference shapes agree; resolve shape or parallel-sharding issues first when false |
| `all_close` | final `torch.allclose` result for the configured `rtol` and `atol` |
| `max_abs_diff` / `mean_abs_diff` | maximum and mean absolute error |
| `max_rel_diff_percent` / `avg_rel_diff_percent` | relative error percentages; interpret large values together with absolute error because a near-zero denominator can amplify them |
| `corr_coef` | Pearson correlation; a high value alone does not make a case pass |
| `error_distribution` | relative-error bucket counts, useful for separating isolated spikes from broad drift |

`SKIP` normally means that an optional inference dump is absent. If it is
unexpected, verify that the request completed on both P and D and that
`OMNI_NPU_DEBUG_INFER_DUMP_DIR` points to the intended fresh run directory.

## One-shot workflow

`workflow.py` replaces the old environment-specific `run_test.sh` hardcoding.
It can run an optional Ansible playbook, wait for both P and D `/health`
endpoints, send the completion request, optionally rsync D-side dumps, and
invoke the comparator for both stages.

The dump roots and prompt length default to the same new/legacy environment
variables used by the runtime. To avoid silently mixing runs, the workflow
refuses pre-existing `.pt` files by default; use a fresh run directory or pass
`--keep-old` deliberately (required for the second `pure_real` run).

```bash
python -m omni_npu.diagnostics.accuracy_alignment.workflow \
  --health-url http://P_HOST:PORT/health \
  --health-url http://D_HOST:PORT/health \
  --request-url http://PROXY_HOST:PORT/v1/completions \
  --model /path/to/model --prompt-length 110 \
  --train-root /path/to/megatron_dump \
  --custom-train-root /path/to/custom_dump \
  --infer-root /path/to/infer_dump \
  --compare-mode train_ref --module-set all
```
