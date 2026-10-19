# KV Dump 使用文档

本文档说明 `omni_npu.connector.kv_dump` 模块的使用方法。该模块在 vLLM NPU 推理过程中对 KV cache 做 MD5 哈希，有两大用途：

1. **KV cache 传输校验**：在 producer/consumer 模式下验证 KV cache 跨节点传输的正确性。
2. **KV cache 踩踏检测**：decode 每一步对比已记录块和未满块。不一致时打错误日志并刷新基线，不中断当前 step。

## 1. 功能概述

- **Prefill 阶段**：对所有新分配的 KV cache 块做哈希，记录完整快照。
- **Decode 阶段**：每步校验已计算块与历史是否一致，并对新请求记录初始哈希。块或 token 不一致时写 `KV HASH MISMATCH` 日志，然后用当前内容更新基线。
- **输出**：按 rank 分别输出 JSON 文件，key 为 `req_id`，value 为嵌套的 block hash 数据。

## 2. 启用方式

通过 `OMNI_KV_DUMP_PATH` 控制，兼容旧名 `KV_DUMP_PATH`。未设置或为空则不启用。

```bash
export OMNI_KV_DUMP_PATH=/path/to/output/
```

## 3. 代码接入

在 `GPUModelRunner.execute_model` 与 `sample_tokens` 两处包裹：

```python
from omni_npu.connector.kv_dump import maybe_dump_kv

@maybe_dump_kv("update")   # execute_model: 前段记账
def execute_model(self, scheduler_output, ...):
    ...

@maybe_dump_kv("finish")   # sample_tokens 返回后补 prefill 哈希
def sample_tokens(self, grammar_output):
    ...
```

`maybe_dump_kv(stage)` 是一个装饰器工厂：若 `OMNI_KV_DUMP_PATH`（兼容旧名 `KV_DUMP_PATH`）为空则旁路；否则 `stage="update"` 在 `execute_model` 前后记账或校验，`stage="finish"` 在 `sample_tokens` 之后调用 `Dump.finish()`，补做 prefill 的哈希。Decode 不校验本步 MTP spec token，见第 8 节。

## 4. 输出说明

- **目录结构**：`{OMNI_KV_DUMP_PATH}/{YYYYMMDD-HHMM}/`
- **文件命名**：`rank0.json`、`rank1.json` …（按 `get_world_group().rank`）
- **写入方式**：后台线程每秒轮询队列，以流式 JSON 追加写入，避免全量加载。

## 5. 生产者 / 消费者角色

模块根据 `kv_transfer_config.kv_role` 区分行为：

- **`kv_producer`（生产者）**：真正执行 dump，将 KV cache 哈希写入 JSON。
- **`kv_consumer`（消费者）**：不写 JSON，做两项校验：
  - `_check_blks`：已满块与上一轮 `hashed_batch` 对比。
  - `_check_toks`：未满块里已计算的 token 与 `hist_hash` 对比。

  不一致时记录差异字节数和首个偏移，并刷新基线。请求本步没被调度时保留跟踪状态；出现在 `finished_req_ids` 时删除。抢占恢复带的是完整 block 表，会替换旧表并重建基线。cached 列表里没有跟踪状态的请求跳过本步校验，同一 `req_id` 只打一次错误日志。

## 6. 关键日志

运行时可在日志中看到耗时信息：

```
[Xms] hash blocks
[Xms] check_blks ok: blocks=N
[Xms] check_toks ok: blocks=N
[Xms] save json: /path/to/rank0.json
!!! KV HASH MISMATCH :: DECODE BLOCK !!!
MOME CACHE POSITIONS source=...
MOME READ UNTRACKED source=...
kv dump missing cached request req_id=...
```

## 7. 性能影响

> ⚠️ **此工具会显著降低推理性能，仅限调试场景使用，不可常态部署。**

开启后性能劣化情况：

| 场景 | 性能劣化 |
|------|---------|
| KV cache 传输（producer dump） | 约 **3 倍** |
| Decode 逐 step 校验 | 约 **5 倍** |

劣化主要来源于：每步对全部活跃块的哈希计算、CPU/GPU 间数据搬运（`.contiguous().cpu()`）、以及后台 JSON 序列化写入。

## 8. MTP 与 MoME 位置检查

KV 哈希没有单独的 MTP 层校验。

- Prefill 在 `sample_tokens` 返回后的 `Dump.finish()` 取哈希。draft forward 已经结束，写进本次 block 表、且属于 runner `kv_cache_groups` 的 KV 会进入哈希。
- Decode 用 `tok_num - num_spec_tokens` 做对比。本步尚未提交的 spec token 不进入已满块和未满块校验，不能用来判断 MTP draft 刚写出的 KV。开启 MTP 时模块仍会跑，不会因此中断；这类 token 的不一致不会出现在 `KV HASH MISMATCH` 里。

`OMNI_KV_DUMP_PATH` 非空时，`maybe_note_mome_cache` 另做 MoME cache 行的 token 位置检查。主模型 `draft=False`；MTP draft 用 `draft_head` 和 `draft_step`。位置对不上打 `MOME READ` / `MOME READ UNTRACKED`，推理继续。检查抛异常时只记日志。

## 9. 其它注意事项

- 依赖 vLLM 内部 API（`GPUModelRunner`、`SchedulerOutput`、`extract_layer_index` 等），升级 vLLM 版本时需关注兼容性。
