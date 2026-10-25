# MSProbe 精度定位指南

推理或训练过程中常出现精度异常——输出偏离预期、数值不稳定（NaN/Inf）、预测与标签不再匹配。要定位根因，需要在模型执行时捕获并留存中间数据：feature map、权重、激活、各层输出等。通过在特定阶段捕获关键张量、记录核心层的输入/输出对，并保留上下文元数据（prompt、张量 dtype、硬件配置等），就能系统地追溯精度劣化或数值错误的起点。本指南描述 omni-npu 服务下诊断精度问题的端到端流程：准备、数据采集、分析与验证。

底层工具为 MindStudio Probe（msprobe），omni-npu 复用 vllm-ascend 的集成方式。更多细节见 [Ascend/msprobe](https://gitcode.com/Ascend/msprobe)。

## 0. 背景概念

`msprobe` 支持三种精度级别：

- **L0**：在 module 层 dump 张量，并生成 `construct.json` 供可视化工具重建网络结构。需传入 model 或 submodule 句柄。
- **L1**：仅采集算子级统计信息，适合轻量排查。
- **mix**：同时采集结构信息与算子统计，适用于同时需要图重建与数值比对的场景。

## 1. 前置条件

### 1.1 安装 `msprobe`

通过 pip 安装：

```bash
pip install mindstudio-probe
```

### 1.2 图模式 dump（可选）

如需 dump cudagraph 图，需从源码安装：

1. 从源码安装 `aclgraph_dump`：

   ```bash
   git clone https://gitcode.com/Ascend/msprobe.git
   cd msprobe
   pip install uv
   python3 build.py -e include-mod=aclgraph_dump -e no-check=true
   pip install artifacts/mindstudio_probe*.whl
   ```

源码编译统一使用 MindStudio 标准构建环境，参考 [MindStudio 工具开发环境安装指导](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/dev_env_setup.md)

## 2. 使用 `msprobe` 采集数据

采集数据时通常遵循"由粗到细"的策略：先定位出现问题的 token，再决定在其周围采样哪些范围。典型流程如下。

### 2.1 准备 dump 配置内容

准备一份能被 `PrecisionDebugger` 解析的配置。omni-npu 支持两种传入方式（与 vllm-ascend 一致）：

- 通过 `--additional-config` 内联传入 `msprobe_dump_config`（dict，启动时自动落盘成 `.omni_npu/msprobe/msprobe_dump_config.json`）。
- 通过 `--additional-config` 传入 `msprobe_dump_config_path`（指向已存在的配置文件路径字符串）。

两者**互斥**，只能设置其一。常用字段如下：

| 字段         | 说明                                                                                                                                          | 必填 | eager 模式 | 图模式 |
|:------------:|:----------------------------------------------------------------------------------------------------------------------------------------------|:----:|:----------:|:------:|
| `task`       | dump 任务类型。PyTorch 常见值有 `"statistics"` 与 `"tensor"`。statistics 采集张量统计（均值/方差/最大/最小等），tensor 捕获任意张量。                |  是  |     ✅     |   ✅   |
| `dump_path`  | dump 结果存放目录。省略时 `msprobe` 使用默认路径。                                                                                              |  否  |     ✅     |   ✅   |
| `rank`       | 采样的 rank。空列表采集所有 rank；单卡任务必须设为 `[]`。                                                                                       |  否  |     ✅     |   ✅   |
| `step`       | 采样的 token 迭代。空列表表示每次迭代都采集。                                                                                                  |  否  |     ✅     |   ❌   |
| `level`      | dump 级别字符串（`"L0"`、`"L1"`、`"mix"`）。`L0` 面向 `nn.Module`，`L1` 面向 `torch.api`，`mix` 两者都采集。                                       |  是  |     ✅     |   ✅   |
| `async_dump` | 是否开启异步 dump（PyTorch `statistics`/`tensor` 任务支持）。默认 `false`。                                                                    |  否  |     ✅     |   ❌   |
| `scope`      | 采样的 module 范围。空列表采集所有 module。                                                                                                    |  否  |     ✅     |   ❌   |
| `dump_enable`| `PrecisionDebugger` 运行期动态开关，允许在同一个作业内按需开/关 dump。                                                                          |  否  |     ✅     |   ❌   |
| `list`       | 采样的算子范围。空列表采集所有算子。                                                                                                          |  否  |     ✅     |   ✅   |

要限制捕获的算子，可配置 `list` 块：

- `scope`（list[str]）：PyTorch PyNative 场景下限定 dump 范围。提供符合工具命名约定的两个 module 或 API 名以锁定区间，仅 dump 两者之间的数据。示例：

  ```json
  "scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]
  "scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]
  ```

  `level` 决定可提供的内容——`level=L0` 时为 module，`level=L1` 时为 API，`level=mix` 时两者皆可。

- `list`（list[str]）：自定义算子列表。选项包括：
    - 提供 PyTorch pynative 场景下特定 API 的全名，仅 dump 这些 API。示例：`"list": ["Tensor.permute.1.forward", "Torch.relu.3.forward"]`。
    - `level=mix` 时可提供 module 名，dump 会展开为该 module 运行期间产生的所有数据。
    - 提供子串（如 `"list": ["relu"]`），dump 所有名字包含该子串的 API；`level=mix` 时名字包含该子串的 module 也会被展开。

示例配置：
eager 模式：

```json
{
  "task": "statistics",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,

  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

图模式：

```json
{
  "task": "statistics",
  "level": "L1",
  "dump_path": "/home/data_dump",
  "statistics": {
    "list": []
  }
}
```

## 3. 在 omni-npu 中启用 `msprobe`

omni-npu 通过 `--additional-config` 传入 dump 配置，与 vllm-ascend 的入口一致。下面给出两种部署形态的用法。

### 3.1 直接用 vLLM 启动

通过 `--additional-config` 内联传入 `msprobe_dump_config`：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{
    "msprobe_dump_config": {
      "task": "statistics",
      "level": "L1",
      "dump_path": "/data/msprobe_dump",
      "statistics": {
        "list": []
      }
    }
  }' &
```

兼容模式（指向已存在的配置文件）同样支持：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 8000 \
  --additional-config '{"msprobe_dump_config_path": "/data/msprobe_config.json"}' &
```

### 3.2 通过 omni_cli 启动

omni-npu 的多机部署通常使用 `omni_cli`，`--additional-config` 通过 `omni_cli cfg` 设置：

```bash
# 内联 msprobe_dump_config
omni_cli cfg --set all args --additional-config '{
  "msprobe_dump_config": {
    "task": "statistics",
    "level": "L1",
    "dump_path": "/data/msprobe_dump",
    "statistics": { "list": [] }
  }
}'

# 或指向配置文件
omni_cli cfg --set all args --additional-config '{"msprobe_dump_config_path": "/data/msprobe_config.json"}'

omni_cli run_docker
omni_cli start
```

## 4. 发送请求并采集 dump

1. 照常发送推理请求，例如：

   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen2.5-0.5B-Instruct",
           "prompt": "Explain gravity in one sentence.",
           "max_completion_tokens": 32,
           "temperature": 0
         }' | python -m json.tool
   ```

2. 每次请求驱动 `msprobe: start → forward → stop → step` 序列。omni-npu 的 `NPUModelRunner.execute_model` 在入口调用 `_start_dump_data()`，并用 `try/finally` 在每条返回路径上调用 `_finalize_dump_data()`——即使推理提前返回或抛异常，也能拿到一份完整数据。

3. dump 文件写入 `dump_path`，通常包含：
   - 按算子/module 分组的张量文件。
   - `dump.json`，记录 dtype、shape、min/max、`requires_grad` 等元数据。
   - `construct.json`，当 `level` 为 `L0` 或 `mix` 时生成（可视化所需）。

   目录布局示例：
   eager 模式：

   ```text
   ├── dump_path
   │   ├── step0
   │   │   ├── rank0
   │   │   │   ├── dump_tensor_data
   │   │   │   │    ├── Tensor.permute.1.forward.pt                       # 格式: {api_type}.{api_name}.{call_count}.forward.{input/output}.{arg_index}.
   │   │   │   │    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # 格式: {Module}.{module_name}.{class_name}.forward.{call_count}.{input/output}.{arg_index}.
   │   │   │   │    └── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # Module 参数数据: {Module}.{module_name}.{class_name}.forward.{call_count}.parameters.{parameter_name}.
   │   │   │   ├── dump.json
   │   │   │   ├── stack.json
   │   │   │   ├── dump_error_info.log
   │   │   │   └── construct.json
   │   │   ├── rank1
   │   │   │   ├── dump_tensor_data
   │   │   │   │    └── ...
   │   │   │   ├── dump.json
   │   │   │   ├── stack.json
   │   │   │   ├── dump_error_info.log
   │   │   │   └── construct.json
   │   │   ├── ...
   │   │   │
   │   │   └── rank7
   │   ├── step1
   │   │   ├── ...
   │   ├── step2
   ```

   - `rank`：设备 ID，每张卡把数据写入对应的 `rank{ID}` 目录；非分布式场景目录名为 `rank`。
   - `dump_tensor_data`：采集到的张量数据。
   - `dump.json`：每个 API 或 module 的前向数据统计，包括名称、dtype、shape、max、min、mean、L2 范数，以及 `summary_mode="md5"` 时的 CRC-32。详见 [dump.json 文件说明](#dumpjson-file-description)。
   - `dump_error_info.log`：仅在 dump 工具遇错时出现，记录失败日志。
   - `stack.json`：API/module 的调用栈。
   - `construct.json`：层级结构描述，`level=L1` 时为空。

   图模式：

   ```text
   L0_dump
   ├── step0
   │   └── rank0
   │       └── dump.json
   ├── step1
   │   └── rank0
   │       └── dump.json
   ├── step2
   │   └── rank0
   │       └── dump.json
   ├── step3
   │   └── rank0
   │       └── dump.json
   ├── step4
   │   └── rank0
   │       └── dump.json
   └── step5
       └── rank0
           └── dump.json
   ```

   - `dump.json`：详见 [dump.json 文件说明](#dumpjson-file-description)。

## 5. 分析结果

### 5.1 前置条件

通常需要两份 dump 数据：一份来自"问题侧"（暴露精度/数值错误的运行），另一份来自"基准侧"（良好的基线）。两份数据无需完全相同——可来自不同分支、框架版本，甚至不同实现（算子替换、不同图优化开关等）。只要使用相同或相近的输入、硬件拓扑和采样点（step/token），`msprobe` 即可比较并定位发散节点。若找不到完美干净的基准，可先采集问题侧数据，手工构造最小可复现用例，再做自比较。下文假设问题侧 dump 为 `problem_dump`，基准侧为 `bench_dump`。

### 5.2 可视化

使用 `msprobe graph_visualize` 构建或比较图，再用 TensorBoard（`tb_graph_ascend` 插件）打开生成的 `*.vis.db` 文件。

1. 确保 dump 数据可视化就绪：
   - dump 级别必须为 `L0` 或 `mix`，使 `construct.json` 非空。
   - 每个 rank 目录应包含 `dump.json`、`stack.json`、`construct.json`。

2. 选择命令模式：
   - 单图构建：

     ```bash
     msprobe graph_visualize -tp <target_path> -o <output_path>
     ```

   - 图比较：

     ```bash
     msprobe graph_visualize -tp <target_path> -gp <golden_path> -o <output_path>
     ```

   - 常用可选参数：
     - `-oc` / `--overflow_check`：开启溢出标记
     - `-fm` / `--fuzzy_match`：开启节点映射的模糊匹配
     - `-lm` / `--layer_mapping [mapping.yaml]`：跨框架/层映射比较
     - `-tensor_log`：打印逐节点比较日志（tensor dump 场景）
     - `-progress_log`：打印详细进度日志

3. 路径粒度由 `graph_visualize` 自动检测：
   - 单 rank：`.../step0/rank0`
   - 多 rank（批量）：`.../step0`
   - 多 step（批量）：含 `step*` 的 dump 根路径

4. 输出文件：
   - 单图构建：`build_{timestamp}.vis.db`
   - 图比较：`compare_{timestamp}.vis.db`

5. 启动 TensorBoard 指向输出目录：

   ```bash
   tensorboard --logdir <output_path> --bind_all --port <optional_port>
   ```

6. 在可视化 UI 中检查结构与数值差异：
   - 切换 rank/step 快速定位不稳定节点。
   - 用搜索/过滤聚焦目标算子/module。
   - 比较模式下优先查看高亮的高差异节点，并追踪周边 I/O/参数。

## 6. 故障排查

- `RuntimeError: Failed to import AclGraphDumper from msprobe`：图模式需要从源码安装带 `aclgraph_dump` 的 msprobe，见 [1.2 节](#12-图模式-dump可选)。
- `ImportError: No module named 'msprobe'`：eager 模式下配了 `msprobe_dump_config` 却未安装 msprobe。omni-npu 采用硬失败策略——配了 dump 就必须有 msprobe，不会静默跳过。执行 `pip install mindstudio-probe`。
- 没有生成 dump 文件：确认 JSON 路径正确且每个节点对其有写权限；分布式场景需设置 `keep_all_ranks`，使每个 rank 各自写自己的 dump。
- dump 文件过大：先用 `statistics` 任务定位异常张量，再用 `scope`/`list`/`tensor_list`、`filters`、`token_range` 等缩小范围。

---

## Appendix

### dump.json file description

#### L0 level

L0 `dump.json` 包含 module 的前向 I/O 及参数。以 PyTorch `Conv2d` 为例，网络代码为：

`output = self.conv2(input)  # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`

`dump.json` 包含以下条目：

- `Module.conv2.Conv2d.forward.0`：module 的前向数据。`input_args` 为位置参数，`input_kwargs` 为关键字参数，`output` 存储前向输出，`parameters` 存储权重/偏置。

**注**：当传入 dump API 的 `model` 参数为 `List[torch.nn.Module]` 或 `Tuple[torch.nn.Module]` 时，module 级名称会包含列表索引（`{Module}.{index}.*`），例如 `Module.0.conv1.Conv2d.forward.0`。

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "pytorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 1.638758659362793,
     "Min": 0.0,
     "Mean": 0.2544615864753723,
     "Norm": 70.50277709960938,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 1.6815717220306396,
     "Min": -1.5120246410369873,
     "Mean": -0.025344856083393097,
     "Norm": 149.65576171875,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
    }
   ],
   "parameters": {
    "weight": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.05992485210299492,
     "Min": -0.05999220535159111,
     "Mean": -0.0006165213999338448,
     "Norm": 3.421217418558838,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
    },
    "bias": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.0489412058939071,
     "Mean": 0.006410328671336174,
     "Norm": 0.172635138820056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
    }
   }
  }
 }
}
```

#### L1 level

L1 `dump.json` 记录 API 的前向 I/O。以 PyTorch `relu` 函数为例（`output = torch.nn.functional.relu(input)`），文件包含：

- `Functional.relu.0.forward`：API 的前向数据。`input_args` 为位置参数，`input_kwargs` 为关键字参数，`output` 存储前向输出。

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "pytorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": -1.3364859819412231,
     "Mean": 0.03711778670549393,
     "Norm": 236.20692443847656,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": 0.0,
     "Mean": 0.1684949392040715,
     "Norm": 175.23345947265625,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.output.0.pt"
    }
   ]
  }
 }
}
```

#### mix level

`mix` 的 `dump.json` 同时包含 L0 和 L1 级数据，文件格式同上。

### 使用参考
[vLLM采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/vllm_dump_instruct.md)
[aclgraph_dump 使用指南](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/aclgraph_dump_instruct.md)
[PyTorch场景精度比对](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)