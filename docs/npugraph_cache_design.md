# FX py_code 编译缓存设计（npugraph_ex 编译产物跨启动复用）

> 分支：`feat/fx-pycode-cache`（基于 `mb/develop`）。
> 参照实现：vllm-ascend main `vllm_ascend/compilation/compiler_interface.py`（AscendCompiler，槽位寻址）。
> 实证环境：CANN 9.1.0 / torch_npu 2.12.0 / vllm 0.25.1 fork（omni）。证据轮次 2026-10-15 → 10-16（19 boot）。


> **实现状态（v3.10）**：本文机制均已合入 `feat/fx-pycode-cache` 分支；池化部分
> （mooncake 层）在 fusion `Pangu-FusionPool/npugraph_cache`，设计正本随包走
> （`npugraph_cache/docs/npugraph_cache_pool_design.md`）。canon 无条件执行。
> 本文档描述 digest/canon 侧（omni）；文中旧轮次验证记录保留当时名称。

## 1. 背景

92B INT8 decode 每次启动都要把模型经 torch.compile 编成 NPU 图代码：dynamo 把执行流
追踪成 FX 计算图，npugraph_ex 后端把 FX 图编译成一段自包含的 Python 源码（下称
py_code），运行期由它重建并捕获 AclGraph。一个引擎进程共 64 个编译单元（4 个
server × 16：4 个 backbone 区间 + 12 个 eagle/MTP 单元），冷启动编译段合计约
209 s（r14 ArmB 分区计时：区间 (8,8)/{4,4}/(12,12)/(1,2048) 首轮各 33.9/34.3/
33.7/42.3 s，加 Dynamo 追踪窗口），**每次 boot 恒付、零复用**。

零复用的原因是门没开：vLLM 框架自带编译缓存编排（四个后端钩子 + 落盘目录 +
槽位索引），但 omni 的 NpuGraphExAdaptor 只实现了 `compile()` 且返回句柄恒为
`None`——框架拿不到句柄就一个字节都不落盘，缓存体系从未启用。

vllm-ascend main 已为 npugraph_ex 实现了 py_code 缓存，可直接参照。但直接照搬会
踩我们栈上的两类实测问题（同槽位多图、运行期形状漂移，见 §2.3），因此本设计以
其为骨架，把寻址方式从"信任框架槽位"改造为"内容寻址 + 读侧校验"——这是与参照
实现的根本差异，也是本设计的核心（§3.2）。

目标收益：暖启动跳过 backend 编译段（上界 ~144 s/rank，Dynamo 追踪不跳）；与
静态内核池（static kernel cache）完全正交，可叠加。

## 2. 架构

### 2.1 分层与 torch_npu 的关系

```
vLLM 框架（编译缓存编排层）
  ├─ 槽位键 artifact_compile_range_{s}_{e}_subgraph_{i}（纯位置编号，零内容）
  ├─ 缓存目录 $VLLM_CACHE_ROOT/torch_compile_cache/<四因子哈希>/rank_<r>_<dp>/<prefix>/
  ├─ 索引文件 vllm_compile_cache.py（槽位 → 句柄）
  └─ 四个后端钩子：initialize_cache / compute_hash / compile / load
        │
omni NpuGraphExAdaptor（本补丁，omni/compilation/npugraph_ex.py）
  ├─ compile：调 npugraph_ex 后端 + 挂存盘钩子截获产物
  ├─ load：按内容指纹找文件、重建可执行函数
  └─ 静态内核运行时准备（与 static kernel cache 线共享）
        │
npugraph_ex（torch_npu 2.12.0 内置的 dynamo 编译后端）
  ├─ FX 图 → py_code（codegen 产物为 Python 源码文本）
  ├─ _CompiledFxGraph.load_artifacts：py_code → 可执行函数（版本门 "0.1"）
  └─ ⚠ 同一文件、两个模块名：npugraph_ex 的源文件在进程内会被加载两份
     ——顶层包名 `npugraph_ex`（backend 实际使用的类来自它）与
     `torch_npu.dynamo.npugraph_ex`（同一套文件的另一份拷贝）。Python 按
     模块名登记对象，两个名字即两个互不相识的独立模块，其中的
     `_NpuFxCompiler` 是两个不同的类对象（代码相同、身份不同）。存盘手段
     是临时替换该类的方法——若只替换其中一个名字下的类而实际编译走的是
     另一个名字下的类，替换永远不生效且不报任何错（缓存静默为空）。因此
     补丁枚举两个模块名，找到几个类就替换几个，结束时逐一恢复
        │
CANN / NPU 运行时（aclgraph 捕获、静态内核派发）
```

与 torch_npu 的关系一句话：**npugraph_ex 是 torch_npu 自带的图编译后端，我们不改
它一行代码**，只在调用窗外临时 monkey-patch 其内部函数 `_NpuFxCompiler
._get_compiled_gm` 截获 `get_code()` 产物；读回用其自带的 `load_artifacts`。产物
本身是自包含源码：scope_enter 在运行期按内嵌 tag 字符串 get-or-create 注册事件/
流、重建 AclGraph（config 以字面量内嵌于代码），因此跨进程加载不依赖当 boot 的
追踪期注册表。

### 2.2 四个后端钩子与生命周期（框架钩子路）

框架通过 `CompilerInterface` 的四个钩子驱动后端编译器，本设计与 vllm-ascend
参照实现实现的是同一个接口：

| 钩子 | 职责 | 框架调用点（vllm 0.25.1） | vllm-ascend | 本实现 |
| --- | --- | --- | --- | --- |
| initialize_cache | 框架把缓存目录与开关告诉后端 | backends.py:1094 | compiler_interface.py:282 | npugraph_ex.py:521 |
| compute_hash | 后端贡献"编译器因子"（目录哈希四因子之一，§2.3） | backends.py:1030 | :266 | :491 |
| compile | 编译一张图，返回 (可执行函数， 句柄即提货单) | backends.py:353 | :290 | :527 |
| load | 按句柄取货；返回 None 即回落到编译 | backends.py:250 | :338 | :735 |

（行号为撰写时快照，随各仓演进会漂移。）顺序即生命周期：
**compute_hash → initialize_cache → load → compile**——启动时框架先算目录
哈希（用到钩子 compute_hash）→ 把目录告诉后端（initialize_cache）→ 每张图先
按句柄要货（load）→ 要不到才让后端编（compile），编完收句柄记账入索引。

- **冷 boot**：每单元 load 必 miss → compile 在调用窗内截获 py_code 写
  `{cache_dir}/{key}.{digest}.ngc`，返回 `(callable, 句柄)`，框架记入索引。
- **暖 boot**：load 用**当前图现算 digest** 拼文件名，存在则读文件重建函数，
  backend 编译整体跳过；任何 miss/异常返回 `None` 回落 compile（新产物按新
  digest 落盘，自愈）。
- 总开关 `VLLM_DISABLE_COMPILE_CACHE=1` = 本特性之前的原行为；启用须解除并将
  `VLLM_CACHE_ROOT` 指到持久卷（默认容器本地盘，Pod 亡即失）。

### 2.3 缓存目录布局与目录哈希（四因子）

§2.1 分层图中的"<四因子哈希>"在此展开成完整定义。目录自上而下：

```
$VLLM_CACHE_ROOT/torch_compile_cache/<目录哈希>/     ← 任一因子变 = 换目录 = 整目录作废
└── rank_0_1/                                        ← rank × dp 下标（多 server 天然隔离）
    ├── backbone/                                    ← 模型前缀：主模型（model_tag 默认值）
    │   ├── vllm_compile_cache.py                    ← 索引：槽位 → 登记（运行时生成的缓存态）
    │   └── artifact_compile_range_8_8_subgraph_0.<digest>.ngc   ← 产物
    └── eagle_head/                                  ← 模型前缀：draft（装载时 set_model_tag）
        ├── vllm_compile_cache.py
        └── artifact_compile_range_8_8_subgraph_0.<digest>.ngc
            ↑ 同一槽位、多个 digest 共存（MTP 三变体），靠读侧内容指纹区分
```

目录哈希 = `sha256(str([env 哈希, config 哈希, 源码哈希, 编译器哈希]))[:10]`，由
`VllmBackend.__call__` 在进程内首个后端调用时计算一次并记在 compilation_config 上。
四个因子（以 vllm 0.25.1 为准）：

| 因子 | 算什么（来源） | 换目录的典型触发 | 不参与的典型项（易误解） |
| --- | --- | --- | --- |
| ① env 哈希 | `envs.compile_factors()`：全部已注册 vLLM 环境变量减 ignore 表，逐项归一 | 换注意力后端等实现选择类 env | 未注册的 HCCL_\* / TORCH_\* / 本设计 FX_\* 开关；被 ignore 的 VLLM_CACHE_ROOT、日志类 |
| ② config 哈希 | `VllmConfig.compute_hash()`：vllm 版本 + 全部配置节（model、cache、parallel、scheduler、compilation、kernel、…）各自的 compute_hash + additional_config 整体 json 哈希 | 改 TP / 量化 / max_num_batched_tokens / compilation 配置 / npugraph_ex_config 任一键 | num_gpu_blocks、gpu_memory_utilization 等运行期派生量（cache 节排除表——§2.3 缺口 2 的来源） |
| ③ 源码哈希 | 进程内首个后端调用模型（backbone）的 traced_files（路径 + 全文）sha256 | 改 backbone 及其内联文件（含纯注释改动） | eagle 源码（traced_files 消费即清空；draft 图差异由内容指纹 §3.2 兜底） |
| ④ 编译器哈希 | 本 adaptor `compute_hash`：backend 名 + torch_npu 版本 + CANN 版本 + npugraph_ex_config 子字典 | 升级 torch_npu / CANN | torch / python 版本（未单独列入） |

各配置节默认包含全部字段、仅减各自的排除表（scheduler 节只含 max_num_batched_tokens、
compilation 节排除目录/计时/traced_files）；npugraph_ex_config 随 additional_config 进
入 ②，又随 ④ 重复登记一份作双保险。

失效语义：任一因子变化 ⇒ 新目录 ⇒ 索引与产物整体不可达 ⇒ 下一靴全量重编一次后
自愈。粒度是**整目录级**，与内容指纹的单文件级（§3.1）互补——"版本与配置进目录
哈希、图内容进内容指纹"的分界由此而来。vllm 0.25.1 起框架还会在缓存目录写
`cache_key_factors.json`（env 原始因子与三个子哈希），是排查目录级 miss 的第一手
材料（镜像 fork 若无此文件，不影响机制）。

### 2.4 槽位键的两个缺口（为什么必须内容寻址）

槽位键只含"编译区间 + 子图序号"，不含图内容。两个实测缺口：

1. **同槽位多图**：MTP 形态下 eagle_head 同一 key 下实测共存 3 个变体
   （layers.46/47/48 三张不同的图）。按槽位取货只能拿到最后写入者——错载。
2. **运行期形状漂移**：KV 池块数靠启动时显存探测（同配置 ±1 漂移），框架配置
   哈希明确排除它（"不影响编译图形状"的假设在 CUDA 符号形状下成立、在捕获式
   实编路径下不成立——块数被烤进 py_code）。槽位法对其失明。

两者在 vllm-ascend 的被测形态（单模型、无 MTP）中不出现，因此其槽位法可用；
在我们栈中是硬约束，引出 §3.2 的内容寻址。

## 3. 设计方案

### 3.1 总纲：内容寻址 + 读侧校验

文件名 `{key}.{digest}.ngc`——前半是框架槽位键（兼容框架索引），后半是图内容
指纹（正确性的来源）。写侧截获产物即存；读侧**不信用写入期 handle 里的旧
digest**，用当前图现场重算。由此得到两条硬性质：

- **不错载**：图内容任何变化 → digest 变 → 文件名对不上 → 干净 miss、当场重编。
  绝不发生"槽位命中但内容不对"的静默错载。
- **自愈**：miss 永不阻塞启动（返回 None 回落框架编译路径），新产物自动补盘。

与 vllm-ascend 参照实现的差异一览：

| 维度 | vllm-ascend | 本设计 |
| --- | --- | --- |
| 文件名 | 槽位键 | 槽位键 + 内容指纹 |
| miss 语义 | load 内自建重编（复制预处理） | 返回 None 回落框架（自愈） |
| 补丁覆盖的模块名 | 只补丁单一模块名 | 枚举两个模块名逐个补丁（同一文件在进程内被加载两份，见 §2.1） |
| 非 str 产物 | `"triton_kernel_wrapper" in py_code` 会 TypeError | isinstance 门控，分桶跳过 |
| 哈希因子 | 4 因子 + TODO | 增补 CANN 版本 + npugraph_ex 配置子字典 |

### 3.2 指纹：如何保证命中的缓存是对的

> **部署注意（v43 实测）**：框架 config_hash 含**模型路径**（vllm `config/model.py:386-409`
> 的 ignored_factors 无 `model`）——同一份权重换路径访问（symlink 改名等）会使目录哈希
> 与 bundle 整体换代、池全 miss；跨路径实验必须共用同一路径。


digest = `sha256(被哈希文本)[:16]`，被哈希文本 = 图内逐节点一行 + 每个
example_input 一行（save/load 两侧对称计算，均取改写前的图）。槽位隔离由
文件名前半段承担，key 不在哈希文本内。

**指纹的组成**（确定性文本序列，逐节点一行、按拓扑位置编号 n1…nN，与
tracer 变量名无关；占位符/输出名按位置丢弃；字符串叶经 `^graph_\d+_`→
`graph*_` 归一化；实测摘录）：

```
call_function|<OpOverloadPacket(op='npu.npu_rms_norm')>|t(n60,n125,1e-05)|d{}
call_function|<OpOverloadPacket(op='vllm.npu_pangu_mome_conv_from_context')>|t(n118,n27,n28,'model.layers.46.self_attn')|d{'phase':'decode'}
call_function|<OpOverloadPacket(op='npu.npu_moe_distribute_dispatch_v2')>|t()|d{'comm_alg':'fullmesh_v2','ep_rank_id':0,…}
tensor:torch.bfloat16:6641,5,1024        ← example_input 的 dtype:shape（KV 池张量，N 在此入指纹）
```

即：算子身份 + 拓扑结构 + 全部标量/字符串参数（含层身份、ep_rank_id）+
kwargs（排序）+ 输入张量 dtype/shape。`NPUGRAPH_DIGEST_DUMP` 落盘的正是
这份原文——指纹输入随时可审计，两代 dump diff 即可定位任何 miss。

正确性论证是一个双向不变式：

> **语义相同的图 ⇒ digest 相同**（该命中的命中，不多付重编）；
> **语义不同的图 ⇒ digest 必不同**（不该命中的绝不命中，不错载）。

实现上分两半：**输入覆盖一切语义维度**（防漏检，表 A），**剔除一切非语义
噪声**（防误检，表 B）。

**表 A｜保留的语义维度**（漏检=错载，故从严）：

| 信息 | 形式 | 为什么必须在 | 实测样例（摘自 dump） |
| --- | --- | --- | --- |
| 算子集合与拓扑 | 每节点一行 `{op}\|{target}\|{args}\|{kwargs}`，节点按拓扑位置编号 n1…nN | 图变了 digest 必须变 | — |
| 算子身份 | OpOverload 保留 repr | 换算子实现即换产物 | `<OpOverloadPacket(op='npu.npu_rms_norm')>` |
| 层身份 | get_attr/call_module 的 target 名 + 算子参数里的层名字符串 | MTP 三变体的唯一区分点，天然分键 | `'model.layers.46.self_attn'`、`d{'phase':'decode'}` |
| 全部标量/kwargs 常量 | 字面量逐叶渲染，dict 按键排序 | eps、ep_rank_id 等语义参数 | `t(n60,n125,1e-05)`、`'ep_rank_id':0` |
| 输入 dtype:shape | 每个 example_input 一行 `tensor:{dtype}:{shape}` | KV 池块数 N 由此进入：漂移 → 换 digest → 干净 miss（而非错载） | `tensor:torch.bfloat16:6641,5,1024` |
| 符号形状 | backed SymInt 按 hint 值；unbacked 按首现序字母（保留 tie 结构） | 动态维的取值与并列关系 | `symint:8` / `symint:$a` |

**表 B｜剔除的非语义噪声**（不剔=跨 boot 假 miss；v1→v4 三层治理，每层都有
实测定罪证据）：

| 噪声源 | 症状 | 归一化手段 | 定罪证据 |
| --- | --- | --- | --- |
| 命名层：dynamo 变量名 | 同槽两轮产物逐字节等价而 raw digest 全异 | 节点按拓扑位置命名；placeholder/output 名丢弃；SymInt 按 backed hint 取值；kwargs 排序 | v1（C1r3 vs C2r3） |
| 渲染层：repr 内存地址、torch 运行期符号名（"s0"/"s3"） | 两轮产物 md5 全等而 digest 全变 | `_stable_repr`：backed sym 按 hint、unbacked 按首现序字母；callable 按 `module.__qualname__`（OpOverload 保 repr）；含 ` at 0x` 按类型兜底。门禁=双进程同图同 digest | v2→v3（run6 vs run7 backbone 逐字节同、digest 全异） |
| 计数器层：npugraph_ex tag 的本 boot 构建计数器 | 命中加载的 boot 少 build 即 tag 整体前移 | 字符串叶 `^graph_\d+_`→`graph*_`；stream id/设备值双代稳定，保留 | v4（eagle tag `graph_4/8/12`→`1/4/7`，位移恰=该 rank 加载单元数） |
| 参数值 | — | get_attr 只留 target 名（特性：权重换代/换权重池不失效——产物是代码，权重运行期解析） | — |

**边界**（有意不含，各有归属）：stride/device 不入指纹（codegen 的
`assert_size_stride` 守卫兜底，错则捕获期响亮失败）；槽位 key 不入（文件名
前半段承担）；node.meta 不入；编译器管线与配置不入——归目录哈希四因子
（§2.4）管。

**读侧校验落位**：digest 在 save/load 两侧对称计算，且都在 tuple 化改写**之前**
取图（同一前置内容）；load 侧用当前图现算（§3.1）。

**安全网**（指纹之外的防线）：

- 非 str 产物（codegen 不支持的配置返回 runner 对象）、含 Triton 内核的图
  （kernel_side_table 进程内不可序列化）、get_code 异常——一律不落盘，每 boot
  重编（正确性优先）；
- `load_artifacts` 版本门 "0.1" 不匹配 → miss；
- compute_hash 增补 torch_npu/CANN 版本与 npugraph_ex 配置子字典 → 版本或配置
  变化整体换缓存目录（粗粒度失效，双保险）。

**运行期正确性**（命中之后仍要正确）：

- **事件 tag 落盘改名**：多单元加载时各产物的 `graph_N_` 事件 tag 撞名，get-or-
  create 会令两单元共享同一事件对象、交错执行互相误唤醒。落盘前把事件类 tag 改
  为 `gfx{digest12}_` 唯一命名空间；**裸流 tag 保留原名**——跨单元共享流对象是
  get-or-create 的设计语义（实测 eagle 产物引用 backbone 建立的流对象）。
- **静态内核触发链保留**：全命中时框架从 load 早退、不进 compile，静态内核的
  vendored patch 应用会丢 → 抽 `_ensure_static_kernel_runtime_setup()` 供
  compile/load 双路调用（幂等）；且 codegen 产物以字面量内嵌全量 config 重建
  AclGraph，暖 boot 静态内核派发链完整（C 臂 op_summary static 计数与全编臂
  逐位一致实证）。
- **AOTAutogradCache 冲突压制**：框架缓存开启时 dynamo `aot_compile_fullgraph`
  会 patch `bundled/force_autograd_cache=True`，令 torch 自家 AOT 缓存对本后端
  产物断言（expected OutputCode, got _CompiledFxGraph）→ compile() 内对后端调用
  生效期 `torch._functorch.config.patch(四旗 False)`（框架对自家 InductorAdaptor
  的同款处置）。
- **诊断插桩 NPUGRAPH_DIGEST_DUMP**：env 置位时把被哈希的精确原文落盘（best-effort），
  再遇 miss 直接 diff 两代原文逐行定位，不靠猜。

### 3.3 形态归一化：算指纹前统一同步 lowering 形态

**问题**：同一张逻辑图到达后端时可能处于两种等价的同步 lowering 形态（机制见
§3.4）；原始指纹因此对一张图给出两个值——假阴性的 miss，梯次收敛与重复产物
皆源于此。

**设计**：算指纹前，把图 **deepcopy 一份，在副本上运行 npugraph_ex 自己的
`replace_stream_event_pass`**——也就是后端稍后会对真图执行的那道工序——对翻译
后的副本计算指纹。两种形态由此收敛到后端自身的目标词汇（命名事件形态），两种
写法命中同一份存货。翻译规则不自行实现，直接复用 pass 本体：等价性由构造保证。

实现要点：

- **必须在后端调用窗口内执行**：pass 依赖 dynamo 的外部对象表（流/事件对象），
  窗口外对象已回收（`User object is no longer alive`，门 1 v2 实证）。指纹在
  `compile()` / `load()` 入口计算，恰在窗口内。
- **副作用隔离**（门 1：12/12 PASS）：`GraphCounter.COUNT` 存/取还原（真编译
  的编号不受扰动）；`_create_event_by_name` 在**每个持有该名字的模块**上替换为
  空操作（`graph_pass` 与 `replace_stream_event` 各持一份绑定——`from … import`
  按模块绑定，只改定义模块无效），不创建真实 NPU event；调用方图不被修改。
- **失败安全**：pass 不可用/签名变化/翻译异常 → 记录一行 `[npugraph-canon] … FAIL`
  并**退回严格指纹**（= 本特性引入前的行为）。canon 无条件执行、无开关（原
  `FX_CANON_PASS` 已于 2026-10-27 删除——与异常自动回退完全冗余）。
- **实现状态**：归一化逻辑**已合入本分支**（2026-10-26，自 SFS 三合一实验补丁逐字
  移植，`import time` 在模块顶）。此前它只存在于实验补丁：v3.1 池化验证轮曾用分支
  文件覆盖部署补丁而把 canon 丢掉，实测退化为 §3.4 梯次（每靴 12/16 命中、族序
  4_4→12_12，NPUGRAPH_DIGEST_DUMP 对 diff 实锤 streams↔tagged 形态翻转）——已定罪修复。
  无条件执行（原 `FX_CANON_PASS` 开关已删除，2026-10-27）。
- **开销**（实测）：backbone 大图（~11k 节点）deepcopy+pass 约 0.6–1.1 s/单元，
  eagle 小图 ~8 ms；单靴 compilation +3~6 s（相对 ~158 s 收益可忽略）。
- **规范形即后端口径**：翻译产物与后端在 tagged 形态图上原样透传的结果一致
  （后者 `canon == raw`，日志可直接对照）。

**验证**（2026-10-22/23，门 1–3，A/B 臂形态、独立缓存根、实验补丁）：

- 门 1（机制）：翻译完整（`streams.*` 残留 0）、计数器还原、事件表 0→0（拦下
  2 个真事件）、调用方图零污染、2025 节点图 52 ms。
- 门 2（等价/收敛）：4 靴——boot1 populate 64 编译落盘；**boot2/3/4 全命中
  （Loaded 64 = 16/rank，Compiling 0，Saved 0）**；同一单元原始指纹跨靴变化
  （如 `8935b73e`/tagged → `56226f02`/mixed）而归一化指纹恒定；落盘恰
  **64 = 4 rank × 16 个图**（无重复代；旧口径 112 含梯次重复）；离线 432 份
  dump、按 key+rank+层签名 64 组回归 **0 残差**。
- 门 3（精度）：CEval 1346 题 RC=0，correct 1161 vs 同形态全编基线 1156
  （**wrong_growth −5**，门限 8）；accuracy 0.8626 vs 0.8588；warm 靴
  init engine 594–596 s。

### 3.4 首灌收敛行为（归一一化无条件；梯次为历史机制）

**现状（无条件归一化）**：boot 1 populate（全编一次）→ **boot 2 起永久全命中**。
两种同步 lowering 形态在算指纹前被统一翻译到后端自己的命名事件词汇（§3.3），
不存在"关闭形态"——曾有 `FX_CANON_PASS` 开关，2026-10-27 删除（与异常自动回退冗余）。

**梯次机制（历史，归一化引入前的行为，保留作原理说明）**：同一张逻辑图有两种等价
同步 lowering 写法（流式 `streams.*` vs 命名事件 `npugraph_ex.tagged_event_*`），选择
取决于"该模型在本 boot 是否发生过实际编译"（缓存加载不触发切换）。不做归一化时：
boot 1 混合落盘 → 每 boot 恰有一族单元（实测族序 4_4→12_12→1_2048）指纹不匹配、
实际编译一次并触发 lowering 路径切换 → 五靴到达不动点。此类 miss 为假阴性（两形态
语义等价，非正确性问题）。v3.1 池化首轮的 12/16 命中即部署丢失归一化后的梯次复现
（FX_DIGEST_DUMP 跨靴 diff 定罪）。

### 3.5 部署要求

- **缓存开关只有一个：框架原生的 `VLLM_DISABLE_COMPILE_CACHE`**（经
  initialize_cache→disable_cache 到达两个钩子的门）。不设 fx 级配置键：static kernel
  池指纹对 whole `npugraph_ex_config` 做 sha256（键集+值类型双敏感），增删键会平移
  池指纹、破坏池复用，且该键与框架 env 开关完全冗余（2026-10-27 评审删除
  `fx_cache_enable`，原"opt-out 置 false"用法由 env 承担）。
- 形态归一化**无开关、无条件执行**（§3.3）：pass 异常自动回退严格指纹（5 靴
  梯次行为）；原 `FX_CANON_PASS` env 已删除（2026-10-27，与自动回退冗余）。
- 解除 `VLLM_DISABLE_COMPILE_CACHE`；`VLLM_CACHE_ROOT` 指到持久卷（k8s 验证用
  SFS 子目录；写盘走"临时文件+原子替换"，SFS 拒 rename 时回退直写——同 key 单
  rank 内串行，直写安全）。
- **KV 块数不钉扎（决策 2026-10-22）**：生产不允许加
  `--num-gpu-blocks-override`，因此保持指纹对池块数严格敏感——
  **同一 N 命中、N 漂移即整靴全量 miss 重编**（每个 decode 单元都以池张量为
  输入，实测留存产物 28/28 单元含池块数，无"部分重编"可省）。
  代价有上界：**每个新出现的 N 只付一次重编**——产物按 N 分代共存，N 在有限
  取值间振荡时第二靴起恢复全命中；仅在 N 单调漂移（每靴都换新值）时缓存失效。
  依据：产物的 `assert_size_stride` 形状守卫在首次调用（即捕获期）执行
  （`_codegen_assert_size_stride`，生成于 `fx2acl_converter.py:605`，包在
  `if _is_first_run:` 内），加载异 N 产物会当场断言，因此**不可**把 N 从指纹
  豁免。若将来 N 单调漂移成为常态，再评估"剥池守卫 + 归一化 N"方案——证据面
  已备：N 在全部产物中只出现在守卫与未使用的默认值里，算子序列 0 处含 N。

### 3.6 已知边界

- 形态归一化（§3.3）依赖 npugraph_ex 私有 pass（`replace_stream_event_pass`）：
  签名/行为变化由异常自动回退兜底退回严格指纹（后果仅是恢复 5 靴梯次，
  非错误）。归一化关闭时的梯次收敛见 §3.4。
- 同 key 多 digest 变体文件共存（~100KB-1.4MB/个），不做兄弟清理（同 boot 合法
  多形态会被误删）；digest 算法换版使旧 .ngc 自然旁置。归一化开启后每个图仅有
  一个 digest（实测 4 rank × 16 图 = 64 个产物），此项主要约束回退形态。
- repr 层每进程 token 是 digest 天然污染源：新增叶类型必须进 `_stable_repr` 分支
  或证明 repr 稳定，诊断先开 NPUGRAPH_DIGEST_DUMP。
- 与 per-key static kernel cache（M3）无耦合：键体系、存储介质、失效判据独立。上池
  （mooncake）为后续可选扩展（产物 KB~百 KB 级，优先级低于 .run 池化）。

### 3.7 验证摘要（19 boot，A/B/C 三臂 + 归一化门 1–3）

| 项 | 结果 |
| --- | --- |
| C 臂（static kernel 池开）FULL HIT | init engine 263→97 s（−165.6 s/rank，−78%）；编译段 209→53 s；零编译行 |
| A/B 臂（无池） | B FULL HIT；B−A1 = −158 s/rank 全额在编译段，非编译段 543 s 持平=与 static kernel 池正交 |
| 梯次收敛 | 两环境 5-boot 零偏离，命中/重编数精确如预测 |
| 精度 | CEval 1346 题 wrong_growth −1/−2（重跑噪声带内） |
| 静态内核派发 | profile static 计数与全编臂逐位一致（load 路径 static kernel 准备有效） |
| 形态归一化 · 门 1 | pass 副本 12/12 PASS：翻译完整、计数器/事件表零污染、2025 节点 52 ms |
| 形态归一化 · 门 2 | 4 靴：boot2 起全命中（Loaded 64、Compiling 0）；原始指纹跨靴变化而归一指纹恒定；产物 64（无重复）；离线 64 组回归 0 残差 |
| 形态归一化 · 门 3 | CEval RC=0、wrong_growth −5（≤8）、accuracy 0.8626 vs 0.8588；warm init engine 594–596 s |

门禁件沿用 r14/r15（就绪口径、CEval tol8、msprof op_summary OP State 权威判据）；
分级判读与逐轮证据见台账 fx 前缀条目。
