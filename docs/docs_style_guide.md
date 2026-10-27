# OmniInfer 文档规范与目录清单

> **适用范围**：OmniInfer（vLLM 昇腾 NPU 平台适配开源项目）
> **参考实践**：vLLM / PyTorch / Kubernetes 文档体系，采用 Diátaxis 四层文档法
> （一种专门针对**软件技术文档**的系统化架构方法论，由 Canonical 公司的文档专家 Daniele Procida 提出。它将技术文档清晰地划分为**四个核心维度**，每个维度对应用户在特定场景下的不同需求和思维模式：Tutorial → How-to → Reference → Explanation）。

---

## 1. 现状与问题

| 现状 | 问题 |
|---|---|
| `docs/` 仅有 installation / quick_start / CLI / kv_dump 四篇 | 覆盖面窄，无特性级用户指南 |
| README 链接 `omni/accelerators/sched/global_proxy/README.md`、`omni/accelerators/placement/README.md` | 仓库中无 `accelerators/` 目录，链接失效 |
| 环境变量说明散落在 `omni/worker/README.md`（profiling）等模块 README | 用户需翻源码目录找配置项 |
| 设计知识只存在于代码注释与 SIG 会议材料 | 无 design 文档，外部贡献者上手成本高 |
| 无统一命名/双语/图表规范 | 各文档风格不一致 |

## 2. 文档分层模型

```
L0  README.md（中英双语）        —— 项目门面：特性矩阵、架构图、快速链接
L1  Quick Start                —— 教程：30 分钟跑通一个最小推理/PD 部署
L2  Installation Guide         —— 部署：硬件/驱动/镜像/依赖检查与安装
L3  User Guide（按特性）        —— How-to：针对每个核心特性的使用方法
L4  Design Docs                —— 解释：重要特性的设计原理与架构决策
L5  Developer Guide            —— 贡献：配置开发、测试、调试、补丁机制
L6  Reference                  —— 查询：配置项、环境变量、API、支持的模型
```

分层原则：
- **用户文档（L1–L3、L6）面向使用者**，以任务为导向，命令可复制执行；
- **设计与开发文档（L4–L5）面向开发者/贡献者**，记录"为什么这么设计"，而非代码复读；

## 3. 建议目录清单

```
docs/
├── figures/                              # 全部配图统一存放
├── quick_start.md                        # 快速开始（单机推理 + 最小 PD 部署）
├── installation/
│   ├── installation_guide.md             # 现有 omni_infer_installation_guide.md 迁入
│   ├── hardware_and_driver_check.md      # npu-smi / CANN / 驱动检查（从安装指南拆出）
│   └── container_guide.md                # 镜像获取与 docker 使用（现 docker/README.md、docker/codes/README.md、docker/copy_data/README.md 迁入）
├── user_guide/
│   ├── quant_tools.md                    # 量化工具使用入口：索引并链接 tools/quant 下按模型拆分的量化文档（modelslim、jointfix），文档本体留在原地
│   ├── profiling.md                      # 性能数据采集（现 omni/worker/README.md 内容迁入），入口链接 benchmarks/README.md 压测说明
│   ├── trace.md                          # 现有 tools/omni_trace/README.md 迁入（trace 功能配置与使用）
│   ├── metrics_monitoring.md             # omni_metrics 监控指标
│   ├── ansible_deployment.md             # Ansible 部署脚本使用指南（Inventory 约定、Profile 配置、Tags、常用命令）
│   ├── k8s_deployment.md                 # 现有 tools/deploy/k8s/README.md 迁入（KubeInfer k8s 环境服务拉起）
│   ├── kv_dump.md                        # 现有 docs/kv_dump.md 迁入（KV cache 传输校验与踩踏检测工具使用）
│   └── troubleshooting.md                # FAQ / 常见报错
├── design/
│   ├── overview.md                       # 总体架构（对应 high_level_arch 图）
│   ├── expert_load_balancing.md          # 专家负载均衡设计（OmniPlacement / EPLB，分层冗余、动态放置、通信后端）
│   ├── acl_graph.md                      # ACL Graph 捕获与编译 pass 设计
│   ├── kv_transfer.md                    # KV 传输连接器设计（LLMDataDist / EC 抽象与实现）
│   ├── kv_offload.md                     # KV CPU 卸载调度器设计
│   ├── ansible_deployment.md             # Ansible 部署框架设计（三层架构：Playbook 配置层 / common 任务层 / 专用 Role 编排层，运行链路）
├── reference/
│   ├── env_vars.md                       # 全量环境变量清单（收敛散落各处的说明）
│   ├── additional_config.md              # OmniAdditionalConfig 参数参考（仅 vLLM 命令行 --additional-config 传入的参数）
│   ├── model_extra_config.md             # ModelExtraConfig 参数参考（TaskConfig / ModelParallelConfig / ModelOperatorOptConfig 三类模型配置项）及其启动校验规则（现 omni/docs/config_validation_rules.md 内容合入：参数及校验）
│   └── supported_models.md               # 支持模型矩阵：支持的模型清单、各模型对应的 Ansible 拉起脚本、各模型不支持的关键特性（对应 omni/model_config 及 match_hf_configs.json 登记信息）
├── developer
│   ├── contribution_overview.md          # 贡献流程总览（承接 CONTRIBUTION.md）
│   ├── config_dev_guide.md               # 现有文档迁入
│   ├── model_config_dev_guide.md         # 现 omni/model_config/README.md 迁入
│   ├── vllm_patches_guide.md             # 补丁开发规范（现 omni/vllm_patches/README.md 迁入）：补丁类定义、@register_patch 注册、命名与放置约定
│   ├── testing_guide.md                  # 测试分层：无硬件单测（tests/unit）与集成测试（tests/integration）
│   └── docs_style_guide.md               # 本规范自身
└── community.md（可选）                   # 社区、SIG、活动入口（链接 omniai/community）
```

迁移映射（现有文档 → 新结构）：

| 现有位置 | 迁移目标 |
|---|---|
| `docs/omni_infer_installation_guide.md` | `installation/installation_guide.md` |
| `docs/omni_infer_quick_start.md` | `quick_start.md` |
| `docs/kv_dump.md` | `user_guide/kv_dump.md`（不改名） |
| `omni/docs/config_dev_guide.md`、`config_validation_rules.md` | `developer/config_dev_guide.md` / `reference/model_extra_config.md`（校验规则部分合入） |
| `omni/model_config/README.md`、`omni/vllm_patches/README.md` | 分别迁入 `developer/model_config_dev_guide.md`、`developer/vllm_patches_guide.md` |
| `omni/worker/README.md`（profiling 部分） | `user_guide/profiling.md` |
| `tools/omni_trace/README.md` | `user_guide/trace.md` |
| `tools/deploy/k8s/README.md` | `user_guide/k8s_deployment.md` |
| `docker/README.md`、`docker/codes/README.md`、`docker/copy_data/README.md` | `installation/container_guide.md` |
| `tests/docs/`（README / QUICKSTART / STRUCTURE / TESTING 四篇） | 合并迁入 `developer/testing_guide.md` |
| `tools/ansible/DEVELOPMENT_GUIDE.md` | 拆分：使用部分（§3/§5/§7–§9）→ `user_guide/ansible_deployment.md`；设计部分（§1/§2.1 三层架构与运行链路）→ `design/ansible_deployment.md`；Playbook 编写与 Role 扩展部分（§4/§10）留在 `tools/deploy/ansible/` 原地（面向仓库内贡献者）或迁入 `developer/` |

冗余文档清理（随文档整改一并处理）：

| 文件 | 问题 | 处理 |
|---|---|---|
| `components/README.md` | 空文件（0 字节） | 删除 |
| `omni/TEST_SUMMARY.md` | 一次性测试搭建工作记录，内容已被 `tests/docs/README.md` 覆盖 | 删除 |
| `tools/ansible/` 整套目录 | 与 `tools/deploy/ansible/` 平行重复且已分叉（缺 inventories、epd_server、fusion_pool role 及生产 playbook） | 删除，统一保留 `tools/deploy/ansible/`；本规范中 Ansible 相关迁移来源均以 `tools/deploy/ansible/` 为准 |

## 4. Design 文档模板参考

每个 `design/*.md` 统一采用以下目录结构（业界 RFC/Design Doc 最佳实践）：

```markdown
# <特性名> 设计文档

1. 背景与动机            # 要解决什么问题，为什么现有方案不够
2. 目标 / 非目标          # 明确边界，防止范围蔓延
3. 总体架构              # 必须配图（架构图 + 数据流图）
4. 关键设计决策          # 每条：决策内容 → 备选方案对比 → 选择理由
5. 模块说明              # 关键类/接口职责，不逐行解释代码
6. 与 vLLM 上游的交互    # 覆盖了哪些上游接口、patch 点、版本耦合
7. 性能与验证            # 基准数据、测试方案、已知限制
8. 兼容性与版本约束       # vLLM / torch_npu / CANN / 驱动版本矩阵
9. 后续演进              # Roadmap、待决问题
10. 参考资料
```

## 5. 写作规范

1. **命名**：目录与文件一律小写 + 下划线（`docs_style_guide.md`）；design 文档名 = 特性名，不写 `design_xxx.md` 前缀（已在 `design/` 目录下）。
2. **双语**：`docs/` 正文中文为主，`README` 与每篇文档首行提供英文版链接；英文版以 `_en` 后缀平行存放（与现有 `README.md` / `README_en.md` 惯例一致）。
3. **配图**：统一存 `docs/figures/`，文件名语义化（如 `pd_disaggregation_topology.png`），禁止 UUID 类命名；源文件（drawio 等）一并入库。
4. **命令块**：所有 shell 命令标注解释性注释；配置示例给出最小可用配置 + 关键参数说明，禁止出现内网地址、账号等敏感信息。
5. **交叉引用**：文档间用相对路径链接并随 PR 校验死链；README 的"核心特性"列表必须链接到对应 user guide（当前指向不存在的 `accelerators/` 路径，需修复或先创建占位文档）。
6. **Reference 单点维护**：涉及环境变量、additional_config、ModelExtraConfig 参数、支持模型清单等已有 reference 文档的内容，user_guide 与 developer 文档只写"做什么、怎么选"，点到参数时链接到 `reference/` 对应文档，不在本篇展开重复罗列，避免参数说明多处维护产生分叉。
7. **版本声明**：每篇文档头部标注适用的 vLLM / CANN / 镜像版本（项目与 vLLM 0.25.1 强耦合，文档随版本演进需显式版本化）。
8. **图数约定**：design 文档至少 1 张架构图；user guide 关键步骤至少 1 个可复制执行的完整示例。
9. **流程约束**：新增特性 PR 必须附带对应 user guide 章节或 design 文档（写入 CONTRIBUTION.md 作为合入门禁）；纯代码重构若改变文档所述行为，需同步更新文档并在 PR 描述中列出。
