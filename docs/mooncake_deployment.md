# Mooncake 普通与高可用部署

OmniInfer 仓库提供 [`tools/scripts/start_mooncake.py`](../tools/scripts/start_mooncake.py)。下游选择部署模式，提供存储
主机 IP、集群名称和数据目录，脚本负责 etcd、Master 和常驻内存池的参数及生命周期。
启动器本身仅使用 Python 标准库，可作为容器前台入口或由作业调度器启动；它启动的
provider 和检查子进程会导入 Mooncake。使用 NPU wheel 时，即使只存储 CPU 数据，
Mooncake 原生库仍可能依赖 CANN 和宿主机 driver 库。
对下文所述包含 Mooncake 激活脚本的镜像，直接运行 Python 入口即可；CANN、Mooncake
及动态库路径由入口自动初始化，无需先手动 `source`，也无需额外的启动 `.sh` 文件。

| 模式 | 存储主机 | 服务组成 | 客户端副本数 |
| --- | --- | --- | --- |
| `--mode standalone`（普通部署） | 1 台 | 单 etcd、非 HA Master、单 provider | 1 |
| `--mode ha`（高可用部署，默认） | 3 台独立主机 | 每台一个 etcd member、候选 Master 和 provider | 2 |

普通模式保留 etcd 作为 Transfer Engine 的元数据服务，客户端直接连接单 Master。
HA 模式的三个 etcd member 负责仲裁，Master 开启 etcd 选主和 OpLog 回放。
两种模式的 provider 都承载实际 CPU embedding 数据，RL writer 和推理 worker 始终是
`global_segment_size=0` 的纯客户端，各角色共用脚本生成的地址和 namespace。
`start`、`client-config` 和 `check` 必须使用一致的 mode、hosts、cluster-id 和端口。
省略 `--mode` 时仍使用 HA，兼容原有三节点命令。

## 环境准备

- 普通模式需要一台 Linux 存储主机；HA 模式需要三台**独立 Linux 主机**，地址互通。
  同机的三个地址不代表三个故障域。RL/推理客户端可以运行在存储主机或其他可达主机上。
- 各节点使用同一版本的本仓库和 Python >= 3.11。容器内仓库路径为 `/workspace/omniinfer`，
  启动入口为该仓库下的 `tools/scripts/start_mooncake.py`。
- 镜像内准备 `etcd`，以及支持 etcd 的 Mooncake `0.3.13.post1`。
  启动器接受 `mooncake-transfer-engine-non-cuda`、`mooncake-transfer-engine-npu`
  或 `mooncake-transfer-engine`，同一 Python 环境必须只安装其中一个发行包，且版本匹配。
  已有依赖满足要求时直接复用，无需重新安装。
  已有部署验证过 etcd 3.7.1。默认 CUDA wheel/镜像可能不含 etcd 后端，不能只以 import 成功为准。
  所选 wheel 必须含 `libetcd_wrapper.so`、Store/ReplicateConfig API，Master 必须支持 HA 和
  OpLog 参数。普通模式使用同一套依赖，也需要 etcd 后端。
- 每节点至少预留 `--pool-mib` 指定的内存（默认 4096 MiB），另留本地缓冲和服务开销。
  etcd 的 `--data-dir` 使用本机持久盘；不同节点不能共享同一 etcd 数据目录。
- 当前入口部署 CPU/TCP、内网无 TLS 的服务，适用于互信的训练网络；不向公网开放。
  已有 RDMA/KV 通道应使用不同端口和 namespace。

脚本不下载依赖、不启动 SSH、不修改训练拓扑。将同一个入口交给现有容器/调度器，在各
存储节点启动即可；HA 的三个节点需并行启动。容器使用 host 网络，直接运行容器内的仓库代码，
挂载持久化数据目录，并给至少 60 秒停止宽限。容器或 systemd 应配置自动重启，以覆盖启动器自身退出
或整机重启；脚本内部负责子进程异常退出后的退避重启。

### 复用已包含 NPU wheel 的镜像

部分镜像将 Mooncake 放在 `/opt/mooncake-py311`，默认 Python 环境不能直接导入。
Linux 下，`start_mooncake.py` 在 `start`（包括 `--check-only`）、`check` 和内部 native
子进程的依赖检查之前，检测 `/opt/mooncake-runtime/activate.sh`。该文件存在时，入口通过
Bash 加载镜像内已有的 CANN、Mooncake 环境，并补齐 driver 动态库路径。
下面展示的是入口内部自动执行的步骤，部署者直接使用后面的 Python/Docker 命令即可：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /opt/mooncake-runtime/activate.sh
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MC_FORCE_TCP=1
```

初始化后，入口使用原解释器和原参数重新执行 Python，使 `LD_LIBRARY_PATH` 在加载原生库前
生效。后续子进程继承该环境，不会重复初始化。

若 Mooncake 激活脚本不存在，入口沿用调用者已经配置的通用环境；若该脚本存在但 CANN
`set_env.sh` 缺失，则明确报错；任一环境脚本执行失败时立即退出，不继续启动服务。
`--dry-run`、`client-config` 和 `--help` 不加载原生环境，也不能用来验证 CANN/driver 是否可用；
本机依赖预检使用 `start --check-only`，启动后的读写验收使用 `check`。
这些步骤只激活已有依赖，不安装或重新编译任何包。存储容器可使用 `runc`，不映射 NPU
设备，仍需将宿主机 `/usr/local/Ascend/driver` 只读挂入容器以提供链接所需的库。
本入口始终配置 CPU/TCP，这种容器不验证 NPU 或 RDMA 数据传输。

下面在宿主机准备持久化数据目录，并启动容器。容器直接使用 `/workspace/omniinfer` 中的
仓库代码。`IMAGE` 填写使用的 OmniInfer 镜像，IP 为示例值；所有命令使用本文默认端口，
执行前应确认这些端口可用。

```bash
IMAGE='your-registry/your-omniinfer-image:tag'
DATA_DIR=/srv/mooncake/rl-embeddings-dev/data
mkdir -p "$DATA_DIR"

docker run -d --name mooncake-standalone \
  --runtime=runc --network=host --restart=unless-stopped --stop-timeout=75 \
  --workdir /workspace/omniinfer \
  --mount "type=bind,src=$DATA_DIR,dst=/var/lib/mooncake" \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  --entrypoint python3 "$IMAGE" tools/scripts/start_mooncake.py start \
  --mode standalone --hosts 10.0.0.11 --cluster-id rl-embeddings-dev \
  --data-dir /var/lib/mooncake --pool-mib 4096 \
  --master-bin /opt/mooncake-py311/mooncake/mooncake_master \
  --dependency-timeout 90

docker exec -w /workspace/omniinfer mooncake-standalone python3 tools/scripts/start_mooncake.py check \
  --mode standalone --hosts 10.0.0.11 --cluster-id rl-embeddings-dev \
  --local-host 10.0.0.11
```

上述 `check` 在存储主机验证读写；还应按下文在实际 RL/推理主机执行。`docker exec`
启动的新 Python 进程也会由该入口完成环境初始化。HA 使用相同镜像和 Python 入口，
在三台主机分别提供独立数据目录，按下文设置相同的三个
`--hosts`、cluster-id 及各自的 `--node-rank`，并行启动。

`--master-bin` 可选；示例直接指定该镜像已有的原生 Master，避免通过 Python CLI 入口加载。
省略时使用当前 Python 环境的 `mooncake.cli`。直接执行原生 Master 仍需要正确的动态库路径，
provider 和客户端仍需要可用的 Python wheel。
自动初始化只覆盖该入口及其子进程。独立的 RL/推理应用仍需通过各自的镜像入口激活依赖；
`client-config` 生成的环境文件只包含连接配置，不负责初始化这些应用的 CANN 环境。
后续章节的 Python 命令均在容器内的 `/workspace/omniinfer` 仓库根目录执行。
通过 `docker run` 启动时，将相同参数传给上面的 Python 入口。

## 普通部署：单节点启动与接入

假设存储主机是 `10.0.0.11`，在该主机的容器内执行：

```bash
python3 tools/scripts/start_mooncake.py start \
  --mode standalone \
  --hosts 10.0.0.11 \
  --cluster-id rl-embeddings-dev \
  --data-dir /var/lib/mooncake \
  --pool-mib 4096
```

普通模式的 `--hosts` 只接收一个地址，`--node-rank` 可省略，默认为 `0`。
启动器在本机托管 etcd、单 Master 和常驻存储池，并处理子进程异常重启。

在每台 RL/推理主机的容器内生成本机配置并验收。`--local-host` 始终填写**当前客户端主机**的
可达地址，下面假设是 `10.0.0.21`；`--model` 改为实际 served model 名称：

```bash
python3 tools/scripts/start_mooncake.py client-config \
  --mode standalone \
  --hosts 10.0.0.11 \
  --cluster-id rl-embeddings-dev \
  --local-host 10.0.0.21 \
  --model /models/pangu \
  --output-dir /job/feature-store

source /job/feature-store/feature-store.env

python3 tools/scripts/start_mooncake.py check \
  --mode standalone \
  --hosts 10.0.0.11 \
  --cluster-id rl-embeddings-dev \
  --local-host 10.0.0.21
```

生成配置的 `master_server_addr` 是 `10.0.0.11:50061`，`metadata_server` 是
`etcd://10.0.0.11:2379`，`replica_num=1`。`check` 验证单 etcd 健康、完整副本位于配置的
存储主机，并验证 writer 退出后新客户端仍能读回。

普通模式适合开发联调或可以接受单点故障的训练任务，没有主节点或数据副本容灾。
provider/主机重启后，RL 需要重新发布丢失的 embedding；进程自动重启和 etcd 持久盘
不等于 payload 持久化。切换模式时使用新的 cluster-id 和数据目录，重新生成客户端配置
并发布所需特征。

## 高可用部署：三个节点启动

假设三个存储节点是 `10.0.0.11 / 10.0.0.12 / 10.0.0.13`，在第一台的容器内执行：

```bash
python3 tools/scripts/start_mooncake.py start \
  --hosts 10.0.0.11 10.0.0.12 10.0.0.13 \
  --cluster-id rl-embeddings-v1 \
  --node-rank 0 \
  --data-dir /var/lib/mooncake \
  --pool-mib 4096
```

第二、三台分别改为 `--node-rank 1`、`--node-rank 2`。
**hosts 的顺序、cluster-id、端口设置必须一致**，路径在每台主机是独立的本地目录。
三个启动进程需同时运行，先启动一个节点后等待它就绪再启动第二个节点会因无 quorum 超时。

## 通用参数与生命周期

可先加 `--dry-run` 打印计划，不需要安装 native 依赖、不写文件也不启动服务；
加 `--check-only` 检查本机依赖、Master 参数、etcd 库及端口占用后退出。
该预检不能替代实际启动后的 etcd 后端和网络验收。

`--dependency-timeout` 默认为 90 秒，分别限制 Master `--help` 参数检查和 native 依赖
导入子进程；两项检查各自拥有完整超时，不共用一个总时限。NPU wheel 首次加载较慢时可按需
增大该值。它不限制环境激活脚本的执行时间，不控制服务就绪等待，也不替代 `--startup-timeout`。

### 大对象读取租约

`start --kv-lease-ttl-ms` 设置 Mooncake 对象读取租约，单位是正整数毫秒。
普通模式和 HA 模式都支持；例如需使用 60 秒租约时，将上面 `start` 命令的末尾改为：

```bash
  --pool-mib 4096 \
  --kv-lease-ttl-ms 60000
```

脚本将该值传给 Master 的 `--default_kv_lease_ttl=60000`。省略此选项时不显式传参，
沿用 Mooncake 0.3.13.post1 的默认 `10000` ms，或已有 `--master-bin` wrapper 设置。
此选项仅用于 `start`，无需传给 `client-config` 或 `check`。修改后需重启 Master 才会生效，
不支持热更新；HA 的每台候选 Master 必须使用相同值。

该租约用于保护读取中的对象，与 etcd 租约、FeatureStore 的 `io_timeout_seconds` 分别配置，
也不是对象的存活 TTL：到期不代表自动删除对象。读取传输超过租约可能报 `LEASE_EXPIRED`，
单独增大客户端 I/O 超时无法延长 Master 发出的租约。

2026-09-18 的同机 standalone TCP 测试中，2560 MiB 对象由 8 个 rank 并发 GET，
10 秒租约下出现 `LEASE_EXPIRED`；调整为 60 秒后完成 1 次预热和 5 次正式测试。
该值应根据部署中最大并发 GET 的尾延迟加裕量设置，不把这次结果作为所有负载的固定推荐值。
更长租约会延后正常删除或淘汰时的内存回收，应结合存储池容量和实际并发读延迟调整。

### 端口与进程管理

默认端口：

| 端口 | 用途 | 参数 |
| --- | --- | --- |
| 2379 | etcd 客户端，存储节点与 RL/推理节点均需可达 | `--etcd-port` |
| 2380 | etcd peer 监听，HA 模式用于 member 之间通信 | `--peer-port` |
| 50061 | 各 Master RPC | `--master-port` |
| 9013 | 本机 Master 指标，仅监听 127.0.0.1 | `--metrics-port` |
| 15200–15299 | provider 的 TCP RPC 范围 | `--provider-ports` |
| 15300–15399 | RL/推理/probe 客户端 TCP RPC 范围 | `--client-ports` |

TCP 数据传输所需连接应按 Mooncake 和所在网络策略放通，不能只放通 etcd。
高并发 TP/DP 进程数较多时增大客户端端口范围；脚本拒绝端口范围重叠。

每个节点的目录包含：

- `deployment.json`：固定部署模式、集群、成员顺序及本机 rank，重启时核对身份。
- `etcd/`：持久化元数据，正常重启保留，不自动重置或强制重建集群。
- `etcd.log / master.log / provider.log`：追加的子进程日志，长期运行应由宿主配置日志轮转。
- `node-ready.json`：本机 etcd 健康、Master 就绪（HA 模式也允许 standby）、provider 完成注册的标志。
  它不等于完成客户端读写或 HA 数据冗余验收；启动训练前执行 `check`。
- `launcher.lock`：防止同一目录被两个启动器同时接管。

启动器接收 SIGINT/SIGTERM 后清理自己启动的进程组，移除就绪标记，保留 etcd 数据。
子进程异常退出会退避重启，五分钟内超过 `--max-restarts`（默认 10）则整体失败退出；
首次启动超过 `--startup-timeout`（默认 180 秒）未就绪也会失败。运行中 etcd 不可用或 HA 失去 quorum
会撤下就绪标记并等待恢复，不循环删除数据。启动器被 SIGKILL 后可能留下标记，
不要仅凭静态文件判断存活；容器/cgroup 应负责整个进程组的生命周期。
更换 etcd 成员或磁盘丢失需要按 etcd 成员恢复流程处理，不能把新的空目录当作恢复。

## 给 RL 和推理生成本机配置

在每台 RL/推理主机的容器内运行一次，`--local-host` 填**当前客户端主机**的可达地址。
`--model` 使用任务真实 served model 名称（RL 若使用模型绝对路径作为名称，照原值填写）。
下面为默认 HA 模式示例；普通模式使用上文的 `--mode standalone` 和单个存储地址：

```bash
python3 tools/scripts/start_mooncake.py client-config \
  --hosts 10.0.0.11 10.0.0.12 10.0.0.13 \
  --cluster-id rl-embeddings-v1 \
  --local-host 10.0.0.21 \
  --model /models/pangu \
  --output-dir /job/feature-store

source /job/feature-store/feature-store.env
```

生成的 `feature-store.json` 可直接交给
[Omni-NPU](https://gitee.com/omniai/omni-npu) 的 `FeatureWriter.from_config()` 和推理端 `OMNI_FEATURE_STORE_CONFIG`；
HA 模式请求 `replica_num=2`，普通模式请求 `replica_num=1`。
如果已有 JSON，合并 `mooncake` 字段并保留原有预算等设置，
不要直接覆盖已有调优配置。完整训练参数、EC 配置及生成接口见
Omni-NPU 仓库中的 `docs/feature_store.md`，RL/推理环境需安装包含该 feature store 功能的 Omni-NPU 版本。

`feature-store.env` 同时设置 `MC_STORE_CLUSTER_ID`、`MC_METADATA_CLUSTER_ID`、
`MC_TCP_BIND_ADDRESS`、`MC_FORCE_TCP=1`、客户端端口及 `OMNI_FEATURE_STORE_CONFIG`。
此处的 `source feature-store.env` 用于给 RL/推理进程加载连接配置，仍需执行；入口的自动
CANN/Mooncake 初始化不会修改调用者的 shell 或其他应用进程的环境。
**须在导入 Mooncake、启动 Ray/API/worker 前加载**。已有 Ray 集群需在各节点启动时注入，
不能只修改 head 或 API 进程的环境。各节点 JSON 路径可以一致，但内容/本地 bind 地址必须
分别生成；不要把 head 的 IP 通过全局 runtime_env 广播。这里不设置 KV 通道的
`MOONCAKE_CONFIG_PATH`。

启动器也为 provider 和 `check` 子进程统一设置 `MC_FORCE_TCP=1`，确保 NPU wheel 的
本部署走 CPU/TCP。该变量作用于整个进程中的所有 Mooncake engine，不限于这个 feature
store；不要将它写成通用镜像的全局 `ENV`。同一进程若还使用 Mooncake NPU/RDMA 通道，
应将 TCP feature-store 与该通道隔离到不同进程或部署，并分别配置环境，不能直接全局加载
这里生成的 TCP 环境。

HA 模式下，旧的外部 RL 配置渲染器若仅允许单端点字符串，需要先适配分号分隔的 etcd 地址再使用，
不能为了通过旧校验删掉另外两个 endpoint。Omni-NPU feature store 客户端接受生成的多端点字符串。

RL 调用保持原样：

```python
from omni_npu.feature_store import FeatureWriter

with FeatureWriter.from_config() as writer:
    writer.put_embedding(mm_hash, final_embedding)
```

## 启动验收

在实际 RL/推理主机的容器内执行。下面为 HA 模式示例，普通模式命令见上文：

```bash
python3 tools/scripts/start_mooncake.py check \
  --hosts 10.0.0.11 10.0.0.12 10.0.0.13 \
  --cluster-id rl-embeddings-v1 \
  --local-host 10.0.0.21
```

`check` 在 HA 模式下检查至少两个 etcd endpoint 健康，在普通模式下检查唯一的 etcd endpoint。
随后用随机的专属 key 写入 1 MiB 数据，检查其完整内存副本位于配置中的存储主机：HA 至少
两个独立主机，普通模式一个主机。关闭 writer，再用新的纯客户端读回
逐字节比较，最后只删除该探针 key。native 初始化/关闭也受父进程超时控制。
返回非零时不要启动新训练任务；HA 模式仅剩两个 etcd endpoint 会报告 degraded。
每个客户端网络故障域都应运行该探针。它不主动终止任何集群节点，也不替代故障注入测试。

## 高可用的边界

- 三成员 etcd + 三候选 Master 可容忍一个仲裁成员/主节点失效；Master 的 OpLog 保存
  控制面元数据，**不保存 embedding payload**。选主/重连期间仍可能有暂态读写错误，
  应由 RL 处理失败并重新发布需要的特征，不承诺零中断或自动重试业务请求。
- 生成配置请求双副本，且每台主机只有一个 provider，正常分配时副本位于不同主机。
  Mooncake 0.3.13.post1 的分配是 best-effort：可用容量不足时 PUT 可能仅写入一个副本
  仍返回成功。部署探针检查的是当次探针对象；`FeatureWriter` 不对每个对象强制验数，
  也不自动补齐故障后缺少的副本。已有配置省略 `replica_num` 时仍保持原来的单副本行为。
- DRAM 数据会随 provider/主机重启丢失。单节点故障后的旧数据依赖存活副本；
  三个 provider 全停、数据被逐出或仅剩副本失效时，RL 必须重新写入。etcd 持久盘和
  自动重启不能提供 payload 持久化。本入口不配置 SSD offload 或 Master snapshot。
- 启动器托管子进程崩溃恢复，不替代机器/容器编排。上线前还应在隔离集群逐台停机，
  验证同一存活客户端恢复、故障前 key 和故障后新 key 完整性以及恢复时间；
  不把本地脚本测试当作真实三机 HA 验收。

参数依据固定版本官方
[部署指南](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/docs/source/deployment/mooncake-store-deployment-guide.md)、
[客户端绑定](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-integration/store/store_py.cpp)、
[副本分配实现](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-store/src/master_service.cpp)。


## FeatureStore 客户端容量和 I/O 配置

通过 `start_mooncake.py client-config` 生成 `feature-store.json` 和配套的
`feature-store.env`。容量与 I/O 参数如下，省略时使用表中的默认值：

| client-config 参数 | JSON 字段 | 默认值 | 范围或约束 |
|---|---|---|---|
| `--max-payload-mib` | `max_payload_bytes` | 32 MiB | 正整数，单个 embedding 上限 |
| `--max-inflight-mib` | `max_inflight_bytes` | 1024 MiB | 至少为单个 embedding 上限的 4 倍 |
| `--max-items` | `max_items` | 8 | 1..128，单次请求的特征引用数量 |
| `--io-workers` | `io_workers` | 4 | 1..64，特征 I/O 线程数 |
| `--max-inflight-requests` | `max_inflight_requests` | 8 | 1..1024，特征准备并发请求数 |
| `--io-timeout-seconds` | `io_timeout_seconds` | 30.0 秒 | 大于 0 且不超过 600，支持小数 |
| `--buffer-mib` | `mooncake.local_buffer_size` | 64 MiB | 至少 32 MiB，本地传输缓冲 |

前六项参数仅适用于 `client-config`；`--buffer-mib` 也用于其他操作。
MiB 参数写入 JSON 时会乘以 `1024 ** 2`，转换为字节。
非法参数会在写文件前被拒绝，不覆盖现有配置。
`max_inflight_requests` 控制特征准备并发，与推理客户端的 HTTP 并发数不同。
这些客户端配置不改变 Mooncake 的存储池大小；存储池由 `start --pool-mib` 配置。

例如，在客户端 `10.0.0.21` 上生成连接存储主机 `10.0.0.11` 的配置：

```bash
python3 tools/scripts/start_mooncake.py client-config \
  --mode standalone --hosts 10.0.0.11 \
  --cluster-id rl-embeddings-v1 --local-host 10.0.0.21 \
  --model pangu_ultra_moe --buffer-mib 256 \
  --max-payload-mib 64 --max-inflight-mib 2048 \
  --max-items 8 --io-workers 4 --max-inflight-requests 8 \
  --io-timeout-seconds 30 \
  --output-dir /data/feature-store/client
```

请替换为实际存储地址、客户端地址、集群标识和模型名称。
生成结果中的容量与 I/O 字段为：

```json
{
  "max_payload_bytes": 67108864,
  "max_inflight_bytes": 2147483648,
  "max_items": 8,
  "io_workers": 4,
  "max_inflight_requests": 8,
  "io_timeout_seconds": 30.0
}
```

运行客户端或推理服务之前，在同一启动环境中加载生成的环境文件：

```bash
source /data/feature-store/client/feature-store.env
```

`client-config` 只生成文件，不启动或重启服务。重新生成会覆盖原来的配置文件，
因此每次都应传入需要保留的非默认设置。运行中的消费者不会自动重新加载文件：
修改容量或 I/O 配置后，重新启动相应的 P 推理服务；独立客户端重新执行命令即可。
仅修改这些客户端配置，无需重启 Mooncake 存储服务。
使用新参数前，确认执行的生成脚本已更新到支持这些参数的版本。
