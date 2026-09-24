# omni_cli 使用文档

## 安装

requirements：

```
ansible>=8.0
pyyaml>=6.0
chardet
build
```

安装cli工具：
```bash
cd omniinfer/tools
python -m build --wheel      # 无wheel包，源码安装
cd dist
pip install omni_cli-0.4.0-py3-none-any.whl --force-reinstall --no-build-isolation --no-deps
```

## 快速开始

omni_cli 会在命令执行路径创建配置文件，并依赖该配置文件。不同用户、模型请在不同路径下执行。add_node第一次执行时会创建该配置文件。以下内容以deepseek 1p1d 三机部署为例，使用镜像容器中已经安装的omniinfer和vllm，依赖于容器中/workspace/omniinfer的拉起脚本。

1. 添加节点

   ```bash
   omni_cli add_node --role C --name c0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像 --model_name deepseek
   omni_cli add_node --role P --name p0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像 --model_name deepseek
   omni_cli add_node --role D --name d0 --host_ip 本机IP --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像 --model_name deepseek
   omni_cli add_node --role D --name d1 --host_ip 本机IP --master-node 主D节点 --ssh_private_key_file 本机SSH_KEY文件 --docker_image_id 镜像 --model_name deepseek
   ```

2. 修改配置

   omni_cli具有推断配置的功能。因为不同模型、不同场景参数差别巨大，需要先配置模型路径，omni_cli依据路径名称推断模型，并提供部分场景默认配置。

   ```bash
   # 配置所有节点，支持按照Role、节点名称分别配置，参见下文omni_cli cfg详细介绍
   omni_cli cfg --set all env --MODEL_PATH 模型路径
   omni_cli cfg --set all env --LOG_PATH 日志路径
   omni_cli cfg --set all --container_name_prefix  容器名称前缀 # 依据前缀生成所有容器名称，如 prefix_prefill_p0
   ```

3. 服务拉起

   ```bash
   omni_cli run_docker  # 如需使用指定容器，必须跳过此步
   omni_cli start       # 服务拉起，包括ranktable生成、global proxy拉起
   omni_cli stop        # 服务停止
   ```

## 命令列表

   ```bash
   # 节点管理
   omni_cli ls
   omni_cli add_node
   omni_cli rm_node

   # 配置管理
   omni_cli cfg
   omni_cli inspect

   # 服务管理
   omni_cli run_docker
   omni_cli start
   omni_cli stop

   # 开发工具
   omni_cli upgrade     # 开发中
   omni_cli collect_log # 开发中
   omni_cli install_code
   omni_cli sync_code
   ```

### 节点管理

1. omni_cli ls

   功能：输出当前的节点，包括role, name, IP。

   使用示例：

   ```bash
   omni_cli ls
   ```

   效果展示：

   ```
   Role  | Name  | IP Address    | ascend_rt_visible_devices
   ---------------------------------------------------------
   C     | c0    | 127.0.0.1     | N/A
   D     | d0    | 127.0.0.2     | 0,1,2,3,4,5,6,7
   D     | d1    | 127.0.0.3     | 8,9,10,11,12,13,14,15
   P     | p0    | 127.0.0.1     | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
   ---------------------------------------------------------

   ```
2. omni_cli add_node

   功能：在当前工作区初始化或修改server_profiles.yml文件并向其添加节点。后续支持动态扩缩容

   参数说明：

   > --help: 帮助信息
   >
   > --role：节点角色（必需），从 ['P', 'D', 'C'] 中选择，分别代表Prefill，Decode 和 Global Proxy
   >
   > --name：节点名称（必需）
   >
   > --host_ip：主机IP地址（必需）
   >
   > --ssh_private_key_file：SSH私钥文件路径（必需）
   >
   > --docker_image_id：Docker镜像ID（必需）
   >
   > --user：SSH用户名，默认为 "root"
   >
   > --ssh_common_args：SSH通用参数，默认为"-o StrictHostKeyChecking=no -o IdentitiesOnly=yes"
   >
   >  --master-node：主节点名称，默认为当前添加的节点自己，用于多机组服务。主节点无需设置，从节点需要配置主节点，例如，d0.
   >   
   > --ascend_rt_visible_devices：实例需要用到的卡号, 默认为"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
   > 
   > --model_name：部署模型的名称, 默认指定为"deepseek", 每个模型有一套默认的环境变量与参数配置

   使用示例：

   ```bash
   omni_cli add_node --role P --name p0 --host_ip 127.0.0.1 \
   --ssh_private_key_file /path/to/my/key --docker_image_id myapp:latest \
   --ascend_rt_visible_devices 0,1,2,3 --model_name deepseek
   ```

3. omni_cli rm_node

   功能：从配置文件中删除指定节点。后续支持动态扩缩容

   参数说明：

   > --help: 帮助信息
   >
   > --role：节点角色（必需），从 ['P', 'D', 'C'] 中选择
   >
   > --name：要删除的节点名称（必需）
   >
   > --deploy_path：配置文件路径，默认为 $pwd/server_profiles.yml

    使用示例：

   ```bash
   omni_cli rm_node --role P --name p0
   ```

### 配置管理

1. omni_cli cfg

   命令执行时会检测当前配置文件中MODEL_PATH是否已经配置，如果没有配置，则拒绝生效，要求先配置MODEL_PATH。

   功能：修改或者删除节点的环境变量和参数。可以对所有节点或者某组节点批量修改或删除配置，也可以只针对某个节点修改或删除配置。

   使用示例：
   ```bash

   # NAME 可以是 all(所有节点)， p/d/c(按照Role)，p0/d0/c0(按照名称)

   # 配置设置
   #
   # 环境变量
   omni_cli cfg --set NAME env --MODEL_PATH /data/models/DeepSeek-R1-w8a8-fusion
   # 一般 args
   omni_cli cfg --set NAME args --num-servers 32
   # extra-args
   omni_cli cfg --set NAME args --extra-args '--max-num-batched-tokens 30000 --max-num-seqs 32'
   # additional-config
   omni_cli cfg --set NAME args --additional-config '--graph_model_compile_config level:1'
   # 单独支持 容器名称、镜像、visible device 配置
   omni_cli cfg --set NAME --container_name_prefix docker_name # 设置容器名前缀
   omni_cli cfg --set NAME --container_name docker_name # 设置容器名
   omni_cli cfg --set NAME --DOCKER_IMAGE_ID swr.cn-east-4.myhuaweicloud.com/omni-ci/omni_infer-a3-arm:master-202508191159-daily # 设置镜像 ID
   omni_cli cfg --set NAME --ascend_rt_visible_devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 # 设置卡号

   # 配置删除
   #
   # 环境变量
   omni_cli cfg --delete NAME env --MODEL_PATH
   # 一般 args
   omni_cli cfg --delete NAME args --num-servers
   # extra-args
   omni_cli cfg --delete NAME args --extra-args '--max-num-batched-tokens --max-num-seqs'
   # additional-config
   omni_cli cfg --delete NAME args --additional-config '--graph_model_compile_config'
   # 单独支持 容器名称、镜像、visible device 配置
   omni_cli cfg --delete NAME --container_name
   omni_cli cfg --delete NAME --DOCKER_IMAGE_ID
   omni_cli cfg --delete NAME --ascend_rt_visible_devices

   ```

2. omni_cli inspect

   功能：根据名称，输出该节点所有配置信息

   参数说明：
   > name: 节点名称，可通过omni_cli ls查看

   使用示例:
   ``` bash
   omni_cli inspect p0
   ```

### 服务管理

1. omni_cli run_docker

   功能：根据 add_node、cfg配置，启动所有容器。如果已有同名容器，会将同名容器销毁，重新创建

   参数说明：

   > --help: 帮助信息
   >
   > --config_path：配置文件路径，默认为 $pwd/server_profiles.yml
   >
   > --dry-run：测试模式，只显示将要执行的操作而不实际执行

    使用示例：

   ```bash
   omni_cli run_docker --config_path test.yml
   ```

2. omni_cli start

   功能：依据指定配置文件在已有容器中拉起推理服务。普通模式会生成 ranktable；只有配置文件同时包含非空的 C、P、D 分组时才会自动拉起 global proxy。`--run-dev` 和 `--proxy-only` 的行为见下方单角色部署说明。

   参数说明：

   > --help: 帮助信息
   >
   > --skip-verify-config:  跳过配置校验，避免配置推断修改用户指定配置
   >
   > --config_path:  使用指定omni_cli配置文件拉起服务，默认为 $pwd/server_profiles.yml
   >
   > --run-dev: 开发者快速拉起，跳过 ranktable 生成和 global proxy 自动拉起。使用前需确保服务依赖的 ranktable 等文件已经准备完成；如需 Proxy，可再使用 `--proxy-only` 单独拉起。
   >    
   > --proxy-only: 仅拉起global proxy。

   使用示例：

   ```bash
   omni_cli start --config_path server_profiles.yml
   ```

#### 使用 P-only / D-only YAML

单角色部署用于 OmniCache 场景；容器中的 connector 和 OX 需包含 dynamic OX（!2441）改动。P/D 可以分别启动，无需在本次 YAML 中配置另一侧节点。

##### 1. 配置模板格式

`tools/omni_cli/template/` 新增以下参考模板：

| 模板 | 分组与主机 | 含义 |
|---|---|---|
| `cli_glm5_moe_w4a8c16_a2_p_only_p0.yaml` | `all.children.P.hosts` 下为 p0–p3；可选 `C.hosts.c0` | 一组四机 Prefill，p0 为 master；C 默认与 p0 同节点、同容器 |
| `cli_glm5_moe_w4a8c16_a2_d_only_d0.yaml` | 仅 `all.children.D.hosts`，包含 d0–d3 | 一组四机 Decode，d0 为 master；不需要 P/C 分组 |

模板中的 `127.0.0.x`、镜像、SSH 密钥、模型/日志路径、网卡和端口都需要按实际环境修改。每个主机保留完整的 `env`、`args`、容器及 SSH 配置。`ansible_host` 是当前节点地址，`host_ip` 和 `env.HOST_IP` 指向本组 master；P 侧 `LOCAL_HOST_IP` 是当前节点地址，`P_NODE_PORT_LIST` 是本组四台机器的 IP 列表（逗号分隔，不带端口）。`master_node` 填写 YAML 中的 master 主机名。

文件名中的 p0/d0 表示实例；主机名 p0–p3/d0–d3 表示该实例内的四台机器。部署第二个 P 实例时，复制 P 模板并修改节点及相关配置即可。模板参数仅供参考，不可原样用于实际部署。

##### 2. 使用 `--config_path` 指定操作范围

`--config_path FILE` 让命令读取指定 YAML，只处理其中配置的主机；省略时使用当前目录的 `server_profiles.yml`。它不是节点发现功能，不会自动读取其他 YAML 或补全另一角色。`--deploy_path` 作为旧参数继续兼容。

`start`、`stop`、`run_docker`、`sync_code`、`install_code`、`inspect`、`cfg`、`ls`、`add_node`、`rm_node` 均支持该参数。以下以 P-only 为例，D-only 使用同样流程替换配置路径：

```bash
# 创建容器；同名容器会被删除并重建
omni_cli run_docker --config_path /path/to/p_only.yml

# 可选：同步并安装代码。/path/to/code 下需有 omniinfer/ 子目录
omni_cli sync_code --code_path /path/to/code --config_path /path/to/p_only.yml
omni_cli install_code --config_path /path/to/p_only.yml

# 启动 P/D，跳过配置自动检查和修正；ranktable 等依赖须已准备好
omni_cli start --skip-verify-config --run-dev --config_path /path/to/p_only.yml

# 停止该配置的 P/D 进程，保留容器
omni_cli stop --config_path /path/to/p_only.yml
```

需要生成 ranktable 时，启动命令去掉 `--run-dev`。`--skip-verify-config` 只跳过配置检查与自动修正，不控制 Proxy；`stop` 不接受这两个参数，也不会停止 Proxy。

##### 3. `--proxy-only` 与 Proxy 启动关系

以下说明适用于实际启动模式（未使用 `--cloud-mode`）：

| 启动方式 | P/D 服务 | Proxy | ranktable |
|---|---|---|---|
| `start`（普通模式） | 启动 YAML 中的 P/D | 仅当同一 YAML 同时有非空 C、P、D 时自动启动 | 生成 |
| `start --run-dev` | 启动 YAML 中的 P/D | 不启动 | 不生成 |
| `start --proxy-only` | 不启动 | 只启动 C 中配置的 Proxy | 不生成 |
| `start --run-dev --proxy-only` | 不启动 | 同上一行，`--proxy-only` 优先 | 不生成 |

P-only 即使保留 C，普通 `start` 也不会自动启动 Proxy。需要时单独执行：

```bash
omni_cli start --proxy-only --config_path /path/to/p_only.yml
```

该 YAML 需要包含 C 和 P。没有 D 时，CLI 使用不可用的 Decode 占位 endpoint，仅用于验证 Prefill 链路，不能完成完整 P/D 推理；需要完整推理时，Proxy 使用的 YAML 必须包含实际 P/D endpoint。不同 YAML 的 endpoint 不会自动合并。重复对同一 C 容器执行 Proxy-only 会重启并替换其 nginx 配置。

3. omni_cli stop

   功能：在已配置容器中，将服务停止。注：不会停止配置容器之外的服务

   参数说明:

   > --help: 帮助信息
   >
   > --config_path: 指定配置文件；只停止该文件中 P/D 主机对应容器内的 Python、vLLM、Ray 和 OX 进程，不停止 Proxy

   使用示例：

   ```bash
   omni_cli stop --config_path server_profiles.yml
   ```

### 开发工具

1. omni_cli upgrade

    功能：执行安装流程，将omniinfer和global proxy安装包安装进入指定容器中。

    参数说明：

   > --help: 帮助信息
   >
   > --omni_package:  需要安装的omniinfer包
   >
   > --vllm_package:  需要安装的vllm包
   >
   > --proxy_package: 需要安装的Global Proxy包

    使用示例：

   ``` bash
   omni_cli upgrade --omni_package XXXXX.whl  --vllm_package XXXXX.whl --proxy_package XXXXX.rpm
   ```

2. omni_cli collect_log

    功能：将日志从各个容器中收集到执行机路径。

    参数说明：

   > --name: 指定需要获取的日志，all, P, D, C分别代表所有日志、所有P日志、所有D日志、Proxy日志

    使用示例：

   ```bash
   omni_cli collect_log --name all
   ```

3. omni_cli install_code

    功能：执行安装流程，将容器中/workspace/omniinfer代码采用editable模式安装，替换容器中自带omni_infer和vllm。对于同一个容器，只需执行一次，后续仅同步代码 sync_code 即可

    参数说明：

   > --help: 帮助信息
   >
   > --config_path：配置文件路径，默认为 $pwd/server_profiles.yml；旧参数 `--deploy_path` 继续兼容
   >
   > --dry_run：测试模式，只显示将要执行的操作而不实际执行

    使用示例：

   ```bash
   omni_cli install_code --config_path /path/to/server_profiles.yml
   ```

4. omni_cli sync_code --code_path

    功能：将指定路径源代码同步到所有节点对应的容器中的/workspace/omniinfer。

    参数说明：

   > --help: 帮助信息
   >
   > --code_path：要同步的代码路径（必需）。
   >
   > --config_path：配置文件路径，默认为 $pwd/server_profiles.yml；旧参数 `--deploy_path` 继续兼容
   >
   > --dry_run：测试模式，只显示将要执行的操作而不实际执行

    使用示例：

   ```bash
   omni_cli sync_code \
     --config_path /path/to/server_profiles.yml \
     --code_path /path/to/my/code   # 路径会被补全为 /path/to/my/code/omniinfer
   ```

## 开发计划
1. cfg 支持按照特性级别配置，并检查单个服务内配置是否冲突、互斥
2. 当前设计所有模型、场景默认/最优配置在同一个文件tools/omni_cli/configs/default_profiles.yml，后续要将不同模型最优配置拆分为独立文件
3. 拉起服务调用start命令，需要校验多个服务间的配置是否有差异，防止不同服务特性配置不一致
4. start命令要支持单独拉P/D/C
5. ls命令需要返回容器状态
6. start命令要检查服务是否拉起，以进度条方式展示
7. 完成 omni_cli upgrade
8. 完成 omni_cli collect_log
