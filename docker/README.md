## 安装前准备

构建镜像前，先按架构准备好本地安装包，并放到 `docker/copy_data/`。  
各包的下载入口、筛选项、文件名匹配规则见 **[copy_data/README.md](./copy_data/README.md)**。

当前文档示例组合（**aarch64 + Python 3.11**）：

| 包 | 示例文件 | 下载地址 |
| --- | --- | --- |
| CANN Toolkit | `Ascend-cann-toolkit_8.3.T1_linux-aarch64.run` | [昇腾 CANN 社区版](https://www.hiascend.com/zh/developer/download/community/result?module=cann) |
| CANN Kernels | `Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run` | 同上 |
| torch_npu | `torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl` | [Ascend/pytorch Releases](https://gitcode.com/Ascend/pytorch/releases) |
| torchvision | `torchvision-0.24.0-cp311-cp311-linux_aarch64.whl` | [PyTorch torchvision wheel](https://download.pytorch.org/whl/torchvision/) |
| 自定义算子 / 离线源码 | 算子源码 + 编包脚本等 | 自行准备，放到 `docker/codes/`，详见 [codes/README.md](./codes/README.md) |

### CANN 包怎么下

1. 打开 [CANN 社区版下载页](https://www.hiascend.com/zh/developer/download/community/result?module=cann)。
2. 选择与 torch_npu 配套的 CANN 版本（当前示例 `8.3.T1`；以 [torch_npu 发行说明](https://gitcode.com/Ascend/pytorch/releases) 中的「配套版本」为准）。
3. 按芯片选型：A3/910C 下 `Atlas-A3-cann-kernels_*.run`；A2/910B 下对应的 A2 kernels。
4. 架构选 `linux-aarch64` 或 `linux-x86_64`，与 `--arch` 一致。
5. **整包模式（默认 `--cann-install-mode whole`）**至少需要：
   - `Ascend-cann-toolkit_<version>_linux-<arch>.run`
   - 对应芯片的 `*cann-kernels*.run` 或 `*ops*.run`
   - 可选：`Ascend-cann-nnal_*.run`
6. 将 `.run` 文件放到 `docker/copy_data/`，保持官网原始文件名。

分包安装（`--cann-install-mode split`）需要的文件列表见 [copy_data/README.md](./copy_data/README.md)。

### torch_npu 怎么下

1. 打开 [https://gitcode.com/Ascend/pytorch/releases](https://gitcode.com/Ascend/pytorch/releases)。
2. 选择与目标 PyTorch 版本对应的 Release（当前示例为 torch 2.9.0，选名称含 `pytorch2.9.0` 的发行版，下载 `torch_npu-2.9.0.post2-...whl`）。
3. Assets 中按 **Python 版本 + 架构** 选 wheel：
   - Python 3.11 + aarch64：`torch_npu-*-cp311-*-aarch64.whl`
   - Python 3.11 + x86_64：`torch_npu-*-cp311-*-x86_64.whl`
4. 同一架构只放一个 torch_npu wheel 到 `docker/copy_data/`。

### 注意

1. **aarch64** 只需提供对应版本的 `torch_npu` 和 `torchvision`，安装 `torch_npu` 时会自动下载匹配的 `torch`；**x86_64 必须自行提供**同版本的 `torch-*-x86_64.whl`。
2. `torchvision` 与 `torch` 有严格版本对应。当前示例：`torchvision-0.24.0` 依赖 `torch-2.9.0`。
3. wheel / `.run` 等本地安装包统一放在 **`docker/copy_data/`**（相对本仓库的 `docker` 目录）。
4. 自定义算子源码、编包脚本、预编译算子包、以及无法 git 拉取的代码，统一放在 **`docker/codes/`**，目录约定见 [codes/README.md](./codes/README.md)。
5. 镜像内默认 Python 为 `3.11.12`（`--python-version`），因此 wheel 一般选 `cp311`。

准备完成后，`docker/copy_data/` 在 aarch64 整包模式下至少应类似：

```text
docker/copy_data/
├── Ascend-cann-toolkit_8.3.T1_linux-aarch64.run
├── Atlas-A3-cann-kernels_8.3.T1_linux-aarch64.run
├── torch_npu-2.9.0.post2-cp311-cp311-manylinux_2_28_aarch64.whl
└── torchvision-0.24.0-cp311-cp311-linux_aarch64.whl
```

## 镜像分层

下面说明脚本中涉及的镜像分层（层次越高依赖越低层），便于理解 L1 / L2 构建的职责和包含的软件包。

- 基础镜像 (`BASE_IMAGE`)
	- 说明：系统级基础镜像，通常由用户预先提供或基于官方发行版构建（脚本默认 `test-infer-base:0.1`）。
	- 要求：已安装与 `--python-version` 匹配的 Python，并能使用 `yum` / `pip`。

- L1 镜像（由 `Dockerfile.base` 生成，脚本变量名为 `L1_IMAGE`）
	- 说明：在 `BASE_IMAGE` 上安装 CANN、`torch_npu`、`torchvision`（以及 x86_64 下的 `torch`）和内核相关组件。
	- 作用：为推理与硬件加速提供运行时环境（Ascend runtime、toolkit、kernels、`torch_npu` 等），作为可复用的中间层供 L2 依赖。
	- 输入包来源：`docker/copy_data/`。

- L2 镜像（由 `Dockerfile.omniinfer` 生成，脚本变量名为 `L2_IMAGE`）
	- 说明：应用层 / apiserver 镜像，基于 L1 构建，包含 omniinfer 服务、Python 依赖、模型服务与用户级工具。
	- 作用：打包并暴露模型推理服务（apiserver / omniinfer）、管理脚本、自定义算子以及运行时第三方 Python 包，直接用于运行容器。
	- 输入来源：`docker/codes/`、`docker/requirements/`，以及 `--branch` 指定的 omniinfer 源码。

## 镜像一键构建

`docker_build_run.sh` 支持一键构建：可单独制作 L1 / L2，支持 CANN 整包或分包安装，支持自定义算子（需提供算子包和对应 build 脚本）。

在 `docker/` 目录下执行：

```bash
cd docker
bash docker_build_run.sh [options]
```

### docker_build_run 脚本参数说明

`docker_build_run.sh` 中参数的各个字段说明如下：

| 字段 | 含义 |
| :--- | :--- |
| `--arch <arch>` | 目标构建平台架构：`aarch64` 或 `x86_64`。默认：`aarch64` |
| `--proxy <proxy>` | 构建时 HTTP/HTTPS 代理。默认空；示例：`http://proxy.example.com:8080` |
| `--no-proxy <hosts>` | 构建时不经过代理的主机列表，以逗号分隔。默认空 |
| `--hugging-face-proxy <proxy>` | 运行时容器下载模型用的 HTTP/HTTPS 代理。默认空，与 `--proxy` 独立 |
| `--pip-index-url <url>` | Docker 构建中 pip 索引地址。默认：`https://mirrors.huaweicloud.com/repository/pypi/simple` |
| `--pip-trusted-host <host>` | pip 信任的主机，默认：`mirrors.huaweicloud.com`；更换源时同步调整，可传 `""` 清空 |
| `--rustup-dist-server <url>` | Rust 工具链下载地址，默认：`https://rsproxy.cn` |
| `--rustup-update-root <url>` | Rustup 更新地址，默认：`https://rsproxy.cn/rustup` |
| `--rustup-init-url <url>` | Rustup 安装脚本的 HTTPS 地址，默认：`https://rsproxy.cn/rustup-init.sh` |
| `--cargo-registry-url <url>` | Cargo registry 地址，默认：`sparse+https://rsproxy.cn/index/`；sparse 地址末尾保留 `/` |
| `--model-name <name>` | 运行时要下载或启动的模型名（例如 `"Qwen/Qwen2.5-0.5B"`），会传给容器。 |
| `--cann-install-mode <split\|whole>` | L1 构建时 CANN 安装方式：`whole`（整包，默认）或 `split`（分包）。 |
| `--base-image <image>` | 构建 L1 时使用的系统基础镜像 tag。默认：`test-infer-base:0.1` |
| `--L1-image <image>` | L1 构建完成后的镜像 tag。默认：`test-infer-meddle:0.1` |
| `--L2-image <image>` | L2 构建完成后的镜像 tag。默认：`test-infer-omniinfer:0.1` |
| `--branch <tag>` | 打进镜像的 omniinfer 源码分支或 tag。默认：`master` |
| `--custom-ops <ops>` | 要编进镜像的自定义算子（逗号分隔的脚本名，不含 `.sh`）。默认空。 |
| `--npu-platform <platform>` | 编自定义算子时的硬件平台：`910B` 或 `910C`。默认：`910C` |
| `--python-version <version>` | 构建时使用的 Python 版本。默认：`3.11.12`（对应 wheel 的 `cp311`） |
| `--start-server <True\|False>` | 构建结束后是否启动容器执行 `start_server.sh`。默认：`True` |
| `--build-target <L1\|L2\|both\|skip>` | 构建目标：`L1` 只构建 `Dockerfile.base`；`L2` 只构建 `Dockerfile.omniinfer`；`both` 先 L1 再 L2（默认）；`skip` 跳过构建。 |
| `--vllm-version <version>` | 安装的 vLLM 版本。默认：`v0.14.0` |
| `--install-modules <modules>` | 要安装的 omniinfer 模块，逗号分隔。默认：`omni-proxy`；`omni-npu` 已由 omniinfer 自身集成，无需再传入。 |
| `--skip-pull <True\|False>` | 是否向 omniinfer `build/build.sh` 传递 `--skip-pull`。默认：`false` |
| `--build-for-roma <True\|False>` | 是否继续构建 Roma 镜像。默认：`false` |
| `--roma-image <image>` | Roma 镜像 tag。默认：`test-infer-ROMA:0.1` |

脚本仅通过命令行参数覆盖配置，不读取同名环境变量作为配置入口。不传参数时使用
上述默认值；脚本内部会将参数传入 Docker 构建环境。旧的 `--build-network` 参数已移除，
旧命令应删除该参数，需要自定义 Rust/Cargo 源时使用上面的四个地址参数。

`--proxy` 同时设置构建环境的大小写 HTTP/HTTPS 代理变量，`--no-proxy` 同时设置
大小写 NO_PROXY；未传入时显式清空代理，包括 ALL_PROXY，以覆盖 Docker 客户端的默认
代理配置。运行代理仅由 `--hugging-face-proxy` 设置，不会自动继承构建代理。
代理使用 Docker 预定义构建参数传递，不在 Dockerfile 中声明为 `ARG` 或写入 `ENV`；
构建脚本也不会主动打印完整代理地址。

这些构建代理参数用于构建步骤中的下载；基础镜像拉取所需的代理由 Docker daemon／
BuildKit 自身的网络配置管理。

### 命令执行

下面给出若干常见示例——把占位符替换为实际镜像 tag、模型名、自定义算子包以及构建目标。

**示例 1 — 全量构建（默认 both），并指定自定义镜像 tag**：

```bash
bash docker_build_run.sh \
	--arch aarch64 \
	--base-image new-infer-base:0.1 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image new-infer-omniinfer:0.1 \
	--model-name "Qwen/Qwen2.5-0.5B" \
	--branch master
```

该示例会串行构建 L1 和 L2：先执行 `Dockerfile.base`，基础镜像为 `new-infer-base:0.1`（需自行提供），输出 L1：`new-infer-meddle:0.1`；再执行 `Dockerfile.omniinfer`，输入为该 L1，输出 L2：`new-infer-omniinfer:0.1`。镜像内 omniinfer 代码取 `master` 分支。

**示例 2 — 仅构建 L1（只构建 Dockerfile.base），并使用 split 模式安装 CANN**：

```bash
bash docker_build_run.sh --build-target L1 --cann-install-mode split \
	--pip-index-url "https://mirrors.huaweicloud.com/repository/pypi/simple" \
	--pip-trusted-host "mirrors.huaweicloud.com" \
	--L1-image new-infer-meddle:0.1
```

该示例会跳过 `Dockerfile.omniinfer`，只输出 L1（`new-infer-meddle:0.1`）。若不指定 `--L1-image`，则使用默认 tag `test-infer-meddle:0.1`。若需要指定基础镜像，请同时提供 `--base-image`，否则会使用默认的 `test-infer-base:0.1`。分包所需文件见 [copy_data/README.md](./copy_data/README.md)。

**示例 3 — 仅构建 L2（跳过 base），加入自定义算子包并指定源码版本**：

```bash
bash docker_build_run.sh --build-target L2 \
	--L1-image new-infer-meddle:0.1 \
	--L2-image test-infer-omniinfer:latest \
	--branch "dev_v1.0.0" \
	--custom-ops build_omni-ops_packages \
	--npu-platform 910C \
	--start-server False
```

该示例会跳过 `Dockerfile.base`，只输出 L2（`test-infer-omniinfer:latest`）。**必须**提供已存在的 `--L1-image`，否则会使用默认 `test-infer-meddle:0.1` 导致构建失败。镜像构建完后不启动容器。

关于 `--custom-ops`：传入 `build_omni-ops_packages` 会执行
`docker/codes/ops_build_images/build_omni-ops_packages.sh`，并传入
`--npu-platform <910B|910C>`，在当前 L2 构建容器内编译并安装 omni-ops 推理和
训练算子。源码目录结构、预编译 `.run`/`.whl` 命名、HiXL / moxing / 离线 vLLM 等可选文件，见
**[codes/README.md](./codes/README.md)**。

**示例 4 — 通过代理构建，构建完成后不启动服务**：

```bash
bash docker_build_run.sh \
  --base-image new-infer-base:0.1 \
  --proxy "http://proxy.example.com:8080" \
  --no-proxy "localhost,127.0.0.1,.example.com" \
  --start-server False
```

将示例代理替换成实际可用的代理地址。若构建结束后还要通过代理下载模型，应另行传入
`--hugging-face-proxy "http://runtime-proxy.example.com:8080"` 并设置 `--start-server True`。

**示例 5 — 更换 pip 源**：

```bash
bash docker_build_run.sh \
  --build-target L2 \
  --L1-image new-infer-meddle:0.1 \
  --pip-index-url "https://pypi.org/simple" \
  --pip-trusted-host "" \
  --start-server False
```

此例使用 PyPI 并清空 trusted host，使用正常的 HTTPS 证书校验。网络配置会传给内置
omni-ops 编译入口，L1、L2 和算子编译不会各自切回内网 pip 源。

**示例 6 — 更换 Rust/Cargo 下载地址**：

下面的 `mirror.example.com` 是占位域名，应替换成实际部署的镜像站；四个路径应与该
站点的 Rustup、工具链和 Cargo 服务对应。一般公网构建无需传入这些参数。

```bash
bash docker_build_run.sh \
  --build-target L2 \
  --L1-image new-infer-meddle:0.1 \
  --rustup-dist-server "https://mirror.example.com/rustup-static" \
  --rustup-update-root "https://mirror.example.com/rustup-static/rustup" \
  --rustup-init-url "https://mirror.example.com/rustup-static/rustup-init.sh" \
  --cargo-registry-url "sparse+https://mirror.example.com/cargo/" \
  --start-server False
```

## 构建环境的 DNS 配置

DNS 用于把下载站域名解析为 IP 地址。镜像构建默认使用构建环境提供的 DNS，Dockerfile
不修改 `/etc/resolv.conf`，也不包含内部 DNS 地址。网络正常时无需额外设置。
DNS 与下载源地址是两项配置：更换下载源不会自动更换 DNS。

以下配置均位于构建机器上，不需要在本仓库新增文件。`<DNS_IP_1>`、`<DNS_IP_2>` 和
`<SEARCH_DOMAIN>` 均为占位符，应用配置前须替换为构建网络中实际可用的值。

### 普通 Docker 构建：默认 docker driver

使用 `docker buildx ls` 查看构建器及其 driver。一键脚本使用 `docker build`，默认走
Docker 自带的构建器。Linux Docker Engine 的 DNS 可在宿主机 `/etc/docker/daemon.json`
中配置；将以下字段合并到已有 JSON 中，保留其他配置：

```json
{
  "dns": ["<DNS_IP_1>", "<DNS_IP_2>"]
}
```

在允许重启 Docker 的时段执行 `sudo systemctl restart docker` 使配置生效；这会影响
该宿主机上的容器。Docker Desktop 用户可在 Settings → Docker Engine 中合并配置并
应用重启。Rootless Docker 通常使用 `~/.config/docker/daemon.json`，由对应用户重启其
Docker 服务。

随后重新执行原来的 `bash docker_build_run.sh ...` 命令，无需传入 DNS 参数。
参考 [Docker DNS 配置说明](https://docs.docker.com/engine/daemon/troubleshoot/#specify-dns-servers-for-docker)。

### 独立 BuildKit／buildx 构建器

独立运行的 BuildKit 有自己的 DNS 配置，不应只修改宿主机 Docker daemon 配置。
其配置通常位于 `/etc/buildkit/buildkitd.toml`，rootless 模式通常位于
`~/.config/buildkit/buildkitd.toml`；也可能由运维启动参数指定其他路径。在该配置中合并：

```toml
[dns]
  nameservers = ["<DNS_IP_1>", "<DNS_IP_2>"]
  # 仅在需要补全内部短域名时设置，否则省略下面一行。
  searchDomains = ["<SEARCH_DOMAIN>"]
```

直接运行的 BuildKit 需由运维重启对应实例。使用 buildx 的 `docker-container` driver
时，在宿主机准备好上述配置，然后创建加载该配置的构建器，例如：

```bash
docker buildx create \
  --name omniinfer-builder \
  --driver docker-container \
  --buildkitd-config /etc/buildkit/buildkitd.toml \
  --bootstrap
```

一键脚本不会自动切换到这个独立构建器。需要使用它时，应显式执行 `docker buildx build`
并指定 `--builder`。例如，在 `docker/` 目录中使用已有 L1 镜像构建 L2：

```bash
docker buildx build \
  --builder omniinfer-builder \
  --load \
  --build-arg BASE_IMAGE="<构建器能够获取的L1镜像>" \
  --build-arg BRANCH=master \
  --target omininfer_openai \
  -f Dockerfile.omniinfer \
  -t new-infer-omniinfer:0.1 .
```

`docker-container` 构建器有独立的镜像存储，基础镜像应能从它可访问的镜像仓库拉取。
自定义源通过同名 `--build-arg` 传入，例如 `--build-arg RUSTUP_DIST_SERVER=...`；代理
可传入 `--build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...`，有需要时也传入小写变量
及 NO_PROXY。参考 [BuildKit 配置方式](https://docs.docker.com/build/buildkit/configure/)
和 [DNS 配置字段](https://docs.docker.com/build/buildkit/toml-configuration/)。

### 如何检查与排查

先确认宿主机能解析下载站，再确认构建步骤也能解析。使用本地已有、包含 `getent` 的
L1 镜像，可执行下面的最小构建来检查默认构建器的 DNS，无需创建测试 Dockerfile：

```bash
printf 'FROM %s\nRUN getent hosts rsproxy.cn\n' "<已存在的L1镜像>" \
  | docker build --no-cache -
```

独立构建器检查时，将最后一段替换成
`docker buildx build --builder omniinfer-builder --no-cache -`，并使用该构建器能获取的
基础镜像。使用代理时，还需确认代理服务器域名能够解析。

`Could not resolve host` 通常需要检查上述 DNS 配置；连接超时还需检查网络路由和代理。
`docker run --dns` 只设置启动后的容器，不能修复之前的镜像构建。

## 容器启动

当前镜像默认将 `ENTRYPOINT` 设置为 `start_server.sh`，因此若只想进 shell，启动容器时需要加 `--entrypoint=bash`。也可以自行覆盖 `ENTRYPOINT`。

一键脚本在 `--start-server True`（默认）时，会用类似下面的方式拉起服务：

```bash
docker run --rm -it --shm-size=500g \
	--net=host --privileged=true \
	--device=/dev/davinci_manager \
	--device=/dev/hisi_hdc \
	--device=/dev/devmm_svm \
	-e PORT=8301 \
	-e ASCEND_RT_VISIBLE_DEVICES=1 \
	-e HTTP_PROXY="${HUGGING_FACE_PROXY}" \
	-e http_proxy="${HUGGING_FACE_PROXY}" \
	-e HTTPS_PROXY="${HUGGING_FACE_PROXY}" \
	-e https_proxy="${HUGGING_FACE_PROXY}" \
	-e NO_PROXY= -e no_proxy= -e ALL_PROXY= -e all_proxy= \
	-e MODEL_NAME="${MODEL_NAME}" \
	<L2镜像> \
	--model "${MODEL_NAME}"
```

只进入容器、不拉服务：

```bash
docker run --rm -it --privileged --net=host \
	--device=/dev/davinci_manager \
	--device=/dev/hisi_hdc \
	--device=/dev/devmm_svm \
	--entrypoint=bash \
	<L2镜像>
```

若启动后的容器需要单独指定 DNS，可在手动运行时添加参数：

```bash
docker run --rm -it \
  --dns "<DNS_IP_1>" --dns "<DNS_IP_2>" \
  --entrypoint=bash \
  "<L2镜像>"
```

该例只用于进入 shell；运行推理服务时需同时保留上面的 NPU 设备等参数。
