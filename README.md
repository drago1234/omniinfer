# Omni-Infer：基于昇腾的超大规模MoE模型推理加速技术

中文 | [View English](./README_en.md)

*社区新闻(更多活动可参考[社区活动日历](https://gitee.com/omniai/community/blob/master/calender.md))* 🔥
- [2025/08] Omni社区与LFAI&Data基金会亚太区域用户组，启动线上技术交流活动**Omni-Talk**，可参考社区活动的[详细安排](https://gitee.com/omniai/community/tree/master/events#%E7%A4%BE%E5%8C%BAwebinar)
- [2025/08] 社区正在筹划与LFAI&Data基金会亚太区域用户组，以及腾讯公司团队一起，于8月30日下午在深圳举办**vLLM Shenzhen Meetup**，可移步访问[活动详情](https://gitee.com/omniai/community/tree/master/events#%E7%A4%BE%E5%8C%BA%E7%AC%AC%E4%BA%8C%E6%AC%A1meetupvllm-shenzhen-meetup)
- [2025/09] 社区将积极参与[GOSIM中国峰会](https://gitee.com/omniai/community/tree/master/events#%E4%B8%9A%E7%95%8C%E4%BC%9A%E8%AE%AE%E5%8F%82%E4%B8%8E)

<details>
<summary>往期活动</summary>

- [2025/08] 社区将积极参与[CCF开源大会](https://mp.weixin.qq.com/s/NOJNP5FPzJj3ipKnVNVfSA)
- [2025/07] 社区正在筹划于7月28日下午在上海举办**首次线下Meetup**，可移步访问[活动详情](https://gitee.com/omniai/community/tree/master/events#%E7%A4%BE%E5%8C%BAmeetup)
- [2025/07] 社区的MTP SIG计划在7月的第三周召开**首次SIG例会**，可参考SIG会议的[详细安排](https://gitee.com/omniai/community/blob/master/sigs/sig-mtp.md)
- [2025/07] 社区的PD分离、EP负载均衡、算子与图三个SIG计划在7月的第二周召开**首次SIG例会**，有兴趣与会的开发者请见三个SIG分别的会议安排：[PD分离SIG会议安排](https://gitee.com/omniai/community/blob/master/sigs/sig-pd-seperation.md),[EP负载均衡SIG会议安排](https://gitee.com/omniai/community/blob/master/sigs/sig-ep-placement.md),[算子与图SIG会议安排](https://gitee.com/omniai/community/blob/master/sigs/sig-ops-and-graph.md).
- [2025/07] 2025年7月5日，在苏州举办的OpenInfra基金会Meetup将有Omni-infer议题分享，详情请见[社区活动页面](https://gitee.com/omniai/community/blob/master/events/README.md)
</details>


Omni-Infer是一套专为昇腾硬件平台定制的强大推理加速工具集，完全兼容业界目前主流的开源大模型推理框架（比如vLLM等），旨在提供高性能、企业级推理能力，具备原生支持且功能集持续扩展。


## 核心特性

- **企业级低延迟P/D调度**：支持xPyD调度及大规模分布式PD部署的横向扩展，确保最低延迟。详情请参考[Global Proxy设计](omni/accelerators/sched/global_proxy/README.md)。
- **请求级负载均衡**：针对所有序列长度优化预填充（prefill）和解码（decode）阶段，实现最大吞吐量与低延迟。
- **优化的MoE专家部署**：支持EP144/EP288配置的大规模混合专家（Mixture of Experts, MoE）模型。
- **MoE专家负载均衡**：具备分层非均匀冗余和近实时动态专家放置功能，提升资源利用效率。详情请参考[OmniPlacement设计](omni/accelerators/placement/README.md)。
- **高级注意力机制优化**：专为LLM、MLLM和MoE模型定制，增强性能与可扩展性。

## 开源社区

有关Omni-Infer社区运作、活动、治理相关的信息，请访问我们的[社区运作仓库](https://gitee.com/omniai/community)

## High-Level 架构图

![image](docs/figures/omni_infer_high_level_arch.png)

## 快速开始

PD分离快速部署示例请参考[指南](docs/omni_infer_quick_start.md)。如需将Omni_Infer集成到项目中，请参考[安装指南](docs/omni_infer_installation_guide.md)和[文档](docs/)获取详细的设置说明和API参考。

Mooncake embedding 存储池的普通部署、高可用部署及 RL/推理客户端配置，请参考 [Mooncake 部署指南](docs/mooncake_deployment.md)。

## 贡献指南

我们欢迎您为Omni_Infer贡献代码！请查看[贡献指南](./CONTRIBUTION.md)，并通过[Gitee Issues](https://gitee.com/omniai/omniinfer/issues/new?issue%5Bassignee_id%5D=0&issue%5Bmilestone_id%5D=0)提交拉取请求或问题。

## 许可证

Omni_Infer基于[MIT许可证](LICENSE)发布。