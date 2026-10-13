# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""把"合成音频"能力补进 vLLM 的 random-mm 数据集。

迁移来源：omni-npu 仓 `src/omni_npu/vllm_patches/patches/common/patch_bench.py`
（分支 develop @ cf2bf5a0）。

**只迁了 `RandomMultiModalAudioDatasetPatch` 这一个类** —— 原文件里还有
BenchEndpointPatch / CustomDatasetPatch / BenchServePatch 三个，本仓暂不需要，
故未迁入（文件名用 `_audio` 后缀标明只含音频这一部分，避免与上游同名文件混淆）。

背景：vLLM 原生 `random-mm` 只有 image/video 两个模态（按 bucket 末位判定），
把音频桶 `(0, min_duration_ms, max_duration_ms)` 误判成 video；而原生的
`custom_audio` 数据集又不支持图片。所以"3K图 + 2.5K文本 + 10s语音"这个口径
必须靠本补丁给 random-mm 补上 audio 模态。

注册名保持 `RandomMultiModalAudioDatasetPatch` —— 压测侧
`vllm_bench.sh --mode omni` 通过 `OMNI_NPU_VLLM_PATCHES` 指名的就是它。

**为什么选原位补丁而非"继承 + 注册新数据集名"**：
1. CLI/文档兼容：压测命令保持 --dataset-name random-mm 与上游文档一致；
   注册新名字（如 random-mm-omni）需同步改所有 SOP/脚本/文档，
   漏改会静默退回到无音频口径；
2. 改动面最小：只覆盖 3 个方法（map_config_to_modality /
   generate_synthetic_audio / generate_mm_item）；继承则要拷贝整个数据集类
   （数百行）只为加 audio 模态，后续跟上游 diff 的同步成本高；
3. 与仓内机制一致：vllm_patches 是本仓对 vLLM 行为适配的统一载体
   （patches/common 下已有数十个同形态补丁），按 OMNI_NPU_PATCHES_DIR 投递、
   OMNI_NPU_VLLM_PATCHES 可全量或指名开关，import 期应用并有 patch applied
   日志，回归与排查路径统一；
4. 作用域安全：只影响 bench 随机数据集类，不碰 serving 热路径。
（承认"继承+注册"在上游化维度更解耦——若社区倾向该形态可切换，见日落条件 2。）

**日落条件**（任一满足即删除本补丁）：
1. vLLM 上游 random-mm 原生支持 audio 模态
   （bucket 判定不再只按末位 1/>1 区分 image/video）；
2. 本仓 bench 侧改用"继承 + 注册显式数据集名"形态（如 random-mm-omni），
   把模态扩展做成显式数据集——这是比原位补丁更解耦的长期形态。

**适配的 vLLM 版本**：
- 已在容器镜像内 vLLM 0.25.1 实测验证
  （ai-infra-infer-...-omni_develop-vllm-202609210014:0.0.1）：
  修复前 --mode omni 发压启动即抛
  ValueError: Modality video is not in limit_mm_per_prompt: dict_keys(['image','audio'])；
  修复后补丁注册/应用日志齐全、omni 口径（3K图+2.5K文本+10s音频）正常发压；
  纯音频请求（--limit-mm '{"image":0,"audio":1}'）Median TTFT 295ms，
  证明音频真实进入请求而非空跑；
- 版本耦合点：vllm/benchmarks/datasets.py 的 RandomMultiModalDataset 类
  （构造参数与 map_config_to_modality / generate_mm_item /
  generate_synthetic_audio 方法形态）。vLLM 升级复核清单：
  ① 启动日志中补丁 registered/applied 记录；② 一次 --mode omni 冒烟发压；
- 不影响 serving 路径，仅作用于 vllm bench 的随机数据集。
"""

from typing import Any

import base64
import io
import wave

import numpy as np
from vllm.benchmarks.datasets import RandomMultiModalDataset

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


@register_patch("RandomMultiModalAudioDatasetPatch", RandomMultiModalDataset)
class RandomMultiModalAudioDatasetPatch(VLLMPatch):
    """Add synthetic variable-length audio to vLLM's ``random-mm`` dataset.

    Audio reuses the existing three-integer bucket syntax to avoid changing
    vLLM's CLI parser.  A bucket ``(0, min_duration_ms, max_duration_ms)``
    represents one 16 kHz mono WAV whose duration is sampled uniformly from
    the inclusive interval.  For example::

        --random-mm-limit-mm_per-prompt '{"audio": 1}'
        --random-mm_bucket-config '{(0, 1000, 30000): 1.0}'
    """

    _attr_names_to_apply = [
        "map_config_to_modality",
        "generate_synthetic_audio",
        "generate_mm_item",
    ]

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        if config[0] == 0:
            if config[1] <= 0 or config[2] < config[1]:
                raise ValueError(
                    "Invalid audio bucket. Expected "
                    "(0, min_duration_ms, max_duration_ms) with "
                    "0 < min_duration_ms <= max_duration_ms, got "
                    f"{config}."
                )
            return "audio"
        if config[-1] == 1:
            return "image"
        if config[-1] > 1:
            return "video"
        raise ValueError(f"Invalid multimodal item configuration: {config}")

    def generate_synthetic_audio(
        self,
        min_duration_ms: int,
        max_duration_ms: int,
        sample_rate: int = 16000,
    ) -> dict[str, Any]:
        """Generate random-length mono PCM16 WAV as an OpenAI audio URL."""
        duration_ms = int(
            self._rng.integers(min_duration_ms, max_duration_ms + 1)
        )
        num_samples = max(1, round(sample_rate * duration_ms / 1000))
        # Low-amplitude noise avoids clipping while exercising real audio I/O.
        samples = self._rng.integers(
            -4096, 4097, size=num_samples, dtype=np.int16
        )
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

        encoded = base64.b64encode(wav_buffer.getvalue()).decode("ascii")
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
        }

    def generate_mm_item(
        self,
        mm_item_config: tuple[int, int, int],
    ) -> dict[str, Any]:
        modality = self.map_config_to_modality(mm_item_config)
        if modality == "audio":
            return self.generate_synthetic_audio(
                mm_item_config[1], mm_item_config[2]
            )
        if modality == "image":
            from vllm.benchmarks.datasets import process_image

            return process_image(
                self.generate_synthetic_image(
                    mm_item_config[1], mm_item_config[0]
                )
            )
        if modality == "video":
            from vllm.benchmarks.datasets import process_video

            return process_video(
                self.generate_synthetic_video(
                    mm_item_config[1],
                    mm_item_config[0],
                    mm_item_config[2],
                )
            )
        raise ValueError(f"Invalid multimodal item configuration: {mm_item_config}")
