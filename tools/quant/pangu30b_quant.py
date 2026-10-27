#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#
# =============================================================================
# Pangu 30B 系列 W8A8 量化 —— 单文件工具
# =============================================================================
#
# 【一条命令跑完所需量化】
#
#   cd <项目根>/tools/quant
#   setsid nohup python -u pangu30b_quant.py quant \
#       --src <bf16目录> --dst <输出目录> > /tmp/quant.log 2>&1 < /dev/null &
#
#   默认量化「标准 30B 五件套」：
#     attn（qkv_proj / o_proj）+ 路由专家 + 共享专家 + dense MLP(L0/L1) + ViT
#
# 【子命令】
#   quant   从 bf16 全量量化            （--src/--dst/--groups/--dry-run）
#   more    给已有 W8A8 产物补量化       （--src/--dst/--groups/--module-re/--apply）
#   verify  核对产物（结构 + dtype + ignore + 精度 + 体积）
#   survey  勘察 ckpt 或产物结构
#   scope   离线自检量化范围（不需要 torch，秒级）
#
# 【常用】
#   # 只看计划，不写盘
#   python pangu30b_quant.py quant  --src <bf16> --dst <out> --dry-run
#   python pangu30b_quant.py more   --src <W8A8> --dst <out> --groups visual
#   # 核对（--ref 可给 bf16 源，也可给「该组尚未量化」的旧产物）
#   python pangu30b_quant.py verify --ref <参考> --dst <产物>
#
# =============================================================================
# 【量化范围：五组】（唯一事实来源，两个量化路径共用）
# =============================================================================
#   attn            model.layers.<i>.self_attn.{qkv_proj,o_proj}        L×2
#   routed_experts  model.layers.<i>.mlp.experts.<j>.{gate,up,down}_proj  (L-d)×E×3
#   shared_experts  model.layers.<i>.mlp.shared_experts.{gate,up,down}_proj (L-d)×3
#   dense_mlp       model.layers.<i>.mlp.{gate,up,down}_proj（仅前 d 层）  d×3
#   visual          visual.blocks.<i>.{attn.qkv,attn.proj,mlp.up_proj,mlp.down_proj}  26×4
#   合计 = L×2 + (L-d)×E×3 + (L-d)×3 + d×3 + 26×4
#     37 层/384 专家/26 block/d=2  ->  40609   （已与真产物 weight_scale 条数逐条吻合）
#     40 层(含 MTP 3 层)/同上      ->  44080
#
# ⚠️ 三组 MLP 的张量后缀完全相同（都是 `...gate_proj.weight`），只能靠**路径中段**
#    区分（`mlp.` / `mlp.experts.N.` / `mlp.shared_experts.`）。所以正则必须是完整
#    的，不能用宽后缀。
# ⚠️ 每条正则只能匹配 **2-D 的 .weight**。norm / bias 是 1-D，命中会撞上
#    weight_quant() 的 `assert dim() == 2` 而中止。
#
# 【刻意不量化】lm_head / embed_tokens / 全部 norm / mhc_module.phi /
#   branch_alpha,beta,norm_gamma / mlp.gate / e_score_correction_bias /
#   visual 的 norm1,norm2 与 patch_embed.proj / merger.* / vision_projection.fc1 /
#   audio_tower.* / MTP 专有（eh_proj, enorm, hnorm, shared_head.*）
#
# =============================================================================
# 【⚠️ 关于 config.json 的 ignore —— 本项目踩过的最大坑】
# =============================================================================
#   运行时匹配用 vllm 的 `_is_equal_or_regex_match`：
#       带 `re:` 前缀 -> re.match(target[3:], value)
#       否则        -> target == value        （精确相等）
#   即：**ignore 条目要么是精确模块名，要么带 `re:` 前缀**。写裸正则不报错，
#   但一条也匹配不上 —— 运行时会把所有 Linear 按量化处理，而磁盘上那些模块是
#   BF16 且没有 weight_scale；vllm 的 default_weight_loader 只断言形状、不校验
#   dtype，于是 BF16 权重被静默塞进 int8 参数（数值≈0），**全链路零告警**，
#   服务照常拉起但输出完全乱码。
#   所以本工具的 ignore **由产物反推**（见 build_ignore_list），并做自洽性自检。
#
#   前缀写法（必须按命名空间区分）：
#       model.* / lm_head -> 裸名 + `openpangu.language_model.` + `pangu_V5.language_model.`
#       visual.*          -> 只有裸名（视觉塔不加前缀变体）
#       audio_tower.*     -> 由一条正则覆盖：re:^audio_tower\..*
# =============================================================================

import logging
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from glob import glob

logger = logging.getLogger(__name__)

INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"

# -----------------------------------------------------------------------------
# 量化组（唯一事实来源）
# -----------------------------------------------------------------------------
# 每条都是**不带锚点**的完整路径正则（自带顶层前缀）；需要时再拼 `^...$`。
GROUPS = {
    "attn": r"model\.layers\.\d+\.self_attn\.(?:qkv_proj|o_proj)",
    "routed_experts": (r"model\.layers\.\d+\.mlp\.experts\.\d+"
                       r"\.(?:gate_proj|up_proj|down_proj)"),
    "shared_experts": (r"model\.layers\.\d+\.mlp\.shared_experts"
                       r"\.(?:gate_proj|up_proj|down_proj)"),
    "dense_mlp": r"model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)",
    "visual": (r"visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)"
               r"|mlp\.(?:up_proj|down_proj))"),
}
DEFAULT_GROUPS = ("attn", "routed_experts", "dense_mlp", "shared_experts", "visual")

LLM_PREFIXES = ("openpangu.language_model.", "pangu_V5.language_model.")
AUDIO_TOWER_REGEX = r"re:^audio_tower\..*"

# 供 scope 子命令推导预期张量数用
SCOPE_DEFAULTS = {"layers": 37, "experts": 384, "dense": 2, "blocks": 26}


def parse_groups(spec):
    """把 'attn,routed_experts' 或 ['attn'] 解析成组名列表。"""
    if isinstance(spec, str):
        items = [g.strip() for g in spec.split(",")]
    else:
        items = list(spec)
    items = [g for g in items if g]
    unknown = [g for g in items if g not in GROUPS]
    if unknown:
        raise SystemExit("[FATAL] 未知量化组: %s\n        可选: %s"
                         % (", ".join(unknown), ", ".join(sorted(GROUPS))))
    return items


def build_matchers(groups):
    """返回 (full_matcher, module_matcher)。

    full_matcher  匹配 checkpoint 张量全名（含 `.weight`）  —— quant 用
    module_matcher 匹配去掉 `.weight` 的模块名             —— more 用
    """
    alts = "|".join("(?:%s)" % GROUPS[g] for g in groups)
    return (re.compile(r"^(?:" + alts + r")\.weight$"),
            re.compile(r"^(?:" + alts + r")$"))


# -----------------------------------------------------------------------------
# 项目根定位（只有 quant / more 需要量化内核，故按需解析）
# -----------------------------------------------------------------------------
PROJECT_DIR_NAMES = ("omniinfer", "omniinfer-v0.8.0")
OPTIQUANT_REL = os.path.join("tools", "quant", "python")
_OPTIQUANT_MARKER = os.path.join(OPTIQUANT_REL, "optiquant", "int8.py")
_weight_quant = None


def _find_project_root(start):
    cur = os.path.abspath(start)
    for _ in range(8):
        if os.path.basename(cur) in PROJECT_DIR_NAMES:
            return cur
        if os.path.isfile(os.path.join(cur, _OPTIQUANT_MARKER)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def resolve_project_root():
    candidates = []
    env = os.environ.get("OMNIINFER_ROOT")
    if env:
        candidates.append(env)
    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.getcwd())
    for cand in candidates:
        root = _find_project_root(cand)
        if root:
            return root
    return None


def get_weight_quant():
    """按需导入量化内核（复用已验证的 per-channel int8 kernel）。"""
    global _weight_quant
    if _weight_quant is not None:
        return _weight_quant
    root = resolve_project_root()
    if root is None:
        sys.stderr.write(
            "[FATAL] 找不到 omniinfer 项目根（需要 <项目根>/%s/optiquant）。\n"
            "        三种解决方式，任选其一：\n"
            "          1) 在项目内执行：cd <项目根>/tools/quant && python %s ...\n"
            "          2) 把本脚本放到 <项目根>/tools/quant/ 下执行\n"
            "          3) 指定环境变量：OMNIINFER_ROOT=<项目根> python %s ...\n"
            % (OPTIQUANT_REL, os.path.basename(__file__),
               os.path.basename(__file__)))
        sys.exit(2)
    optiquant_root = os.path.join(root, OPTIQUANT_REL)
    if optiquant_root not in sys.path:
        sys.path.insert(0, optiquant_root)
    from optiquant.int8 import weight_quant  # noqa: E402
    _weight_quant = weight_quant
    logger.info("[env]    项目根   : %s", root)
    logger.info("[env]    量化内核 : %s", optiquant_root)
    return _weight_quant


# -----------------------------------------------------------------------------
# ignore 名单：由产物反推
# -----------------------------------------------------------------------------


def build_ignore_list(weight_map):
    """凡是有 `.weight` 但没有同名 `.weight_scale` 的模块 = 未量化，必须进 ignore。"""
    names = set(weight_map)
    base = set()
    for key in names:
        if not key.endswith(".weight"):
            continue
        module = key[: -len(".weight")]
        if (module + ".weight_scale") in names:
            continue
        base.add(module)

    ignore = set()
    for module in base:
        if module.startswith("audio_tower."):
            continue  # 交给正则
        ignore.add(module)
        if module.startswith("model.") or module == "lm_head":
            for prefix in LLM_PREFIXES:
                ignore.add(prefix + module)
    ignore.add(AUDIO_TOWER_REGEX)
    return sorted(ignore)


def is_equal_or_regex_match(value, target):
    """复刻 vllm 的匹配语义，用于自检。"""
    if target.startswith("re:"):
        return re.match(target[3:], value) is not None
    return target == value


def runtime_variants(module):
    if module.startswith("model.") or module == "lm_head":
        return [module] + [p + module for p in LLM_PREFIXES]
    return [module]


def strip_prefix(name):
    for p in LLM_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def selftest_ignore(ignore, weight_map):
    """自检：ignore 与产物是否自洽（用运行时匹配语义复核）。"""
    names = set(weight_map)
    quantized, unquantized = set(), set()
    for key in names:
        if not key.endswith(".weight"):
            continue
        module = key[: -len(".weight")]
        (quantized if (module + ".weight_scale") in names
         else unquantized).add(module)

    def ignored(name):
        return any(is_equal_or_regex_match(name, t) for t in ignore)

    problems = []
    for module in sorted(quantized):
        if ignored(module):
            problems.append("已量化模块却被 ignore 命中: " + module)
    for module in sorted(unquantized):
        if module.startswith("audio_tower."):
            continue
        for name in runtime_variants(module):
            if not ignored(name):
                problems.append("未量化模块未被 ignore 覆盖: " + name)
                break
    return problems


# -----------------------------------------------------------------------------
# 通用小工具
# -----------------------------------------------------------------------------


def load_index(d):
    with open(os.path.join(d, INDEX_NAME), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def load_config(d):
    with open(os.path.join(d, CONFIG_NAME), encoding="utf-8") as f:
        return json.load(f)


def read_shapes(shards):
    """只读 safetensors header 拿形状，不加载张量数据。"""
    from safetensors import safe_open
    shapes = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())
    return shapes


def shards_of(d):
    return sorted(glob(os.path.join(d, "*.safetensors")))


def dir_size(p):
    tot = 0
    for dp, _, fns in os.walk(p):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if os.path.isfile(fp) and not os.path.islink(fp):
                tot += os.path.getsize(fp)
    return tot


def copy_metadata(src, dst, skip_bak=True):
    """把非权重、非备份的文件与目录复制过去。"""
    copied = []
    for entry in sorted(os.listdir(src)):
        if entry.endswith(".safetensors"):
            continue
        if skip_bak and ".bak" in entry:
            continue
        if ".before-" in entry:
            continue  # 后处理过程的中间产物，不带进产物
        s, d = os.path.join(src, entry), os.path.join(dst, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        copied.append(entry)
    return copied


def group_counts(weight_map):
    """按组统计已量化（有 weight_scale）与目标模块数。"""
    names = set(weight_map)
    out = {}
    for g, pat in GROUPS.items():
        rx = re.compile(r"^(?:" + pat + r")$")
        mods = [k[: -len(".weight")] for k in names
                if k.endswith(".weight") and rx.match(k[: -len(".weight")])]
        q = [m for m in mods if (m + ".weight_scale") in names]
        out[g] = (len(mods), len(q))
    return out


def expected_scope_counts(layers, experts, dense, blocks):
    """按公式推导各组的预期张量数。"""
    return {
        "attn": layers * 2,
        "routed_experts": (layers - dense) * experts * 3,
        "shared_experts": (layers - dense) * 3,
        "dense_mlp": dense * 3,
        "visual": blocks * 4,
    }


# =============================================================================
# 子命令：scope —— 离线自检量化范围
# =============================================================================


def cmd_scope(args):
    layers = args.layers
    experts = args.experts
    dense = args.dense
    blocks = args.blocks
    exp = expected_scope_counts(layers, experts, dense, blocks)
    total = sum(exp.values())

    logger.info("%s", "=" * 78)
    logger.info("离线量化范围自检")
    logger.info("  参数: layers=%d  experts=%d  dense=%d  blocks=%d", layers, experts, dense, blocks)
    logger.info("%s", "=" * 78)

    # 构造「标准模型」的全部模块名
    mods = []
    for i in range(layers):
        mods += ["model.layers.%d.self_attn.qkv_proj" % i,
                 "model.layers.%d.self_attn.o_proj" % i]
    for i in range(dense, layers):
        for e in range(experts):
            for p in ("gate_proj", "up_proj", "down_proj"):
                mods.append("model.layers.%d.mlp.experts.%d.%s" % (i, e, p))
        for p in ("gate_proj", "up_proj", "down_proj"):
            mods.append("model.layers.%d.mlp.shared_experts.%s" % (i, p))
    for i in range(dense):
        for p in ("gate_proj", "up_proj", "down_proj"):
            mods.append("model.layers.%d.mlp.%s" % (i, p))
    for b in range(blocks):
        mods += ["visual.blocks.%d.attn.qkv" % b,
                 "visual.blocks.%d.attn.proj" % b,
                 "visual.blocks.%d.mlp.up_proj" % b,
                 "visual.blocks.%d.mlp.down_proj" % b]

    full_re, mod_re = build_matchers(list(GROUPS))
    logger.info("")
    logger.info("[1] 逐组命中")
    bad = 0
    for g in sorted(GROUPS):
        rx = re.compile(r"^(?:" + GROUPS[g] + r")$")
        hit = [m for m in mods if rx.match(m)]
        ok = len(hit) == exp[g]
        bad += 0 if ok else 1
        logger.info("   %s%-18s 预期 %6d  命中 %6d", "OK  " if ok else "!!  ", g, exp[g], len(hit))
    hit_all = [m for m in mods if mod_re.match(m)]
    ok_all = len(hit_all) == total
    bad += 0 if ok_all else 1
    logger.info("   %s%-18s 预期 %6d  命中 %6d", "OK  " if ok_all else "!!  ", "合计", total, len(hit_all))

    logger.info("")
    logger.info("[2] 误伤检查（这些必须一个都不命中）")
    must_not = ["lm_head", "model.embed_tokens", "model.norm"]
    for i in (0, 1, layers // 2, layers - 1):
        must_not += ["model.layers.%d.mlp.gate" % i,
                     "model.layers.%d.input_layernorm" % i,
                     "model.layers.%d.pre_mlp_layernorm" % i,
                     "model.layers.%d.self_attn.k_layernorm" % i,
                     "model.layers.%d.attn_mhc_module.phi" % i,
                     "model.layers.%d.mlp_mhc_module.phi" % i]
    for b in (0, blocks - 1):
        must_not += ["visual.blocks.%d.norm1" % b, "visual.blocks.%d.norm2" % b]
    must_not += ["visual.patch_embed.proj", "visual.merger.mlp.0",
                 "visual.merger.mlp.2", "visual.vision_projection.fc1",
                 "visual.merger.ln_q",
                 "audio_tower.layers.0.self_attn.q_proj",
                 "model.merge_mhc_module.phi"]
    hurt = [m for m in must_not if mod_re.match(m)]
    if hurt:
        bad += 1
        logger.info("   !! 被误命中 %d 个:", len(hurt))
        for m in hurt[:10]:
            logger.info('%s %s', "      ", m)
    else:
        logger.info("   无 —— lm_head / router.gate / norm / MHC.phi / merger /")
        logger.info("        vision_projection / patch_embed / audio_tower / embed_tokens 均未被命中")

    logger.info("")
    logger.info("[3] 白名单正则")
    for g in sorted(GROUPS):
        logger.info("   %-16s %s", g, GROUPS[g])

    logger.info("")
    logger.info("%s", "=" * 78)
    logger.info(
        "RESULT: %s",
        "PASS —— 范围精确，可用于生产" if not bad else "FAIL —— 请先修正则",
    )
    logger.info("%s", "=" * 78)
    return 0 if not bad else 1


# =============================================================================
# 子命令：survey —— 勘察
# =============================================================================


def cmd_survey(args):
    d = os.path.abspath(args.path)
    if not os.path.isdir(d):
        raise SystemExit("[FATAL] 目录不存在: %s" % d)

    logger.info("%s", "=" * 78)
    logger.info("勘察: %s", d)
    logger.info("%s", "=" * 78)
    try:
        cfg = load_config(d)
    except Exception:
        cfg = {}
    for k in ("architectures", "model_type", "num_hidden_layers",
              "first_k_dense_replace", "n_routed_experts", "n_shared_experts",
              "num_nextn_predict_layers", "vision_config"):
        if k in cfg:
            v = cfg[k]
            if isinstance(v, dict):
                v = {kk: v[kk] for kk in ("depth", "hidden_size", "intermediate_size")
                     if kk in v} or "{...}"
            logger.info("  %-26s %s", k, v)

    try:
        wm = load_index(d)
    except Exception as exc:
        raise SystemExit("[FATAL] 读不到 %s: %s" % (INDEX_NAME, exc))

    names = set(wm)
    scale = [k for k in names if k.endswith("weight_scale")]
    logger.info("")
    logger.info("  index 条目        %d", len(wm))
    logger.info("  weight_scale      %d  (%s)", len(scale), "已量化产物" if scale else "bf16 源")
    logger.info("  顶层命名空间      %s", sorted({k.split(".")[0] for k in names}))

    layers = sorted({int(m.group(1)) for m in
                     (re.match(r"model\.layers\.(\d+)\.", k) for k in names) if m})
    if layers:
        logger.info("  model.layers 层号 %d 层: %s%s", len(layers), layers[:4], " ..." if len(layers) > 4 else "")

    shards = shards_of(d)
    logger.info("")
    logger.info("  分片 %d 个:", len(shards))
    for s in shards:
        logger.info("     %-58s %8.2f MB", os.path.basename(s), os.path.getsize(s) / 1e6)
    logger.info("  目录总大小 %.2f GB", dir_size(d) / 1e9)

    if scale:
        qc = cfg.get("quantization_config") or {}
        ig = qc.get("ignore") or []
        logger.info("")
        logger.info("  量化范围:")
        for g, (n, q) in sorted(group_counts(wm).items()):
            logger.info("     %-18s 共 %6d 个, 已量化 %6d", g, n, q)
        logger.info("  ignore 条数       %d  (%s / %s)", len(ig), qc.get("quant_method"), qc.get("quantize"))
        logger.info("")
        logger.info("  ignore 覆盖自洽性:")
        problems = selftest_ignore(ig, wm)
        if problems:
            logger.info("     FAIL(%d):", len(problems))
            for p in problems[:8]:
                logger.info('%s %s', "       ", p)
        else:
            logger.info("     PASS")
    else:
        full_re, _ = build_matchers(list(GROUPS))
        q = [k for k in wm if full_re.match(k)]
        logger.info("")
        logger.info("  按默认五组量化，将命中 %d 个张量；保留 %d 个", len(q), len(wm) - len(q))
    return 0


# =============================================================================
# 子命令：quant —— 从 bf16 全量量化
# =============================================================================


def cmd_quant(args):
    wq = get_weight_quant()
    groups = parse_groups(args.groups)
    full_re, _ = build_matchers(groups)

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)

    logger.info("[groups] 量化范围 : %s", ", ".join(groups))
    logger.info("[groups] 白名单   : %s", full_re.pattern)
    st = os.statvfs(os.path.dirname(dst) or "/")
    logger.info("[disk]  输出盘可用 %.2f GB", st.f_bavail * st.f_frsize / 1e9)

    try:
        weight_map = load_index(src)
    except Exception as exc:
        raise SystemExit("[FATAL] %s 下读不到 %s: %s" % (src, INDEX_NAME, exc))
    shards = shards_of(src)
    if args.file_count:
        shards = shards[:args.file_count]
    if not shards:
        raise SystemExit("[FATAL] %s 下没有 *.safetensors" % src)

    quant = [k for k in weight_map if full_re.match(k)]
    keep = [k for k in weight_map if not full_re.match(k)]
    logger.info("[plan]  分片=%d  张量=%d  量化=%d  保留bf16=%d", len(shards), len(weight_map), len(quant), len(keep))

    report_quant_plan(src, shards, weight_map, quant, keep, groups, full_re)

    if args.dry_run:
        if args.reference_config:
            projected = set(weight_map)
            for key in list(projected):
                if full_re.match(key):
                    projected.add(key.replace("weight", "weight_scale"))
            compare_reference(build_ignore_list(projected), args.reference_config)
        return 0

    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(dst, exist_ok=True)
    index_out = os.path.join(dst, INDEX_NAME)
    config_out = os.path.join(dst, CONFIG_NAME)

    if not (os.path.isfile(index_out) and os.path.isfile(config_out)):
        if args.allow_hf_download and args.model_name:
            from huggingface_hub import snapshot_download
            logger.info("[meta] 从 HuggingFace 下载 %s 的元数据 -> %s", args.model_name, dst)
            snapshot_download(repo_id=args.model_name,
                              ignore_patterns=["*.safetensors"],
                              local_dir=dst, local_dir_use_symlinks=False)
        else:
            logger.info("[meta] 输出目录缺少元数据，从 %s 复制非权重文件 ...", src)
            for entry in copy_metadata(src, dst):
                logger.info("       + %s", entry)
    else:
        logger.info("[meta] 元数据已存在，跳过复制")

    torch.set_default_dtype(torch.bfloat16)
    new_wm, stat = {}, {"quantized": 0, "kept": 0}

    shard_to_keys = {}
    for key, fn in weight_map.items():
        shard_to_keys.setdefault(os.path.basename(fn), []).append(key)

    for base in sorted(shard_to_keys):
        keys = shard_to_keys[base]
        src_path = os.path.join(src, base)
        qk = [k for k in keys if full_re.match(k)]

        if not qk:  # 无量化目标的分片（如音频塔/视觉投影的独立分片）原样复制
            dst_path = os.path.join(dst, base)
            if not os.path.isfile(dst_path):
                shutil.copy2(src_path, dst_path)
                logger.info("[copy ] %-46s %.2f GB（无量化目标，原样复制）", base, os.path.getsize(src_path) / 1e9)
            else:
                logger.info("[copy ] %-46s （已存在，跳过）", base)
            for key in keys:
                new_wm[key] = base
            stat["kept"] += len(keys)
            continue

        out_name = base.replace("model", args.quant_prefix, 1)
        logger.info("[shard] %s -> %s  （量化 %d / 共 %d）", base, out_name, len(qk), len(keys))
        sd = load_file(src_path, device=args.device)
        out = {}
        for name, tensor in sd.items():
            if full_re.match(name):
                if tensor.dim() != 2:
                    raise RuntimeError("白名单张量维度异常（应为 2D）: %s %s"
                                       % (name, tuple(tensor.shape)))
                q, s = wq(tensor)
                out[name] = q
                out[name.replace("weight", "weight_scale")] = s
                stat["quantized"] += 1
            else:
                out[name] = tensor
                stat["kept"] += 1
        del sd
        for key in out:
            new_wm[key] = out_name
        save_file(out, os.path.join(dst, out_name))
        del out

    with open(index_out, encoding="utf-8") as f:
        model_index = json.load(f)
    model_index["weight_map"] = new_wm
    with open(index_out, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    ignore = build_ignore_list(new_wm)
    problems = selftest_ignore(ignore, new_wm)
    logger.info("")
    logger.info("[ignore] 未量化模块 %d 个；自洽性自检：%s", len(ignore), "PASS" if not problems else "FAIL(%d)" % len(problems))
    for p in problems[:10]:
        logger.info("   !! %s", p)
    if problems:
        raise SystemExit("[FATAL] ignore 与产物不一致，已中止写入 config.json")

    if args.reference_config:
        if not compare_reference(ignore, args.reference_config):
            raise SystemExit(
                "[FATAL] 生成的 ignore 与参考产物不一致，已中止写入 config.json。\n"
                "        若这是有意为之（结构确实不同），去掉 --reference-config 重跑。")

    cfg = load_config(dst)
    cfg["quantization_config"] = build_quant_config(ignore)
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    logger.info("")
    logger.info("[done] 量化=%d  保留bf16=%d", stat["quantized"], stat["kept"])
    logger.info("[done] 权重     -> %s", dst)
    logger.info("[done] 索引     -> %s", index_out)
    logger.info("[done] 量化配置 -> %s", config_out)
    logger.info("[done] 目录体积 %.2f GB", dir_size(dst) / 1e9)
    return 0


def report_quant_plan(src, shards, weight_map, quant, keep, groups, full_re):
    logger.info("")
    logger.info("%s", "=" * 78)
    logger.info("量化计划 —— 只读 header，不加载权重")
    logger.info("%s", "=" * 78)

    rx_layer = re.compile(r"\.layers\.\d+\.")
    rx_exp = re.compile(r"\.experts\.\d+\.")
    buckets = {}
    for name in quant:
        key = rx_exp.sub(".experts.M.", rx_layer.sub(".layers.N.", name))
        buckets[key] = buckets.get(key, 0) + 1
    logger.info("")
    logger.info("[将量化为 int8]  共 %d 个", len(quant))
    for key in sorted(buckets, key=lambda k: -buckets[k]):
        logger.info("   %7d  %s", buckets[key], key)

    shard_q = {}
    for name in quant:
        shard_q.setdefault(os.path.basename(weight_map[name]), []).append(name)
    logger.info("")
    logger.info("[分片分布]  共 %d 个分片", len(shards))
    for base in sorted(set(os.path.basename(s) for s in shards)):
        n = len(shard_q.get(base, []))
        logger.info("   %-58s %6d 个量化张量%s", base, n, "（无目标，将原样复制）" if n == 0 else "")

    try:
        shapes = read_shapes(shards)
    except Exception as exc:  # noqa: BLE001
        logger.warning("\n[形状统计] 跳过：%s", exc)
        shapes = None
    if shapes:
        def nparam(ns):
            return sum(math.prod(shapes[n]) for n in ns if n in shapes)
        pq, pk = nparam(quant), nparam(keep)
        logger.info("")
        logger.info("[参数量]  量化目标 = %.3fB   保留 bf16 = %.3fB   合计 = %.3fB", pq / 1e9, pk / 1e9, (pq + pk) / 1e9)

    n_base = int(cfg_get(src, "num_hidden_layers", 0) or 0)
    n_mtp = int(cfg_get(src, "num_nextn_predict_layers", 0) or 0)
    if n_base:
        def layer_of(name):
            m = re.match(r"model\.layers\.(\d+)\.", name)
            return int(m.group(1)) if m else -1
        base_q = [n for n in quant if 0 <= layer_of(n) < n_base]
        mtp_q = [n for n in quant if layer_of(n) >= n_base]
        logger.info("")
        logger.info("[按层拆分] num_hidden_layers=%d  num_nextn_predict_layers=%d", n_base, n_mtp)
        logger.info("   主干层 0~%d        : %d 个量化张量", n_base - 1, len(base_q))
        if mtp_q:
            logger.info(
                "   MTP 层 %d~%d       : %d 个量化张量（占 %.1f%%）",
                n_base,
                max(layer_of(n) for n in mtp_q),
                len(mtp_q),
                len(mtp_q) / len(quant) * 100,
            )

    if shapes:
        odd = sorted(n for n in shapes if len(shapes[n]) == 2 and not n.endswith(".weight"))
        logger.info("")
        logger.info("[风险提示] 2-D 但不以 .weight 结尾的张量: %d 个", len(odd))
        for n in odd[:10]:
            logger.info("      %s  %s", n, shapes[n])
        if not odd:
            logger.info("      （无，说明所有 2-D 参数都按 .weight 命名，ignore 推导无盲区）")

    projected = set(weight_map)
    for key in list(projected):
        if full_re.match(key):
            projected.add(key.replace("weight", "weight_scale"))
    ignore = build_ignore_list(projected)
    logger.info("")
    logger.info("[运行时 ignore 名单]  共 %d 个未量化模块（精确层名，非正则）", len(ignore))
    for name in ignore[:8]:
        logger.info("   %s", name)
    if len(ignore) > 8:
        logger.info("   ... 另 %d 个", len(ignore) - 8)
    problems = selftest_ignore(ignore, projected)
    logger.info("[ignore 自洽性自检]  %s", "PASS —— 与产物状态一致" if not problems else "FAIL(%d)" % len(problems))
    for p in problems[:10]:
        logger.info("     %s", p)
    logger.info("%s", "=" * 78)


def cfg_get(d, key, default=None):
    try:
        return load_config(d).get(key, default)
    except Exception:
        return default


def build_quant_config(ignore):
    """运行时量化配置。ignore 必须是精确模块名列表（或带 re: 前缀的正则）。"""
    return {
        "config_groups": {
            "group_0": {
                "format": "int-quantized",
                "input_activations": {
                    "actorder": None, "block_structure": None, "dynamic": True,
                    "group_size": None, "num_bits": 8, "observer": "memoryless",
                    "observer_kwargs": {}, "strategy": "token",
                    "symmetric": True, "type": "int",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None, "block_structure": None, "dynamic": False,
                    "group_size": None, "num_bits": 8, "observer": "minmax",
                    "observer_kwargs": {}, "strategy": "channel",
                    "symmetric": True, "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantize": "w8a8_dynamic",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def compare_reference(ignore, ref_config_path):
    with open(ref_config_path, encoding="utf-8") as f:
        ref_cfg = json.load(f)
    ref = set(ref_cfg.get("quantization_config", {}).get("ignore") or [])
    cur = set(ignore)
    logger.info("")
    logger.info("[交叉核验] 参考产物: %s", ref_config_path)
    logger.info("   参考 ignore = %d   本次 ignore = %d", len(ref), len(cur))
    logger.info("   参考有而本次没有 (%d):", len(ref - cur))
    for x in sorted(ref - cur)[:10]:
        logger.info("      - %s", x)
    logger.info("   本次有而参考没有 (%d):", len(cur - ref))
    for x in sorted(cur - ref)[:10]:
        logger.info("      + %s", x)
    if ref == cur:
        logger.info("   => 完全一致：运行时 ignore 与已验证产物同构")
    else:
        logger.info("   => 存在差异，请人工确认（结构不同则应分别验证）")
    return ref == cur


# =============================================================================
# 子命令：more —— 给已有 W8A8 产物补量化
# =============================================================================


def cmd_more(args):
    wq = get_weight_quant()
    groups = parse_groups(args.groups) if args.groups else []
    pat_strs = list(args.module_re or []) + [GROUPS[g] for g in groups]
    if not pat_strs:
        logger.info("[FATAL] 必须指定 --groups 或 --module-re")
        return 2
    mod_re = re.compile(r"^(?:" + "|".join("(?:%s)" % p for p in pat_strs) + r")$")

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    logger.info("%s", "=" * 78)
    logger.info("源目录  : %s", src)
    logger.info("目标目录: %s", dst)
    if groups:
        logger.info("模块组  : %s", ", ".join(groups))
    logger.info("匹配正则: %s", mod_re.pattern)
    logger.info("%s", "=" * 78)

    with open(os.path.join(src, INDEX_NAME), encoding="utf-8") as f:
        index = json.load(f)
    wm = index["weight_map"]
    cfg = load_config(src)
    if "quantization_config" not in cfg:
        raise SystemExit("[FATAL] 源目录不是量化产物（config.json 里没有 quantization_config）。\n"
                         "        请改用：%s quant --src ..." % os.path.basename(__file__))
    old_ignore = cfg["quantization_config"].get("ignore") or []

    targets, skipped = set(), set()
    for k in sorted(wm):
        if not k.endswith(".weight"):
            continue
        mod = k[: -len(".weight")]
        if not mod_re.match(mod):
            continue
        if (mod + ".weight_scale") in wm:
            skipped.add(mod)
            continue
        targets.add(mod)
    tkeys = sorted(m + ".weight" for m in targets)

    logger.info("")
    logger.info("[计划] 命中模块 %d 个（已是量化状态的跳过 %d 个）", len(targets), len(skipped))
    for m in sorted(targets)[:6]:
        logger.info("         %s", m)
    if len(targets) > 6:
        logger.info("         ... 另 %d 个", len(targets) - 6)
    if not targets:
        logger.info("   没有需要量化的模块，退出。")
        return 0

    shard_keys = {}
    for k in tkeys:
        shard_keys.setdefault(wm[k], []).append(k)
    all_shards = sorted(set(wm.values()))
    plain = [s for s in all_shards if s not in shard_keys]

    logger.info("")
    logger.info("[计划] 需重写分片 %d 个 / 共 %d 个；直接复制 %d 个", len(shard_keys), len(all_shards), len(plain))
    for f_, ks in sorted(shard_keys.items()):
        logger.info("   重写 %-58s %4d 个张量", f_, len(ks))
    for f_ in plain:
        logger.info("   复制 %s", f_)

    removed = [e for e in old_ignore if strip_prefix(e) in targets]
    new_ignore = [e for e in old_ignore if strip_prefix(e) not in targets]
    if args.rebuild_ignore:
        projected = set(wm)
        for k in tkeys:
            projected.add(k.replace("weight", "weight_scale"))
        new_ignore = build_ignore_list(projected)
        logger.info("")
        logger.info("[计划] ignore 由产物反推重建: %d -> %d", len(old_ignore), len(new_ignore))
    else:
        logger.info("")
        logger.info("[计划] ignore 删除 %d 条: %d -> %d", len(removed), len(old_ignore), len(new_ignore))

    st = os.statvfs(os.path.dirname(dst) or "/")
    logger.info(
        "[计划] 目标目录约需 %.2f GB，所在盘可用 %.2f GB",
        sum(os.path.getsize(os.path.join(src, s)) for s in all_shards) / 1e9,
        st.f_bavail * st.f_frsize / 1e9,
    )
    if os.path.exists(dst):
        logger.info("   !! 目标目录已存在，将被写入（不删除其中已有内容）")

    if not args.apply:
        logger.info("")
        logger.info("（计划模式，未改动任何文件；加 --apply 执行）")
        return 0

    # ---------------- 执行 ----------------
    logger.info("")
    logger.info("%s", "=" * 78)
    if not os.path.isdir(dst):
        os.makedirs(dst)
        logger.info("[执行] 新建目录 %s", dst)

    stat = {"bf16": 0, "i8": 0, "scale": 0, "tensors": len(tkeys)}
    made_tmp, done = [], {}
    try:
        logger.info("[执行] 构建新分片（只写临时文件）")
        for shard in sorted(shard_keys):
            keys = shard_keys[shard]
            logger.info("  --- %s ---", shard)
            tmp_path, problems = _build_shard(wq, src, dst, shard, keys, stat)
            made_tmp.append(tmp_path)
            if problems:
                logger.info("   !! 校验失败:")
                for p in problems[:10]:
                    logger.info("      %s", p)
                raise RuntimeError("分片校验未通过: " + shard)
            logger.info("   OK")
            done[shard] = tmp_path

        logger.info("")
        logger.info("[执行] 落定分片")
        for shard, tmp_path in sorted(done.items()):
            os.replace(tmp_path, os.path.join(dst, shard))
            made_tmp.remove(tmp_path)
            logger.info("   %s", shard)
        for shard in plain:
            dst_p = os.path.join(dst, shard)
            if os.path.exists(dst_p):
                logger.info("   复制 %s  (已存在，跳过)", shard)
                continue
            t0 = time.time()
            shutil.copy2(os.path.join(src, shard), dst_p)
            logger.info("   复制 %-58s (%.1fs)", shard, time.time() - t0)

        logger.info("")
        logger.info("[执行] 复制元数据")
        items = copy_metadata(src, dst)
        logger.info("   %d 项: %s%s", len(items), ", ".join(items[:6]), " ..." if len(items) > 6 else "")

        logger.info("")
        logger.info("[执行] 更新索引")
        for shard, keys in sorted(shard_keys.items()):
            for k in keys:
                wm[k.replace("weight", "weight_scale")] = shard
        index["weight_map"] = wm
        with open(os.path.join(dst, INDEX_NAME), "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
        logger.info("   新增 %d 条 weight_scale，索引总数 %d", len(tkeys), len(wm))

        logger.info("")
        logger.info("[执行] 更新 config.json 的 ignore")
        cfg["quantization_config"]["ignore"] = new_ignore
        with open(os.path.join(dst, CONFIG_NAME), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        logger.info("   ignore: %d -> %d", len(old_ignore), len(new_ignore))

    except Exception as exc:  # noqa: BLE001
        logger.info("")
        logger.error("[中止] 出错: %r", exc)
        for tmp in made_tmp:
            if os.path.isfile(tmp):
                os.remove(tmp)
                logger.info("   已清理 %s", os.path.basename(tmp))
        logger.error("   目标目录中未落定的分片已清理；请检查后重试。")
        return 1

    # ---------------- 终检 ----------------
    logger.info("")
    logger.info("%s", "=" * 78)
    logger.info("[终检] 从磁盘重新读取")
    from safetensors import safe_open
    wm2 = load_index(dst)
    ig2 = load_config(dst)["quantization_config"]["ignore"]
    cache, bad = {}, []
    sample = tkeys[:3] + tkeys[len(tkeys) // 2: len(tkeys) // 2 + 3] + tkeys[-3:]
    for k in sample:
        ks = k.replace("weight", "weight_scale")
        fn = wm2[k]
        f_ = cache.get(fn) or cache.setdefault(
            fn, safe_open(os.path.join(dst, fn), framework="pt"))
        dw, ds = f_.get_slice(k).get_dtype(), f_.get_slice(ks).get_dtype()
        ok = (dw == "I8" and ds == "F32")
        bad += [] if ok else [k]
        logger.info("   %-58s %-4s scale %-4s %s", k, dw, ds, "OK" if ok else "FAIL")

    logger.info("")
    problems = selftest_ignore(ig2, wm2)
    if problems:
        bad.append("selftest")
        logger.info("[终检] 自检 FAIL，共 %d 项:", len(problems))
        for p in problems[:15]:
            logger.info("   %s", p)
        logger.info("   提示：可加 --rebuild-ignore 让 ignore 由产物反推重建")
    else:
        logger.info("[终检] 自检 PASS —— ignore 与磁盘产物完全自洽")

    logger.info("")
    logger.info(
        "[统计] 目标 %d 个张量 | bf16 %.2f MB -> I8 %.2f MB + scale %.2f MB | 净省 %.2f MB",
        stat["tensors"],
        stat["bf16"] / 1e6,
        stat["i8"] / 1e6,
        stat["scale"] / 1e6,
        (stat["bf16"] - stat["i8"] - stat["scale"]) / 1e6,
    )
    st = os.statvfs(dst)
    logger.info("[统计] 新目录体积 %.2f GB | 盘余 %.2f GB", dir_size(dst) / 1e9, st.f_bavail * st.f_frsize / 1e9)
    logger.info("[终检] 结论: %s", "全部通过" if not bad else "存在问题")
    return 0 if not bad else 1


def _build_shard(wq, src_dir, dst_dir, shard, keys, stat):
    import torch
    from safetensors.torch import load_file, save_file

    src_path = os.path.join(src_dir, shard)
    tmp_path = os.path.join(dst_dir, ".tmp-" + shard)

    t0 = time.time()
    sd = load_file(src_path, device="cpu")
    logger.info("      读入 %d 个张量  (%.1fs)", len(sd), time.time() - t0)

    new, t0 = {}, time.time()
    for k, v in sd.items():
        if k in keys:
            if v.dim() != 2:
                raise RuntimeError("目标张量维度异常（应为 2D）: %s %s"
                                   % (k, tuple(v.shape)))
            q, s = wq(v)
            new[k] = q
            new[k.replace("weight", "weight_scale")] = s
            n = v.numel()
            stat["bf16"] += n * 2
            stat["i8"] += n
            stat["scale"] += s.numel() * 4
        else:
            new[k] = v
    logger.info("      量化 %d 个张量  (%.1fs)", len(keys), time.time() - t0)

    t0 = time.time()
    save_file(new, tmp_path)
    logger.info("      写出临时文件 %.2f GB  (%.1fs)", os.path.getsize(tmp_path) / 1e9, time.time() - t0)

    expect = set(sd.keys()) | {k.replace("weight", "weight_scale") for k in keys}
    problems, t0 = [], time.time()
    from safetensors import safe_open
    with safe_open(tmp_path, framework="pt") as f:
        got = set(f.keys())
        if got != expect:
            problems.append("键集合不一致  多=%s  少=%s"
                            % (sorted(got - expect)[:4], sorted(expect - got)[:4]))
        for k in keys:
            if f.get_slice(k).get_dtype() != "I8":
                problems.append("主权重不是 I8: " + k)
            ks = k.replace("weight", "weight_scale")
            if f.get_slice(ks).get_dtype() != "F32":
                problems.append("scale 不是 F32: " + ks)
        nchk = 0
        for k, v in sd.items():
            if k in keys:
                continue
            g = f.get_tensor(k)
            if (g.dtype != v.dtype or tuple(g.shape) != tuple(v.shape)
                    or not torch.equal(g, v)):
                problems.append("未改动张量发生变化: " + k)
            nchk += 1
    logger.info("      校验：%d 个未改动张量逐一比对 + %d 个新张量  (%.1fs)", nchk, len(keys), time.time() - t0)
    return tmp_path, problems


# =============================================================================
# 子命令：verify —— 核对产物
# =============================================================================


def cmd_verify(args):
    dst = os.path.abspath(args.dst)
    ref = os.path.abspath(args.ref) if args.ref else None
    if not os.path.isdir(dst):
        raise SystemExit("[FATAL] 目录不存在: %s" % dst)
    groups = parse_groups(args.groups) if args.groups else list(GROUPS)

    wm = load_index(dst)
    cfg = load_config(dst)
    qc = cfg.get("quantization_config")
    if not qc:
        raise SystemExit("[FATAL] %s 不是量化产物（config.json 里没有 quantization_config）" % dst)

    import torch
    from safetensors import safe_open

    ig = qc.get("ignore") or []
    names = set(wm)
    scale = [k for k in names if k.endswith("weight_scale")]

    logger.info("%s", "=" * 78)
    logger.info("核对: %s", dst)
    logger.info("%s", "=" * 78)
    logger.info(
        "  index 条目 %d | weight_scale %d | ignore %d | %s / %s",
        len(wm),
        len(scale),
        len(ig),
        qc.get("quant_method"),
        qc.get("quantize"),
    )

    cache = {}

    def sl(k):
        fn = wm[k]
        if fn not in cache:
            cache[fn] = safe_open(os.path.join(dst, fn), framework="pt")
        return cache[fn].get_slice(k)

    # --- 1) 逐组 dtype ---
    logger.info("")
    logger.info("[1] 逐组 dtype")
    bad_total, skipped = 0, []
    for g in sorted(groups):
        rx = re.compile(r"^(?:" + GROUPS[g] + r")$")
        mods = [k[: -len(".weight")] for k in names
                if k.endswith(".weight") and rx.match(k[: -len(".weight")])]
        with_scale = [m for m in mods if (m + ".weight_scale") in names]
        if not with_scale:
            skipped.append((g, len(mods)))
            logger.info("   %-18s 共 %5d 个 | 未量化（不在本次范围）", g, len(mods))
            continue
        bad = []
        for m in mods:
            k = m + ".weight"
            ks = m + ".weight_scale"
            if ks not in names:
                bad.append(k + " (缺 scale —— 同组内混合，需排查)")
                continue
            a, b = sl(k), sl(ks)
            if a.get_dtype() != "I8" or b.get_dtype() != "F32":
                bad.append("%s dtype=%s scale=%s" % (k, a.get_dtype(), b.get_dtype()))
        bad_total += len(bad)
        logger.info("   %-18s 共 %5d 个 | I8+F32 正确 %5d | 异常 %d", g, len(mods), len(mods) - len(bad), len(bad))
        for b in bad[:3]:
            logger.info("        !! %s", b)
    if skipped:
        logger.info("   范围外（未量化，符合预期）: %s", ", ".join("%s %d 个" % (t, n) for t, n in skipped))

    # --- 2) 不该量化的仍无 scale ---
    logger.info("")
    logger.info("[2] 不应被量化的模块是否被误量化")
    must_not = [("visual.blocks.*.norm1/norm2", r"^visual\.blocks\.\d+\.norm[12]$"),
                ("visual.patch_embed.proj", r"^visual\.patch_embed\.proj$"),
                ("visual.merger.*", r"^visual\.merger"),
                ("visual.vision_projection.*", r"^visual\.vision_projection"),
                ("mlp.gate（router）", r"^model\.layers\.\d+\.mlp\.gate$"),
                ("audio_tower.*", r"^audio_tower\.")]
    hurt = 0
    for tag, pat in must_not:
        rx = re.compile(pat)
        ms = [k[: -len(".weight")] for k in names
              if k.endswith(".weight") and rx.match(k[: -len(".weight")])]
        q = [m for m in ms if (m + ".weight_scale") in names]
        hurt += len(q)
        logger.info("   %-32s 共 %4d 个, 有 scale %3d  %s", tag, len(ms), len(q), "OK" if not q else "!! 不该有")
    bad_total += hurt

    # --- 3) ignore 自洽性 + 覆盖率 ---
    logger.info("")
    logger.info("[3] ignore 自洽性 / 覆盖率")
    problems = selftest_ignore(ig, wm)
    if problems:
        bad_total += len(problems)
        logger.info("   FAIL(%d):", len(problems))
        for p in problems[:10]:
            logger.info("      %s", p)
    else:
        logger.info("   PASS —— ignore 与磁盘产物完全自洽")
    re_entries = [e for e in ig if e.startswith("re:")]
    if re_entries:
        logger.info("   注意：ignore 含 %d 条正则项（已纳入匹配）：%s", len(re_entries), ", ".join(re_entries[:3]))

    # --- 4) 精度（若给了参考目录） ---
    rels = []
    if ref:
        logger.info("")
        logger.info("[4] 精度（以 %s 的同名张量为基准）", os.path.basename(ref))
        rwm = load_index(ref)
        quant_mods = [m for m in sorted({k[: -len(".weight")] for k in names
                                         if k.endswith(".weight")})
                      if (m + ".weight_scale") in names]
        picks = quant_mods[:2] + quant_mods[len(quant_mods) // 2:
                                            len(quant_mods) // 2 + 2] + quant_mods[-2:]
        logger.info("   %-52s %9s %11s %9s", "tensor", "rel_err", "cos(fp64)", "SQNR_dB")
        for m in picks:
            k = m + ".weight"
            ks = m + ".weight_scale"
            if k not in rwm:
                continue
            with safe_open(os.path.join(ref, rwm[k]), framework="pt") as f:
                w = f.get_tensor(k).double()
            with safe_open(os.path.join(dst, wm[k]), framework="pt") as f:
                q = f.get_tensor(k)
            with safe_open(os.path.join(dst, wm[ks]), framework="pt") as f:
                s = f.get_tensor(ks)
            dq = q.double() * s.double()
            err = dq - w
            rel = (err.norm() / w.norm()).item()
            cos = (torch.dot(dq.flatten(), w.flatten()) / (dq.norm() * w.norm())).item()
            sqnr = (20 * torch.log10(w.norm() / err.norm())).item()
            rels.append(rel)
            logger.info(
                "   %-52s %8.3f%% %11.6f %9.2f",
                m.replace("model.layers.", "L").replace("visual.blocks.", "b"),
                rel * 100,
                cos,
                sqnr,
            )
        if rels:
            logger.info("   平均相对误差 %.3f%%", sum(rels) / len(rels) * 100)
    else:
        logger.info("")
        logger.info("[4] 精度  跳过（未给 --ref）")

    # --- 5) 体积 ---
    logger.info("")
    logger.info("[5] 体积")
    if ref:
        logger.info(
            "   参考 %.2f GB  ->  产物 %.2f GB  (减少 %.2f MB)",
            dir_size(ref) / 1e9,
            dir_size(dst) / 1e9,
            (dir_size(ref) - dir_size(dst)) / 1e6,
        )
    else:
        logger.info("   产物 %.2f GB", dir_size(dst) / 1e9)

    logger.info("")
    logger.info("%s", "=" * 78)
    logger.info("RESULT: %s", "PASS" if bad_total == 0 else "FAIL(%d)" % bad_total)
    logger.info("%s", "=" * 78)
    return 0 if bad_total == 0 else 1


# =============================================================================
# CLI
# =============================================================================


def main():
    ap = argparse.ArgumentParser(
        prog="pangu30b_quant.py",
        description="Pangu 30B 系列 W8A8 量化（单文件工具）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="默认量化范围（五组）: %s\n"
               "  37层/384专家/26block/d=2 -> 40609 个 Linear\n"
               % ", ".join(DEFAULT_GROUPS))
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    p = sub.add_parser("quant", help="从 bf16 全量量化")
    p.add_argument("--src", "--input-bf16-hf-path", dest="src", required=True,
                   help="原始 bf16 权重目录")
    p.add_argument("--dst", "--output-path", dest="dst", required=True,
                   help="量化输出目录")
    p.add_argument("--groups", "--quant-groups", dest="groups",
                   default=",".join(DEFAULT_GROUPS),
                   help="量化哪些组，逗号分隔。可选: " + ", ".join(sorted(GROUPS)))
    p.add_argument("--device", default="cpu", help="cpu 或 npu")
    p.add_argument("--file-count", type=int, default=0, help="只处理前 N 个分片")
    p.add_argument("--quant-prefix", default="quant_model_weight_w8a8_dynamic",
                   help="输出权重文件名前缀")
    p.add_argument("--dry-run", action="store_true", help="只出计划，不写盘")
    p.add_argument("--allow-hf-download", action="store_true",
                   help="允许输出目录缺元数据时从 HF 下载（默认关闭）")
    p.add_argument("--model-name", default=None)
    p.add_argument("--reference-config", default=None,
                   help="已验证产物的 config.json，用于交叉核验 ignore")
    p.set_defaults(func=cmd_quant)

    p = sub.add_parser("more", help="给已有 W8A8 产物补量化")
    p.add_argument("--src", required=True, help="源 W8A8 目录（只读）")
    p.add_argument("--dst", required=True, help="目标目录")
    p.add_argument("--groups", default=None,
                   help="预设组，逗号分隔。可选: " + ", ".join(sorted(GROUPS)))
    p.add_argument("--module-re", action="append", default=None,
                   help="自定义正则（可重复；匹配去掉 .weight 的模块名）")
    p.add_argument("--rebuild-ignore", action="store_true",
                   help="ignore 由产物反推重建（默认只删掉目标模块）")
    p.add_argument("--apply", action="store_true", help="正式执行；默认只出计划")
    p.set_defaults(func=cmd_more)

    p = sub.add_parser("verify", help="核对产物")
    p.add_argument("--dst", required=True, help="待核对的产物目录")
    p.add_argument("--ref", default=None,
                   help="参考目录（bf16 源，或「该组尚未量化」的旧产物）用于精度比对")
    p.add_argument("--groups", default=None, help="只核对指定组")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("survey", help="勘察 ckpt 或产物结构")
    p.add_argument("path", help="待勘察的目录")
    p.set_defaults(func=cmd_survey)

    p = sub.add_parser("scope", help="离线自检量化范围（不需要 torch）")
    p.add_argument("--layers", type=int, default=SCOPE_DEFAULTS["layers"])
    p.add_argument("--experts", type=int, default=SCOPE_DEFAULTS["experts"])
    p.add_argument("--dense", type=int, default=SCOPE_DEFAULTS["dense"],
                   help="first_k_dense_replace")
    p.add_argument("--blocks", type=int, default=SCOPE_DEFAULTS["blocks"],
                   help="vision_config.depth")
    p.set_defaults(func=cmd_scope)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
