# -*- coding: utf-8 -*-
"""
服务快照 JSON → ModelArts 创建请求体(规范) 转换脚本。

用法:
    python convert_body.py <源JSON> [输出JSON] [选项...]
    输出默认: 与源文件同目录, 文件名后缀 "-corrected.json"
    也可用环境变量 MODELARTS_BODY_OUTPUT 指定输出路径(优先级低于命令行第2参数)

选项(均可省略):
    --service-name NAME       覆盖服务名(默认取源 name)
    --group-name NAME         覆盖部署组名(默认 deploy-<服务名>)
    --model-source OBS|EFS    强制 model.source
    --model-address URL       指定 OBS 地址(源缺失时使用)
    --model-source-id UUID    指定 EFS 资产 ID(源缺失时使用)
    --model-mount-path PATH   指定挂载路径
    --model-keep-extra        保留 model 中模板字段之外的其他键
    --fix-unit-names          unit 重名时自动顺序改名
    --allow-unknown-model-address   兼容旧参数(现已无路径校验, 忽略)

model 处理规则(核心):
    不做任何路径格式校验/推导, 路径可为任意值。
    source=OBS → 输出 {source, mount_path, efs_sub_path, read_only,
                       host_cache, address, os_warm_up}
    source=EFS → 输出 {source, mount_path, efs_sub_path, read_only,
                       host_cache, source_id, os_warm_up}
    字段取值优先级: 原 JSON 的值 > 命令行/环境变量指定 > 默认值。

环境变量(可选):
    MODELARTS_WORKSPACE_ID / MODELARTS_SECRET_NAME / MODELARTS_SECRET_TYPE /
    MODELARTS_VERSION / MODELARTS_TYPE / MODELARTS_DEPLOY_TYPE /
    MODELARTS_WORKLOAD_TYPE / MODELARTS_SERVICE_NAME / MODELARTS_GROUP_NAME /
    MODELARTS_MODEL_SOURCE / MODELARTS_MODEL_ADDRESS /
    MODELARTS_MODEL_SOURCE_ID / MODELARTS_MODEL_MOUNT_PATH /
    MODELARTS_MODEL_KEEP_EXTRA=1 / MODELARTS_BODY_OUTPUT
"""
import logging
import copy
import json
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 可配置默认值
def _env(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def _env_bool(name, default=False):
    v = os.environ.get(name)
    return default if v in (None, "") else str(v).lower() in ("1", "true", "yes")


WORKSPACE_ID = _env("MODELARTS_WORKSPACE_ID",
                    "1d670a16e0b34acbb27975dfec4d3790")
SECRET_NAME = _env("MODELARTS_SECRET_NAME", "aifm-infra-infer_MA_secret")
SECRET_TYPE = _env("MODELARTS_SECRET_TYPE", "dew")
DEFAULT_TYPE = _env("MODELARTS_TYPE", "REAL_TIME")
DEFAULT_VERSION = _env("MODELARTS_VERSION", "1.0.0")
DEFAULT_DEPLOY_TYPE = _env("MODELARTS_DEPLOY_TYPE", "MULTI")
DEFAULT_WORKLOAD_TYPE = _env("MODELARTS_WORKLOAD_TYPE", "LWS")
DEFAULT_GROUP_COUNT = int(_env("MODELARTS_GROUP_COUNT", "1"))
DEFAULT_GROUP_WEIGHT = int(_env("MODELARTS_GROUP_WEIGHT", "100"))
DEFAULT_GROUP_PRIORITY = int(_env("MODELARTS_GROUP_PRIORITY", "3"))

# model 各 source 的字段模板与默认值(路径不做任何校验)
MODEL_TEMPLATE = {
    "OBS": ["source", "mount_path", "efs_sub_path", "read_only",
            "host_cache", "address", "os_warm_up"],
    "EFS": ["source", "mount_path", "efs_sub_path", "read_only",
            "host_cache", "source_id", "os_warm_up"],
}
MODEL_DEFAULT = {
    "OBS": {
        "source": "OBS", "mount_path": "/mnt", "efs_sub_path": "/",
        "read_only": False, "host_cache": False, "os_warm_up": False,
        "address": _env("MODELARTS_MODEL_ADDRESS", "obs://bucket-910c-6055"),
    },
    "EFS": {
        "source": "EFS", "mount_path": "/mnt/sfs_turbo/", "efs_sub_path": "/",
        "read_only": False, "host_cache": False, "os_warm_up": False,
        "source_id": _env("MODELARTS_MODEL_SOURCE_ID",
                          "d81f649d-3f0d-42f5-94e7-01c2c7ff59ce"),
    },
}
# 覆盖项(命令行/环境变量优先于默认值, 但低于原 JSON 中已存在的值)
MODEL_OVERRIDE_KEYS = ("address", "source_id", "mount_path")

# runtime_config 默认值(原 JSON 中若已有同名键, 以原 JSON 为准)
RUNTIME_DEFAULT = {
    "service_invoke": {
        "port": 7000, "protocol": "HTTPS", "auth_type": "API_KEY",
        "dynamic_routing_enable": False, "ems_enable": False,
        "internet_access_enable": True, "intranet_approval_enable": True,
        "request_retry_enable": False,
    },
    "service_limit": {
        "rate_limit": {"num": 200, "unit": "SECONDS"},
        "request_size_limit": 20, "request_timeout": 30,
        "ip_white_list": [], "ip_black_list": [],
    },
    "service_secret": {
        "group_enable": False, "secret_enable": False, "secret_volumes": [],
    },
}

# ---------------- 字段白名单(schema 允许的字段) ----------------
ROOT_ALLOWED = {
    "name", "version", "description", "type", "deploy_type", "group_configs",
    "runtime_config", "upgrade_config", "lts_strategy", "log_configs", "tags",
    "workspace_id", "schedule", "custom_metrics_path",
    "deploy_timeout_minutes", "task_type", "workload_type",
}
GROUP_ALLOWED = {
    "id", "name", "pool_id", "count", "system_log_dump_enable",
    "unit_configs", "weight", "secret_type", "secret_name", "priority",
    "high_avail_switch", "schedule_strategy", "version", "version_id",
    "description", "framework", "running_count", "deploy_type",
    "mirror_traffic_enable", "mirror_traffic_weight", "version_count",
    "workload_type", "update_at", "model", "advanced_config",
}
UNIT_ALLOWED = {
    "id", "name", "role", "custom_spec", "flavor", "flavor_display_name",
    "image", "models", "codes", "files", "dumps", "count", "cmd",
    "termination_grace", "envs", "readiness_health", "startup_health",
    "liveness_health", "port", "recovery", "npu_reset_enable", "group_count",
    "affinity", "security_config", "pool_resource_flavor",
}
# 快照里出现在顶层、但属于"组"的字段(下沉到 group_configs[0])
GROUP_FROM_ROOT = {
    "pool_id", "count", "system_log_dump_enable", "weight", "secret_type",
    "secret_name", "priority", "high_avail_switch", "schedule_strategy",
    "framework", "mirror_traffic_enable", "mirror_traffic_weight",
    "deploy_type", "description", "workload_type", "model", "advanced_config",
    "unit_configs",
}
# 响应侧字段(绝不透传)
RESPONSE_ONLY = {
    "id", "infer_name", "status", "create_at", "update_at", "user_name",
    "version_id", "version_count", "running_count", "dispatcher_group_id",
    "failure_reason", "predict_url", "lts_state", "lts_event_state",
    "lts_status", "lts_event_status", "lts_file_status",
}


def _as_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _envs_to_str(envs):
    """envs 必须为 string->string; 非字符串值统一转字符串。"""
    out = {}
    for k, v in (envs or {}).items():
        if v is None:
            continue
        out[str(k)] = v if isinstance(v, str) else json.dumps(
            v, ensure_ascii=False)
    return out


def _deep_merge(base, override):
    """override 覆盖 base(递归合并 dict, 其余直接替换)。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------- model 归一化
def normalize_model(model, report, opts):
    """按 source 输出固定字段集; 不做路径校验/推导。

    取值优先级: 原 JSON 值 > 命令行/环境变量 > 默认值。
    """
    src_raw = str((model or {}).get("source") or "").strip()
    src = (opts.get("model_source") or src_raw or "OBS").upper()
    if src not in MODEL_TEMPLATE:
        report.append("警告: model.source=%r 非 OBS/EFS, 按通用默认值处理" % src)
        src = "OBS"

    template = MODEL_TEMPLATE[src]
    defaults = copy.deepcopy(MODEL_DEFAULT[src])
    overrides = {}
    if opts.get("model_address"):
        overrides["address"] = opts["model_address"]
    if opts.get("model_source_id"):
        overrides["source_id"] = opts["model_source_id"]
    if opts.get("model_mount_path"):
        overrides["mount_path"] = opts["model_mount_path"]

    out = {}
    for k in template:
        if k == "source":
            out[k] = src
            continue
        if k in overrides:
            out[k] = copy.deepcopy(overrides[k])          # 显式覆盖最高优先
            report.append("model.%s 由命令行/环境变量覆盖: %s" % (k, overrides[k]))
        elif model and k in model and model[k] is not None:
            out[k] = copy.deepcopy(model[k])              # 其次用原 JSON 的值
        else:
            out[k] = copy.deepcopy(defaults[k])           # 最后用默认值

    # 布尔字段统一类型
    for k in ("read_only", "host_cache", "os_warm_up"):
        if k in out:
            out[k] = bool(out[k])

    if opts.get("model_keep_extra") and model:
        for k, v in model.items():
            if k not in out and k not in ("type", "readOnly"):
                out[k] = copy.deepcopy(v)
    else:
        skipped = [k for k in (model or {})
                   if k not in out and k not in ("type", "readOnly")]
        if skipped:
            report.append("提示: model 中 %s 不在 %s 模板内, 已省略"
                          "(如需保留: --model-keep-extra)" % (skipped, src))

    report.append("model: source=%s → %s" % (src, json.dumps(out, ensure_ascii=False)))
    if src == "OBS" and not out.get("address"):
        report.append("警告: OBS 挂载缺少 address, 可用 --model-address / "
                      "MODELARTS_MODEL_ADDRESS 指定")
    if src == "EFS" and not out.get("source_id"):
        report.append("警告: EFS 挂载缺少 source_id, 可用 --model-source-id / "
                      "MODELARTS_MODEL_SOURCE_ID 指定")
    return out


# ---------------------------------------------------------------- 主转换
def normalize_create_body(raw, report=None, opts=None, **kw):
    """快照 dict → 规范创建体 dict。"""
    report = report if report is not None else []
    opts = dict(opts or {})
    # 兼容旧调用方式
    if kw.get("fix_unit_names"):
        opts["fix_unit_names"] = True
    if kw.get("allow_unknown_model_address"):
        report.append("提示: --allow-unknown-model-address 已无需使用"
                      "(脚本不再做路径校验)")
    opts.setdefault("fix_unit_names", False)

    # ---- 幂等分支: 已是规范体 ----
    if "group_configs" in raw:
        body = copy.deepcopy(raw)
        body["workspace_id"] = body.get("workspace_id") or WORKSPACE_ID
        body.setdefault("type", DEFAULT_TYPE)
        body.setdefault("version", DEFAULT_VERSION)
        body.setdefault("deploy_type", DEFAULT_DEPLOY_TYPE)
        body.setdefault("workload_type", DEFAULT_WORKLOAD_TYPE)
        body["runtime_config"] = _deep_merge(RUNTIME_DEFAULT,
                                             body.get("runtime_config") or {})
        for i, g in enumerate(body.get("group_configs") or []):
            if g.get("model"):
                g["model"] = normalize_model(g["model"], report, opts)
            if g.get("model", {}).get("source") == "OBS" and not g.get("secret_name"):
                g["secret_name"] = SECRET_NAME
                g.setdefault("secret_type", SECRET_TYPE)
                report.append("group_configs[%d]: OBS 挂载补 secret_name=%s"
                              % (i, SECRET_NAME))
        report.append("输入已是规范体(group_configs): 仅补齐缺失的必填项/归一 model")
        return body

    dropped = set()

    # ---- units ----
    units = []
    seen = {}
    for u in raw.get("unit_configs", []) or []:
        nu = {}
        for k, v in u.items():
            if k in RESPONSE_ONLY or k not in UNIT_ALLOWED or k == "id":
                dropped.add("unit.%s" % k)
                continue
            nu[k] = copy.deepcopy(v)
        if "envs" in nu:
            nu["envs"] = _envs_to_str(nu["envs"])
        if "count" in nu:
            nu["count"] = _as_int(nu["count"], 1)
        if "group_count" in nu:
            nu["group_count"] = _as_int(nu["group_count"], 1)
        if "npu_reset_enable" in nu:
            nu["npu_reset_enable"] = bool(nu["npu_reset_enable"])
        name = nu.get("name", "")
        if name in seen:
            if opts["fix_unit_names"]:
                base = name.rsplit("-", 1)[0] if "-" in name else name
                idx = 1
                while ("%s-%d" % (base, idx)) in seen:
                    idx += 1
                nu["name"] = "%s-%d" % (base, idx)
                report.append("unit 重名自动修正: %s → %s" % (name, nu["name"]))
            else:
                report.append("警告: unit 名称重复 %r(可用 --fix-unit-names 自动改名)"
                              % name)
        seen[nu.get("name", "")] = True
        units.append(nu)

    # ---- group ----
    group = {}
    for k, v in raw.items():
        if k in GROUP_FROM_ROOT:
            group[k] = copy.deepcopy(v)
        elif k in RESPONSE_ONLY or k not in ROOT_ALLOWED:
            dropped.add(k)
    service_name = opts.get("service_name") or raw.get("name", "")
    group["name"] = (opts.get("group_name")
                     or raw.get("group_name")
                     or "deploy-%s" % (service_name or "service"))
    group["unit_configs"] = units
    if "model" in group:
        group["model"] = normalize_model(group["model"], report, opts)
    if "count" in group:
        group["count"] = _as_int(group["count"], DEFAULT_GROUP_COUNT)
    else:
        group["count"] = DEFAULT_GROUP_COUNT
    if "weight" in group:
        group["weight"] = _as_int(group["weight"], DEFAULT_GROUP_WEIGHT)
    if "priority" in group:
        group["priority"] = _as_int(group["priority"], DEFAULT_GROUP_PRIORITY)
    if "system_log_dump_enable" in group:
        group["system_log_dump_enable"] = bool(group["system_log_dump_enable"])
    if group.get("model", {}).get("source") == "OBS":
        if not group.get("secret_name"):
            group["secret_name"] = SECRET_NAME
            report.append("OBS 挂载: 源缺少 secret_name, 已补默认值 %s" % SECRET_NAME)
        group.setdefault("secret_type", SECRET_TYPE)

    # ---- root ----
    body = {
        "name": service_name,
        "version": DEFAULT_VERSION,
        "description": raw.get("description", ""),
        "type": raw.get("type") or DEFAULT_TYPE,
        "deploy_type": raw.get("deploy_type", DEFAULT_DEPLOY_TYPE),
        "group_configs": [group],
        "workspace_id": raw.get("workspace_id") or WORKSPACE_ID,
        "workload_type": raw.get("workload_type", DEFAULT_WORKLOAD_TYPE),
        "runtime_config": _deep_merge(RUNTIME_DEFAULT,
                                      raw.get("runtime_config") or {}),
    }
    for k in ("task_type", "deploy_timeout_minutes", "tags", "schedule",
              "custom_metrics_path", "lts_strategy", "log_configs",
              "upgrade_config"):
        if raw.get(k) not in (None, {}, []):
            body[k] = copy.deepcopy(raw[k])

    if dropped:
        report.append("剔除字段(响应侧/非 schema): %s" % ", ".join(sorted(dropped)))
    if "high_avail_switch" not in group:
        report.append("提示: 组内无 high_avail_switch, 按平台默认提交")
    return body


def check_body(body):
    """必填项自检; 返回 (ok, 问题列表)。"""
    problems = []
    for k in ("name", "type", "deploy_type", "group_configs", "workspace_id",
              "runtime_config"):
        if not body.get(k):
            problems.append("顶层缺少必填字段 %s" % k)
    for i, g in enumerate(body.get("group_configs", [])):
        for k in ("name", "pool_id", "unit_configs"):
            if not g.get(k):
                problems.append("group_configs[%d] 缺少必填字段 %s" % (i, k))
        model = g.get("model") or {}
        if model.get("source") == "OBS" and not model.get("address"):
            problems.append("group_configs[%d].model 为 OBS 但缺少 address" % i)
        if model.get("source") == "EFS" and not model.get("source_id"):
            problems.append("group_configs[%d].model 为 EFS 但缺少 source_id" % i)
        for j, u in enumerate(g.get("unit_configs", [])):
            for k in ("name", "count", "image", "cmd"):
                if not u.get(k):
                    problems.append("group_configs[%d].unit_configs[%d] 缺少 %s"
                                    % (i, j, k))
    return (not problems), problems


# ---------------------------------------------------------------- 入口
VALUED_FLAGS = {
    "--service-name": "service_name",
    "--group-name": "group_name",
    "--model-source": "model_source",
    "--model-address": "model_address",
    "--model-source-id": "model_source_id",
    "--model-mount-path": "model_mount_path",
}
BOOL_FLAGS = {
    "--fix-unit-names": "fix_unit_names",
    "--model-keep-extra": "model_keep_extra",
    "--allow-unknown-model-address": "allow_unknown_model_address",
}


def parse_args(argv):
    opts, positional = {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in VALUED_FLAGS:
            opts[VALUED_FLAGS[a]] = argv[i + 1] if i + 1 < len(argv) else ""
            i += 2
            continue
        if "=" in a and a.split("=", 1)[0] in VALUED_FLAGS:
            k, v = a.split("=", 1)
            opts[VALUED_FLAGS[k]] = v
            i += 1
            continue
        if a in BOOL_FLAGS:
            opts[BOOL_FLAGS[a]] = True
            i += 1
            continue
        if a.startswith("--"):
            logger.warning("警告: 未知参数 %s(已忽略)", a)
            i += 1
            continue
        positional.append(a)
        i += 1
    return opts, positional


def main(argv=None):
    opts, positional = parse_args(list(sys.argv[1:] if argv is None else argv))
    if not positional:
        logger.info("%s", __doc__)
        return 2
    src = positional[0]
    dst = (positional[1] if len(positional) > 1
           else _env("MODELARTS_BODY_OUTPUT", "")
           or os.path.splitext(src)[0] + "-corrected.json")

    # 环境变量作为默认覆盖项(命令行优先)
    for cli_key, env_key in (("service_name", "MODELARTS_SERVICE_NAME"),
                             ("group_name", "MODELARTS_GROUP_NAME"),
                             ("model_source", "MODELARTS_MODEL_SOURCE"),
                             ("model_address", "MODELARTS_MODEL_ADDRESS"),
                             ("model_source_id", "MODELARTS_MODEL_SOURCE_ID"),
                             ("model_mount_path", "MODELARTS_MODEL_MOUNT_PATH")):
        opts.setdefault(cli_key, _env(env_key, "") or None)
        if opts.get(cli_key) in ("", None):
            opts.pop(cli_key, None)
    opts.setdefault("model_keep_extra", _env_bool("MODELARTS_MODEL_KEEP_EXTRA"))

    if not os.path.isfile(src):
        raise FileNotFoundError("源JSON不存在: %s" % src)
    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)

    report = []
    body = normalize_create_body(raw, report, opts)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2, ensure_ascii=False)

    g = body["group_configs"][0]
    ok, problems = check_body(body)
    logger.info('%s %s', "源文件  :", src)
    logger.info('%s %s', "输出文件:", dst)
    logger.info('%s %s', "服务名  :", body["name"])
    logger.info('%s %s %s %s', "组      :", g.get("name"), "| pool:", g.get("pool_id"))
    logger.info('%s %s', "model   :", json.dumps(g.get("model"), ensure_ascii=False))
    logger.info('%s %s', "units   :", [(u.get("name"), u.get("count")) for u in g["unit_configs"]])
    for r in report:
        logger.info('%s %s', "  -", r)
    logger.info('%s %s', "自检    :", "PASS" if ok else "FAIL")
    for p in problems:
        logger.info('%s %s', "    !", p)
    return 0 if ok else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
