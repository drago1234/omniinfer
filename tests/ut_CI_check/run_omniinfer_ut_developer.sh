#!/bin/bash
# =============================================================================
#
#
#用法：
# bash run_omniinfer_pipeline_ut.sh <local_omniinfer_path> <image_name> <base_commit> [work_dir]
# local_omniinfer_path  开发者本地 omniinfer 仓路径（被测代码，须是 git 仓）
# image_name            docker 镜像全名（与 run_docker.sh 直接使用的名字一致）
# base_commit           增量覆盖率基准 commit/分支（必填），用 git diff <base_commit> 取改动
# work_dir              可选，报告输出目录，默认 /tmp/omniinfer_ut_selftest/ut
#
# 可调环境变量：
#   TIMEOUT_MIN  并行测试超时（分钟），默认 20
#   MIN_COVER    全量覆盖率阈值(%)，默认 80
#   MIN_DIFF     增量覆盖率阈值(%)，默认 80
#   STAGE_BASE   staging 根目录，必须在容器挂载路径下，默认 /tmp/omniinfer_ut_selftest
#   PIP_INDEX    容器内预装测试依赖用的 pip 源，默认取宿主机 pip 配置，
#                兜底 https://mirrors.tools.huawei.com/pypi/simple
#
# 例：
#   bash /data/zh00942897/code/vllm_0251_ut/from_zz/run_omniinfer_pipeline_ut.sh \
#     /data/zh00942897/code/vllm_0251_ut/local_code_path/omniinfer \
#     registry-cbu.huawei.com/omniai_omniinfer/ai-infra-infer-1.2.1-arm-openeuler-py311-a3-cann9.1.0-pta_v2.9.0-ai-infra-op_1.5.0-omni_master-vllm-202609041600:0.0.1 \
#     HEAD~1 /data/zh00942897/code/vllm_0251_ut/from_zz/result
# =============================================================================

# 注意：不用全局 set -e，以保证「三项检查各自独立、任一失败不影响其它、全部如实打印」。
set -uo pipefail

# ---------- 参数解析与校验 ----------
usage() {
  # 打印文件顶部的注释块（跳过 shebang，遇到第一行非注释即止）
  awk 'NR==1{next} /^[^#]/{exit} {print}' "$0"
  exit 2
}

LOCAL_OMNI="${1:-}"
IMAGE_NAME="${2:-}"
BASE_COMMIT="${3:-}"
WORK_DIR="${4:-/tmp/omniinfer_ut_selftest/ut}"   # 结果输出目录，任意路径
STAGE_BASE="${STAGE_BASE:-/tmp/omniinfer_ut_selftest}"   # staging 根目录，必须在容器挂载路径下

TIMEOUT_MIN="${TIMEOUT_MIN:-20}"
MIN_COVER="${MIN_COVER:-80}"
MIN_DIFF="${MIN_DIFF:-80}"

if [[ -z "${LOCAL_OMNI}" || -z "${IMAGE_NAME}" || -z "${BASE_COMMIT}" ]]; then
  echo "[ERROR] 缺少必填参数：local_omniinfer_path / image_name / base_commit"
  echo
  usage
fi

# 规范化本地代码路径
LOCAL_OMNI="$(cd "${LOCAL_OMNI}" 2>/dev/null && pwd)" || {
  echo "[ERROR] 本地代码路径不存在：${1}"; exit 2;
}

# 必须是 git 仓
if ! git -C "${LOCAL_OMNI}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ERROR] 不是 git 仓库：${LOCAL_OMNI}（增量覆盖率需要 git 历史）"; exit 2
fi

# base_commit 必须可解析
if ! git -C "${LOCAL_OMNI}" rev-parse --verify "${BASE_COMMIT}^{commit}" >/dev/null 2>&1; then
  echo "[ERROR] base_commit 无法解析：${BASE_COMMIT}（在 ${LOCAL_OMNI} 中不存在该 commit/分支）"; exit 2
fi

# work_dir（结果输出）可以是任意路径；但 staging 必须在容器挂载路径下（容器内才拷得到代码）。
case "${STAGE_BASE}" in
  /tmp/*|/tmp|/data/CI/*|/data/CI|/data/models/*|/data/models) : ;;
  *)
    echo "[ERROR] STAGE_BASE 必须在容器挂载路径下（/tmp、/data/CI、/data/models），当前：${STAGE_BASE}"
    echo "        原因：run_docker.sh 只把这些目录 bind-mount 进容器，容器内才能拷到 stage 的代码。"
    exit 2 ;;
esac

STAGE="${STAGE_BASE}/omniinfer"        # staging（挂载可见，容器读它跑测试）
UT_DIR="${WORK_DIR}"                  # 结果输出目录（任意路径），跑完把报告拷到这里
KNOWN_FAILS="${STAGE_BASE}/known_fails_v1.txt"
DIFF_FILE="${STAGE}/local_diff.txt"
COV_DIR="${STAGE}/tests/coverage_from_dockers"
LOG_DIR="${STAGE}/tests/install_logs"
SETUP_LOG="${UT_DIR}/setup.log"       # docker pull/起容器/装依赖/清理 等噪声输出都进这里

echo "=================== omniinfer UT 本地自测 ==================="
echo "[INFO] 本地代码   : ${LOCAL_OMNI}"
echo "[INFO] 镜像       : ${IMAGE_NAME}"
echo "[INFO] base commit: ${BASE_COMMIT}"
echo "[INFO] 结果输出   : ${UT_DIR}（staging 在 ${STAGE_BASE}）"
echo "[INFO] 超时/阈值  : ${TIMEOUT_MIN}min / 全量>=${MIN_COVER}% / 增量>=${MIN_DIFF}%"
echo "=============================================================="
mkdir -p "${UT_DIR}"
: > "${SETUP_LOG}"   # 详细的环境/安装日志（终端不刷屏，需要时看这里）
echo "[INFO] 详细环境日志: ${SETUP_LOG}"

# ---------- 退出时清理容器 ----------
cleanup() {
  if [[ -f "${STAGE}/tests/rm_docker.sh" ]]; then
    echo "[INFO] 清理容器 DT_1..DT_4 ..."
    bash "${STAGE}/tests/rm_docker.sh" >>"${SETUP_LOG}" 2>&1 || true
  fi
}
trap cleanup EXIT

# ---------- 1) stage 本地代码到挂载可见目录 ----------
echo "[INFO] 准备代码 ..."
mkdir -p "${UT_DIR}" "${STAGE_BASE}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"
# 拷内容（含 .git —— 容器内 run_tests.sh 会 git status/branch/log，缺 .git 会因 set -e 失败）
cp -a "${LOCAL_OMNI}/." "${STAGE}/" || { echo "[ERROR] 拷贝本地代码失败"; exit 1; }
# 清掉可能残留的旧测试产物，避免脏数据
rm -rf "${STAGE}/tests/coverage_from_dockers" \
       "${STAGE}/tests/test_durations_from_dockers" \
       "${STAGE}/tests/install_logs" \
       "${STAGE}/tests/htmlcov" \
       "${STAGE}/tests/.coverage" "${STAGE}"/tests/.coverage.* 2>/dev/null || true

# known_fails 文件（默认空；与开发者自测一致，空表示任何失败都算「新增失败」）
touch "${KNOWN_FAILS}"

# ---------- 2) 生成增量 diff（工作区 vs base_commit，含未提交改动） ----------
git -C "${LOCAL_OMNI}" diff "${BASE_COMMIT}" > "${DIFF_FILE}" || {
  echo "[ERROR] git diff 失败"; exit 1;
}
DIFF_BYTES=$(wc -c < "${DIFF_FILE}" | tr -d ' ')
echo "[INFO] 增量 diff（vs ${BASE_COMMIT}）：${DIFF_BYTES} 字节"$([[ "${DIFF_BYTES}" == "0" ]] && echo "（无改动，增量覆盖率按 100% 处理）")

# ---------- 3) 拉取镜像（噪声进 setup 日志） ----------
echo "[INFO] 拉取镜像 ..."
docker pull "${IMAGE_NAME}" >>"${SETUP_LOG}" 2>&1 || { echo "[ERROR] docker pull 失败，详见 ${SETUP_LOG}"; exit 1; }

# ---------- 4) 起容器（噪声进 setup 日志） ----------
echo "[INFO] 启动 4 个测试容器 ..."
# run_docker.sh 会把宿主的 http_proxy/https_proxy 透传进容器（其 -e http_proxy 那几行）。
# 宿主代理 proxyhk.huawei.com 在容器里解析不了，且环境变量优先级高于 /etc/pip.conf，
# 会让容器内所有 pip 走死路。这里清空后再起容器，容器改走内网源（见下面的 hosts + pip.conf）。
( cd "${STAGE}/tests" && http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= bash run_docker.sh "${IMAGE_NAME}" ) >>"${SETUP_LOG}" 2>&1 || { echo "[ERROR] run_docker.sh 失败，详见 ${SETUP_LOG}"; exit 1; }

# ---------- 4.5) 预装测试依赖到各容器（噪声进 setup 日志） ----------
# 镜像可能不自带 pytest-split/pytest-cov，而库脚本里的 pip install 被注释掉了，装不上就 0 用例。
# 这里预装，并绕过两个坑：清掉容器内不可达的宿主机代理；把宿主机的域名解析结果补进容器
# /etc/hosts（容器有独立的 /etc/hosts，宿主机 hosts 表里的内网 pip 源域名不会带进去）。
# 前提：宿主机自己装得上这两个包；宿主机就装不上的话，先排查宿主机的 pip 源。
# 从各容器名取自库配置（source 不算修改库文件）。
source "${STAGE}/tests/ut_config.sh"
PIP_INDEX="${PIP_INDEX:-$(pip3 config get global.index-url 2>/dev/null)}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.tools.huawei.com/pypi/simple}"
PIP_HOST="${PIP_INDEX#*://}"; PIP_HOST="${PIP_HOST%%/*}"
echo "[INFO] 预装测试依赖(pytest-split/pytest-cov) ..."
# 用宿主机的解析结果补容器 hosts：getent 同时覆盖 /etc/hosts 和 DNS 两条路径，故不硬编码 IP。
PIP_HOST_IP="$(getent hosts "${PIP_HOST}" 2>/dev/null | awk 'NR==1{print $1}')"
for c in "${UT_CONTAINER_NAMES[@]}"; do
  [[ -n "${PIP_HOST_IP}" ]] && docker exec "${c}" bash -c \
    "grep -q ' ${PIP_HOST}\$' /etc/hosts || echo '${PIP_HOST_IP} ${PIP_HOST}' >> /etc/hosts" >>"${SETUP_LOG}" 2>&1
  # 容器内写全局 pip 配置：库脚本 concurrent_test_run_multi_docker.sh 里的
  # `pip install -e /workspace/omniinfer` 拿不到本脚本的 -i 参数，会走容器默认代理
  # proxyhk.huawei.com（开发机上解析不了）去校验 torch 的传递依赖 triton 而失败。
  docker exec "${c}" bash -c \
    "printf '[global]\\nindex-url = %s\\ntrusted-host = %s\\nproxy =\\n' \
       '${PIP_INDEX}' '${PIP_HOST}' > /etc/pip.conf" >>"${SETUP_LOG}" 2>&1
  docker exec -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= "${c}" bash -c "
    python3 -c 'import pytest_split, pytest_cov, chardet' 2>/dev/null && exit 0
    pip install -i '${PIP_INDEX}' --trusted-host '${PIP_HOST}' pytest-split pytest-cov chardet
  " >>"${SETUP_LOG}" 2>&1 || { echo "[ERROR] 容器 ${c} 预装测试依赖失败，详见 ${SETUP_LOG}（先在宿主机试 pip install pytest-split pytest-cov，或用 PIP_INDEX 指定内网源）"; exit 1; }
done

# ---------- 5) 并行跑测试（编排噪声进 setup 日志；每个用例明细在各容器 DT_*.log 里） ----------
echo "[INFO] 并行跑测试中（4 容器，约 10 分钟）..."
timeout "${TIMEOUT_MIN}m" bash "${STAGE}/tests/concurrent_test_run_multi_docker.sh" "${STAGE}" >>"${SETUP_LOG}" 2>&1
run_ret=$?
if [[ ${run_ret} -eq 124 ]]; then
  echo "[WARN] 并行测试在 ${TIMEOUT_MIN} 分钟内未跑完（超时），后续仅对已产出的结果做检查。"
elif [[ ${run_ret} -ne 0 ]]; then
  echo "[WARN] concurrent_test_run_multi_docker.sh 返回非 0（${run_ret}），有容器测试失败，继续检查。"
fi

# ---------- 6) 三项检查（各自独立、如实打印） ----------
CHECK_DIR="${STAGE}/tests/ut_CI_check"

# 6.1 全量覆盖率
echo
echo "---------- [检查1] 全量覆盖率 ----------"
cover_ok=1; cover_pct="N/A"
if [[ -f "${COV_DIR}/coverage_report.txt" ]]; then
  python3 "${CHECK_DIR}/ut_CI_cover_rate_check.py" \
      --report "${COV_DIR}/coverage_report.txt" --min "${MIN_COVER}"
  [[ $? -eq 0 ]] && cover_ok=1 || cover_ok=0
  cover_pct="$(grep '^TOTAL' "${COV_DIR}/coverage_report.txt" | awk '{print $NF}' | tail -1)"
  cover_pct="${cover_pct:-N/A}"
else
  echo "[ERROR] 未找到 coverage_report.txt（测试可能未跑完/合并失败）"; cover_ok=0
fi

# 6.2 增量覆盖率
# ut_CI_diff_cov.py 跑在宿主机上；缺 diff-cover 时它会 pip install，默认走 http_proxy 会超时。
# 这里先用内网源、清掉代理装好，避免检查脚本自己 pip 失败把增量覆盖率判成 FAIL。
echo
echo "---------- [检查2] 增量覆盖率 ----------"
diff_ok=1; diff_pct="N/A"
if ! command -v diff-cover >/dev/null 2>&1 && ! python3 -c "import diff_cover" >/dev/null 2>&1; then
  echo "[INFO] 宿主机预装 diff-cover（清代理 + 内网源）..."
  if ! env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      python3 -m pip install -i "${PIP_INDEX}" --trusted-host "${PIP_HOST}" diff-cover \
      >>"${SETUP_LOG}" 2>&1; then
    echo "[ERROR] 宿主机安装 diff-cover 失败，详见 ${SETUP_LOG}"
    echo "        可手动：http_proxy= https_proxy= python3 -m pip install -i ${PIP_INDEX} --trusted-host ${PIP_HOST} diff-cover"
    diff_ok=0
  fi
fi
if [[ ${diff_ok} -eq 1 && -f "${COV_DIR}/coverage.xml" ]]; then
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    python3 "${CHECK_DIR}/ut_CI_diff_cov.py" \
      --repo-root "${STAGE}" \
      --coverage-xml "${COV_DIR}/coverage.xml" \
      --old-prefix /workspace/omniinfer \
      --diff-file "${DIFF_FILE}" \
      --out-html "${COV_DIR}/diffcov.html" \
      --out-txt  "${COV_DIR}/diffcov.txt" \
      --min "${MIN_DIFF}"
  [[ $? -eq 0 ]] && diff_ok=1 || diff_ok=0
  if [[ "${DIFF_BYTES}" == "0" ]]; then
    diff_pct="100% (无改动)"
  elif [[ -f "${COV_DIR}/diffcov.txt" ]]; then
    diff_pct="$(grep -i '^Coverage:' "${COV_DIR}/diffcov.txt" | awk '{print $2}' | tail -1)"
    [[ -z "${diff_pct}" ]] && diff_pct="见 diffcov.txt"
  fi
else
  echo "[ERROR] 未找到 coverage.xml（测试可能未跑完/合并失败）"; diff_ok=0
fi

# 6.3 失败/新增失败用例
echo
echo "---------- [检查3] 失败用例 ----------"
fail_ok=1; new_fail_cnt="N/A"
shopt -s nullglob
dt_logs=( "${LOG_DIR}"/DT_*.log )
shopt -u nullglob
if [[ ${#dt_logs[@]} -gt 0 ]]; then
  log_args=()
  for l in "${dt_logs[@]}"; do log_args+=( --log "${l}" ); done
  python3 "${CHECK_DIR}/ut_CI_parse_logs.py" \
      "${log_args[@]}" \
      --merged-log "${LOG_DIR}/merged_log.log" \
      --known "${KNOWN_FAILS}" 2>&1 | tee "${UT_DIR}/run_fail.log"
  new_fail_cnt="$(grep '^new_failures:' "${UT_DIR}/run_fail.log" | awk '{print $2}' | tail -1)"
  new_fail_cnt="${new_fail_cnt:-0}"
  if [[ "${new_fail_cnt}" == "0" ]]; then fail_ok=1; else fail_ok=0; fi
else
  echo "[ERROR] 未找到任何 DT_*.log（容器可能未正常启动/运行）"; fail_ok=0; new_fail_cnt="N/A"
fi

# ---------- 7) 拷报告到本地 ut 目录 ----------
echo
echo "[INFO] 收集报告到 ${UT_DIR} ..."
rm -rf "${UT_DIR}/coverage_from_dockers" "${UT_DIR}/install_logs"
[[ -d "${COV_DIR}" ]] && cp -r "${COV_DIR}" "${UT_DIR}/" 2>/dev/null || true
[[ -d "${LOG_DIR}" ]] && cp -r "${LOG_DIR}" "${UT_DIR}/" 2>/dev/null || true

# ---------- 8) 汇总 ----------
pf() { [[ "$1" -eq 1 ]] && echo "✅ PASS" || echo "❌ FAIL"; }

echo
echo "================== 本地自测结果汇总 =================="
echo "  [1] 全量覆盖率 : ${cover_pct} (阈值 ${MIN_COVER}%)   $(pf ${cover_ok})"
echo "  [2] 增量覆盖率 : ${diff_pct} (阈值 ${MIN_DIFF}%)     $(pf ${diff_ok})"
echo "  [3] 新增失败   : ${new_fail_cnt} 个 (要求 0)          $(pf ${fail_ok})"
echo "-----------------------------------------------------"
if [[ ${cover_ok} -eq 1 && ${diff_ok} -eq 1 && ${fail_ok} -eq 1 ]]; then
  final=0; echo "  最终结果       : ✅ 通过 (exit 0)"
else
  final=1; echo "  最终结果       : ❌ 未通过 (exit 1)"
fi
echo "-----------------------------------------------------"
echo "  报告位置："
echo "    全量覆盖率 HTML : ${UT_DIR}/coverage_from_dockers/htmlcov/index.html"
echo "    全量覆盖率 报告 : ${UT_DIR}/coverage_from_dockers/coverage_report.txt"
echo "    增量覆盖率 HTML : ${UT_DIR}/coverage_from_dockers/diffcov.html"
echo "    增量覆盖率 文本 : ${UT_DIR}/coverage_from_dockers/diffcov.txt"
echo "    失败用例汇总    : ${UT_DIR}/run_fail.log"
echo "    各容器日志      : ${UT_DIR}/install_logs/DT_*.log"
echo "    环境/安装日志   : ${SETUP_LOG}"
echo "====================================================="

exit ${final}
