#!/bin/bash

# usage
# bash pipeline_diff_cover_v1.sh ut_test_v1 swr.cn-east-4.myhuaweicloud.com/omni/omni_infer_v1_a3:20260211_970 /data/CI/scripts_diff_cov_v1

# 拉取镜像并提取代码
UT_Container_NAME=$1
IMAGE_NAME=$2
PROJECT_CODE=$3

fail_case=0
mkdir -p ${PROJECT_CODE}/ut/

pip install esdk-obs-python
if [ ! -f "${PROJECT_CODE}/known_fails_v1.txt" ]; then
    cat > "${PROJECT_CODE}/known_fails_v1.txt" << 'EOF'
unit/
EOF
    echo "✅ 文件已创建并写入 ${#test_cases[@]} 个测试用例: ${PROJECT_CODE}/known_fails_v1.txt"
else
    echo "文件已存在: ${PROJECT_CODE}/known_fails_v1.txt"
    rm "${PROJECT_CODE}/known_fails_v1.txt"
    cat > "${PROJECT_CODE}/known_fails_v1.txt" << 'EOF'
unit/
EOF
    echo "✅ 文件已创建并写入 ${#test_cases[@]} 个测试用例: ${PROJECT_CODE}/known_fails_v1.txt"
fi
if [ -f "${PROJECT_CODE}/known_fails_v1.txt" ]; then
    echo "✅ 成功创建空文件: ${PROJECT_CODE}/known_fails_v1.txt"
else
    echo "❌ 文件创建失败"
fi
docker pull ${IMAGE_NAME}
# 创建临时容器并从容器内拷贝文件
docker run --rm --name ${UT_Container_NAME} \
        --net=host \
    		--privileged=true \
        --entrypoint=bash \
    		-u root \
    		-w /workspace \
    		--device=/dev/davinci_manager \
    		--device=/dev/hisi_hdc \
    		--device=/dev/devmm_svm \
    		-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    		-v /usr/local/dcmi:/usr/local/dcmi \
    		-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    		-v /etc/ascend_install.info:/etc/ascend_install.info \
    		-v /usr/local/sbin:/usr/local/sbin \
    		-v /etc/hccn.conf:/etc/hccn.conf \
    		-v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    		-v /data/CI:/data/CI \
        -v /data/models:/data/models \
    		-v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime:ro \
    		${IMAGE_NAME} -c\
            "
        sleep 1
        cp -r  /workspace/omniinfer ${PROJECT_CODE}
        "
bash ${PROJECT_CODE}/omniinfer/tests/rm_docker.sh

# 运行测试
TIMEOUT_MIN=20
cd ${PROJECT_CODE}/omniinfer/tests
bash run_docker.sh ${IMAGE_NAME}


# 并行跑测试
timeout "${TIMEOUT_MIN}m" bash concurrent_test_run_multi_docker.sh ${PROJECT_CODE}/omniinfer
ret=$?

if [[ $ret -eq 124 ]]; then
  echo "[ERROR] concurrent_test_run_multi_docker.sh timed out after ${TIMEOUT_MIN} minutes"
  exit 124
fi

# 结果后处理
# 覆盖率检查
python ${PROJECT_CODE}/omniinfer/tests/ut_CI_check/ut_CI_cover_rate_check.py \
    --report ${PROJECT_CODE}/omniinfer/tests/coverage_from_dockers/coverage_report.txt \
    --min 80

python3 ${PROJECT_CODE}/omniinfer/tests/ut_CI_check/ut_CI_diff_cov.py \
  --repo-root ${PROJECT_CODE}/omniinfer \
  --coverage-xml ${PROJECT_CODE}/omniinfer/tests/coverage_from_dockers/coverage.xml \
  --old-prefix /workspace/omniinfer \
  --out-html ${PROJECT_CODE}/omniinfer/tests/coverage_from_dockers/diffcov.html \
  --out-txt  ${PROJECT_CODE}/omniinfer/tests/coverage_from_dockers/diffcov.txt \
  --min 80
# # exit

# # 日志解析合并及新增错误检查
python ${PROJECT_CODE}/omniinfer/tests/ut_CI_check/ut_CI_parse_logs.py \
    --log ${PROJECT_CODE}/omniinfer/tests/install_logs/DT_1.log \
    --log ${PROJECT_CODE}/omniinfer/tests/install_logs/DT_2.log \
    --log ${PROJECT_CODE}/omniinfer/tests/install_logs/DT_3.log \
    --log ${PROJECT_CODE}/omniinfer/tests/install_logs/DT_4.log \
    --merged-log ${PROJECT_CODE}/omniinfer/tests/install_logs/merged_log.log \
    --known ${PROJECT_CODE}/known_fails_v1.txt  2>&1 | tee ${PROJECT_CODE}/ut/run_fail.log

# # cat ${PROJECT_CODE}/omniinfer/tests/install_logs/merged_log.log

# 容器时间均衡性检查
python ${PROJECT_CODE}/omniinfer/tests/ut_CI_check/ut_CI_check_durations_balance.py \
    --dir ${PROJECT_CODE}/omniinfer/tests/test_durations_from_dockers

cp -r ${PROJECT_CODE}/omniinfer/tests/coverage_from_dockers ${PROJECT_CODE}/ut/
cp -r ${PROJECT_CODE}/omniinfer/tests/install_logs ${PROJECT_CODE}/ut/
# 删除容器
bash ${PROJECT_CODE}/omniinfer/tests/rm_docker.sh
LOG_FILE="${PROJECT_CODE}/ut/coverage_from_dockers/diffcov.txt"
LOG_FILE_cover="${PROJECT_CODE}/ut/coverage_from_dockers/coverage_report.txt"
LOG_FILE_fail="${PROJECT_CODE}/ut/run_fail.log"
# 读取覆盖率
# PatchCover=$(grep "^Coverage:" $LOG_FILE | awk '{print $2}')
# echo $PatchCover
# if (( $(echo "$PatchCover >= 0.8" | bc -l) )); then
#     echo "omni: ✅ 增量覆盖率通过 ($PatchCover >= 80%)"
# else
#     echo "omni: ❌ 增量覆盖率未通过 ($PatchCover < 80%)"
#     fail_case=0
# fi
if [ ! -f "$LOG_FILE" ]; then
    echo "omni: ✅ 增量覆盖率通过 (分支为master时验证通过)"
else
    PatchCover=$(grep "^Coverage:" "$LOG_FILE" | awk '{print $2}' | sed 's/%//')
    if [ -z "$PatchCover" ]; then
        echo "omni: ✅ 增量覆盖率通过"
    else
        echo "${PatchCover}%"
        if [ $PatchCover -ge 80 ] 2>/dev/null; then
            echo "omni: ✅ 增量覆盖率通过 (${PatchCover}% >= 80%)"
        else
            echo "omni: ❌ 增量覆盖率未通过 (${PatchCover}% < 80%)"
            fail_case=1
        fi
    fi
fi
Cover=$(grep "^TOTAL" $LOG_FILE_cover | awk '{print $NF}' | sed 's/%//')
echo $Cover
Cover_int=${Cover%.*}  # 取整数部分，84.5 -> 84
if (( Cover_int >= 80 )); then
    echo "omni: ✅ 覆盖率通过 ($Cover >= 80%)"
else
    echo "omni: ❌ 覆盖率未通过 ($Cover < 80%)"
    fail_case=1
fi
NEW_FAILURES=$(grep "^new_failures:" $LOG_FILE_fail | awk '{print $2}')

# 判断是否不为0
if [ "$NEW_FAILURES" != "0" ] && [ -n "$NEW_FAILURES" ]; then
#if [ "${NEW_FAILURES:-0}" -gt 22 ]; then
    echo "错误：发现新的失败测试，数量: $NEW_FAILURES"
    fail_case=1
else
    echo "✓ 没有新的失败测试"
fi
if [ $fail_case -eq 1 ]; then
    echo "❌ 最终结果：失败 (exit 1)"
    exit 1
else
    echo "✅ 最终结果：成功 (exit 0)"
    exit 0
fi
