#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
init_runtime prefill
export PREFILL_SERVER_LIST="$ASCEND_RT_VISIBLE_DEVICES"
export OMNI_WORKER_CPUSET_MAP="${OMNI_WORKER_CPUSET_MAP:-auto}"
prepare_pool
run_vllm "$@"
