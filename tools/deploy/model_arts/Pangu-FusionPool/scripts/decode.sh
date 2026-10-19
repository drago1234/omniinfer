#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
init_runtime decode
export DECODE_SERVER_LIST="$ASCEND_RT_VISIBLE_DEVICES"
export DECODE_DATA_PARALLEL_SIZE="$DECODE_SERVER_LIST,$DECODE_SERVER_LIST"
prepare_pool
run_vllm "$@"
