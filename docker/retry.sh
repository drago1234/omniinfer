#!/bin/sh
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.


retry() {
  max="${RETRY_MAX:-5}"
  delay="${RETRY_DELAY:-15}"
  attempt=1

  while [ "$attempt" -le "$max" ]; do
    if "$@"; then
      return 0
    fi

    echo ">>> failed (attempt ${attempt}/${max}): $*"
    if [ "$attempt" -eq "$max" ]; then
      return 1
    fi

    sleep "$delay"
    attempt=$((attempt + 1))
  done
}
