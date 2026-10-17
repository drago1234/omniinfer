# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""OMNI-NAN-INF: runtime NaN/Inf detection toggled via ``/collective_rpc``.

* ``detector``: worker-side forward hooks, dump, cross-rank verdict.
* ``collective``: NaN/Inf actions registered with the worker RPC dispatcher.
* ``degradation``: scheduler/API plumbing that fails the tripped batch and
  surfaces it upstream as HTTP 500 ``ModelDegradation``.
"""
