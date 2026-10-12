# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Execute the real init_device prefix, stopping before device initialization."""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as NS
from unittest.mock import Mock,patch

import pytest


class DeviceBoundary(Exception):
    pass


@pytest.mark.parametrize("setting,failed",[
    (None, False), ("", False), ('{"7":"1-2"}', False), ("auto", False),
    ('{"7":"1-2"}', True), ("auto", True),
])
def test_cpu_hook_precedes_npu_context_creation(monkeypatch,setting,failed):
    source=Path(os.environ.get("OMNI_WORKER_INIT_TEST_SOURCE",Path(__file__).parents[3]/"omni/worker/npu_worker.py"))
    tree=ast.parse(source.read_text())
    affinity_import=next(n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='omni_npu.worker.cpu_affinity')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NPUWorker')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_device')
    method.decorator_list=[]
    calls=[]
    def boundary(device):
        calls.append('set_device')
        raise DeviceBoundary
    def bind(rank,config):
        calls.append('bind')
        return None if failed else {'rank':rank}
    binding=Mock(side_effect=bind)
    package=ModuleType('omni_npu');package.__path__=[]
    worker_package=ModuleType('omni_npu.worker');worker_package.__path__=[]
    module=ModuleType('omni_npu.worker.cpu_affinity');module.apply_worker_cpu_affinity=binding
    scope={'os':os,'current_platform':NS(device_type='npu',logical_device_id_to_visible_device_id=lambda rank:rank),
           '_needs_dp_local_rank_offset':lambda config:False,
           'torch':NS(device=lambda name:name,npu=NS(set_device=boundary)),'logger':Mock()}
    with patch.dict(sys.modules,{'omni_npu':package,'omni_npu.worker':worker_package,'omni_npu.worker.cpu_affinity':module}):
        exec(ast.unparse(affinity_import),scope)
    exec('from __future__ import annotations\n'+ast.unparse(method),scope)
    instance=NS(device_config=NS(device=NS(type='npu')),parallel_config=NS(data_parallel_size=1,tensor_parallel_size=16,pipeline_parallel_size=1,nnodes_within_dp=1),local_rank=7,rank=7)
    if setting is not None:
        monkeypatch.setenv('OMNI_WORKER_CPUSET_MAP',setting)
    else:
        monkeypatch.delenv('OMNI_WORKER_CPUSET_MAP',raising=False)
    with pytest.raises(DeviceBoundary):
        scope['init_device'](instance)
    assert calls==(['bind','set_device'] if setting else ['set_device'])
    if setting:
        binding.assert_called_once_with(7,instance.parallel_config)
    else:
        binding.assert_not_called()
    if failed:
        scope['logger'].info.assert_not_called()
        scope['logger'].warning.assert_called_once_with(
            'CPU affinity binding failed for worker rank %s; continuing worker startup',7)
    elif setting:
        scope['logger'].info.assert_called_once_with('Early worker CPU placement: %s',{'rank':7})
        scope['logger'].warning.assert_not_called()
    else:
        scope['logger'].info.assert_not_called()
        scope['logger'].warning.assert_not_called()
