# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Helpers shared by both attention stacks."""

from types import SimpleNamespace

import pytest

from omni_npu.v1.layers.utils import get_cla_mapping


@pytest.mark.unit
def test_get_cla_mapping_accepts_a_dict_and_an_absent_field():
    """The pair form and its rejections are covered through the npu_pangu delegate."""
    assert get_cla_mapping(SimpleNamespace(num_hidden_layers=8)) == {}
    config = SimpleNamespace(num_hidden_layers=8, cla_explicit_mapping={5: 1})
    assert get_cla_mapping(config) == {5: 1}
