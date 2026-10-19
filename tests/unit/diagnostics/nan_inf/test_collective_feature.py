# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for unified NaN/Inf collective controls."""
import pytest
from torch import nn

from omni_npu.diagnostics.nan_inf import collective as _nan_inf_collective  # noqa: F401
from omni_npu.diagnostics.nan_inf.detector import NanInfDetector
from omni_npu.worker.collective_features import (
    collective_feature,
    register_collective_feature,
)

pytestmark = pytest.mark.unit


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.target = nn.Linear(2, 2, bias=False)
        self.last = nn.Linear(2, 2, bias=False)

    def forward(self, value):
        return self.last(self.target(self.first(value)))


class FakeRunner:
    def __init__(self, model):
        self._model = model
        self.nan_inf_detector = NanInfDetector()

    def get_model(self):
        return self._model


class FakeWorker:
    def __init__(self, model):
        self.rank = 7
        self.model_runner = FakeRunner(model)


class TestCollectiveFeature:
    def test_registry_accepts_an_unrelated_feature(self):
        @register_collective_feature("unit_test_feature", "configure")
        def configure(worker, value):
            return {"value": value, "worker": worker.rank}

        response = collective_feature(
            FakeWorker(TinyModel()), "unit_test_feature", value="enabled")
        assert response == {
            "feature": "unit_test_feature",
            "action": "configure",
            "global_rank": 7,
            "result": {"value": "enabled", "worker": 7},
        }

    def test_detector_configure_and_status(self, tmp_path):
        worker = FakeWorker(TinyModel())
        response = collective_feature(
            worker, "nan_inf_detector", enable="true", dump_dir=str(tmp_path))
        assert response["feature"] == "nan_inf_detector"
        assert response["action"] == "configure"
        assert response["global_rank"] == 7
        assert response["result"]["enabled"] is True

        status = collective_feature(worker, "nan_inf_detector", action="status")
        assert status["result"]["enabled"] is True
        disabled = collective_feature(
            worker, "nan_inf_detector", enable="false")
        assert disabled["result"]["enabled"] is False

    @pytest.mark.parametrize("feature,action,error", [
        ("", "configure", "non-empty"),
        ("missing", "configure", "unsupported collective feature"),
        ("nan_inf_detector", "missing", "unsupported action"),
    ])
    def test_rejects_unknown_feature_or_action(self, feature, action, error):
        with pytest.raises(ValueError, match=error):
            collective_feature(FakeWorker(TinyModel()), feature, action=action)

    def test_status_rejects_options(self):
        worker = FakeWorker(TinyModel())
        with pytest.raises(ValueError, match="does not accept"):
            collective_feature(worker, "nan_inf_detector", action="status", x=1)
