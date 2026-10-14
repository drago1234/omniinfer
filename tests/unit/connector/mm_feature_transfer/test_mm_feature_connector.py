# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import pickle
import json
import time
import shutil
import sqlite3
import asyncio
from unittest.mock import MagicMock, patch, call, ANY

import pytest
import torch
from vllm.multimodal.processing import PromptUpdateDetails
from vllm.multimodal.processing.processor import ResolvedPromptUpdate

from omni_npu.connector.mm_feature_transfer.config import DiskConnectorConfig
from omni_npu.connector.mm_feature_transfer.mm_feature_connector.disk_connector import DiskMMFeatureConnector


# -------------------- Fixtures --------------------
@pytest.fixture
def mock_meta():
    """Mock metadata store."""
    meta = MagicMock()
    meta.inc_refcount.return_value = None
    meta.dec_refcount.return_value = None
    meta.add_feature.return_value = None
    meta.get_total_used.return_value = 0
    meta.get_eviction_candidates.return_value = []
    meta.delete_old_features.return_value = []
    # Removal re-checks the refcount before deleting a feature directory; an unset
    # MagicMock would compare as "still referenced" and skip every eviction.
    meta.get_refcount.return_value = 0
    return meta


@pytest.fixture
def mock_mm_item():
    """Mock a MultiModalKwargsItem with tensors."""
    field1 = MagicMock()
    field1.modality = "image"
    field1.data = MagicMock()
    field1.data.numel.return_value = 100
    field1.data.element_size.return_value = 4

    field2 = MagicMock()
    field2.modality = "image"
    field2.data = MagicMock()
    field2.data.numel.return_value = 200
    field2.data.element_size.return_value = 4

    mm_item = MagicMock()
    # keys cantains two fields, one of which is to be excluded
    mm_item.keys.return_value = ["image_tensor", "address"]
    
    # __getitem__ returns the corresponding field
    def _getitem_side_effect(key):
        if key == "image_tensor":
            return field1
        else:
            return field2
    mm_item.__getitem__.side_effect = _getitem_side_effect
    mm_item.__iter__.return_value = iter(["image_tensor", "address"])
    mm_item.values.return_value = [field1, field2]
    mm_item.items.return_value = [("image_tensor", field1), ("address", field2)]
    return mm_item


@pytest.fixture
def mock_prompt_updates():
    """Mock a list of ResolvedPromptUpdate."""
    pu1 = MagicMock()
    pu2 = MagicMock()
    return [pu1, pu2]


def create_resolved_prompt_update(full="dummy_full", modality="image", item_idx=0, mode="replace", target="target"):
    content = PromptUpdateDetails.from_seq(full)  # is_embed is None
    return ResolvedPromptUpdate(
        modality=modality,
        item_idx=item_idx,
        mode=mode,
        target=target,
        content=content
    )


@pytest.fixture
def real_prompt_updates():
    """Return a list of real ResolvedPromptUpdate objects."""
    return [
        create_resolved_prompt_update(full="prompt1", modality="image", item_idx=0, mode="replace", target="tgt1"),
        create_resolved_prompt_update(full="prompt2", modality="video", item_idx=1, mode="append", target="tgt2"),
    ]


@pytest.fixture
def connector(tmp_path, mock_meta):
    """Create a DiskMMFeatureConnector with mocked metadata store and no background threads."""
    config = DiskConnectorConfig(
        # type="DiskMMFeatureConnector",
        storage_path=str(tmp_path / "features"),
        max_size_gb=0.05,
        high_water_ratio=0.85,
        low_water_ratio=0.70,
        ttl_seconds=600,
        cleanup_interval=60,
        exclude_fields=("address", "monotonic_id"),
        is_producer=True,
        metadata_store_type="SQLiteMetadataStore",
    )

    # Patch the metadata store factory to return our mock
    with patch(
        "omni_npu.connector.mm_feature_transfer.mm_feature_connector.disk_connector.create_metadata_store",
        return_value=mock_meta,
    ):
        # Patch _start_background_threads to avoid real thread creation
        with patch.object(DiskMMFeatureConnector, "_start_background_threads") as mock_start:
            conn = DiskMMFeatureConnector(config)
            # Store mock for assertions later
            conn._meta_mock = mock_meta
            conn._start_mock = mock_start
            return conn


# -------------------- Tests for has_item --------------------
class TestHasItem:
    def test_has_item_producer_returns_false_without_incrementing_refcount(self, connector):
        """Producer mode short-circuits before refcount tracking: inc_refcount is not called."""
        mm_hash = "producer_hash"
        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is False
            connector._meta_mock.inc_refcount.assert_not_called()

    def test_has_item_consumer_file_exists_returns_true_and_increments_refcount(self, connector):
        """If metadata file exists in consumer mode, return True and inc refcount."""
        mm_hash = "abc123"
        connector.is_consumer = True
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        metadata_file = os.path.join(feature_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump({"modality": "image", "tensor_keys": []}, f)

        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is True
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)

    def test_has_item_consumer_file_missing_returns_false_and_increments_refcount(self, connector):
        """If metadata file does not exist in consumer mode, return False and inc refcount."""
        mm_hash = "missing_hash"
        connector.is_consumer = True
        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is False
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)

    def test_has_item_consumer_exception_in_inc_refcount_caught_returns_false(self, connector):
        """If inc_refcount raises in consumer mode, catch exception and return False."""
        mm_hash = "error_hash"
        connector.is_consumer = True
        connector._meta_mock.inc_refcount.side_effect = Exception("DB error")
        with patch.object(connector, "is_profile_run", return_value=False):
            result = connector.has_item(mm_hash)
            assert result is False
            connector._meta_mock.inc_refcount.assert_called_once_with(mm_hash)


# -------------------- Tests for save_item_with_updates --------------------
class TestSaveItemWithUpdates:
    def test_save_already_published_skips(self, connector, mock_mm_item, real_prompt_updates):
        """A feature that is already fully published (metadata.json present) is skipped.

        The check is the completeness marker, not the bare directory: a directory left
        behind by a partial write must not block re-saving the feature.
        """
        mm_hash = "existing"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump({"modality": "image", "tensor_keys": []}, f)
        connector.save_item_with_updates(mm_hash, mock_mm_item, real_prompt_updates)
        connector._meta_mock.add_feature.assert_not_called()

    @patch("safetensors.torch.save_file")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("pickle.dump")
    def test_save_successful(self, mock_pickle_dump, mock_open, mock_save_file, 
                             connector, mock_mm_item, real_prompt_updates):
        """Normal save flow: write into a staging dir, register, then publish atomically."""
        mm_hash = "newhash"
        # Call save() directly instead of going through save_item_with_updates(), whose
        # fire-and-forget scheduling would let the mocks expire before the body runs.
        metadata, tensors, updates = connector.prepare(mock_mm_item, real_prompt_updates)
        asyncio.run(connector.save(mm_hash, metadata, tensors, updates))

        feature_dir = connector._get_feature_dir(mm_hash)
        opened = [c.args[0] for c in mock_open.call_args_list if c.args]
        metadata_path = next(p for p in opened if p.endswith("metadata.json"))
        prompt_path = next(p for p in opened if p.endswith("prompt_updates.pkl"))
        tensor_path = mock_save_file.call_args_list[0].args[1]

        # Everything is written into a staging dir next to the final one; the final dir
        # only holds the published result.
        staging_dir = os.path.dirname(metadata_path)
        assert staging_dir.startswith(feature_dir + ".staging-")
        assert os.path.dirname(prompt_path) == staging_dir
        assert os.path.dirname(tensor_path) == staging_dir
        assert os.path.exists(feature_dir)

        # Check tensors saved: only "image_tensor", not "address"
        expected_tensor_calls = [
            call({"image_tensor": ANY}, os.path.join(staging_dir, "image_tensor.safetensors")),
        ]
        mock_save_file.assert_has_calls(expected_tensor_calls, any_order=True)
        assert mock_save_file.call_count == 1

        # Check metadata writing: metadata.json is the completeness marker, written last
        mock_open.assert_any_call(os.path.join(staging_dir, "metadata.json"), "w")
        mock_open.assert_any_call(os.path.join(staging_dir, "prompt_updates.pkl"), "wb")
        assert opened.index(metadata_path) > opened.index(prompt_path)

        # Check pickle dump called
        mock_pickle_dump.assert_called_once()
        args, _ = mock_pickle_dump.call_args
        serialized_list = args[0]  # first argument is the list
        assert isinstance(serialized_list, list)
        # Each item should be result of serialize_prompt_update (we didn't patch that, but we can assume)

        # Check meta.add_feature called with estimated size
        connector._meta_mock.add_feature.assert_called_once_with(mm_hash, ANY)


# -------------------- Tests for load_item_with_updates --------------------
class TestLoadItemWithUpdates:
    def test_load_metadata_missing_returns_none(self, connector):
        """If metadata.json missing, return None."""
        mm_hash = "missing_meta"
        result = connector.load_item_with_updates(mm_hash)
        assert result is None
        # dec_refcount should still be called
        # connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)

    def test_load_tensor_missing_returns_none(self, connector):
        """If a tensor file listed in metadata is missing, return None."""
        mm_hash = "missing_tensor"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        # Create metadata with a tensor key that won't have a file
        metadata = {"modality": "image", "tensor_keys": ["image", "missing"]}
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        # Create only "image.safetensors"
        with open(os.path.join(feature_dir, "image.safetensors"), "w") as f:
            f.write("dummy")  # just to exist

        result = connector.load_item_with_updates(mm_hash)
        assert result is None
        connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)

    def test_load_prompt_file_missing_returns_none(self, connector):
        """If prompt_updates.pkl missing, return None."""
        mm_hash = "missing_prompt"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        metadata = {"modality": "image", "tensor_keys": ["image"]}
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        # Create tensor file
        with open(os.path.join(feature_dir, "image.safetensors"), "w") as f:
            f.write("dummy")
        # No prompt file
        result = connector.load_item_with_updates(mm_hash)
        assert result is None
        connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)
    
    @patch("safetensors.torch.load_file")
    @patch("pickle.load")
    def test_load_successful(
        self, mock_pickle_load, mock_safe_load,
        connector, real_prompt_updates
    ):
        """Full successful load: loads all tensors, reconstructs mm_item and prompt updates."""
        mm_hash = "load_success"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        # Create metadata
        metadata = {"modality": "image", "tensor_keys": ["image"]}
        with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        tensor_path = os.path.join(feature_dir, "image.safetensors")
        with open(tensor_path, "w") as f:
            f.write("dummy")
        prompt_path = os.path.join(feature_dir, "prompt_updates.pkl")
        serialized_data = [{
            "modality": "image", 
            "item_idx": 0, 
            "mode": "replace", 
            "target": "tgt", 
            "content_info": ("full", None, None)
        }]
        with open(prompt_path, "wb") as f:
            pickle.dump(serialized_data, f)

        loaded_tensor = MagicMock()
        mock_safe_load.return_value = {"image": loaded_tensor}
        mock_pickle_load.return_value = serialized_data

        result = connector.load_item_with_updates(mm_hash)

        mock_safe_load.assert_called_once_with(tensor_path)
        mock_pickle_load.assert_called_once()
        assert list(result[0]) == ["image"]
        assert result[0]["image"].data is loaded_tensor
        assert len(result[1]) == 1
        # dec_refcount called
        connector._meta_mock.dec_refcount.assert_called_once_with(mm_hash)


class TestEvictIfNeeded:
    def test_no_eviction_when_usage_below_high_water(self, connector):
        """Used space below high water ratio; no eviction"""
        connector.max_bytes = 100  
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        used = 80  # <= 85
        connector._meta_mock.get_total_used.return_value = used
        connector._evict_if_needed()
        connector._meta_mock.get_eviction_candidates.assert_not_called()
        connector._meta_mock.delete_feature.assert_not_called()

    def test_eviction_when_above_high_water_and_candidates_available(self, connector, tmp_path):
        connector.max_bytes = 100
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        used = 90  # > 85
        connector._meta_mock.get_total_used.return_value = used
        candidates = [("hash1", 15), ("hash2", 10)]
        connector._meta_mock.get_eviction_candidates.return_value = candidates

        for h, _ in candidates:
            dir_path = connector._get_feature_dir(h)
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, "dummy"), "w") as f:
                f.write("test")

        connector._evict_if_needed()

        connector._meta_mock.get_eviction_candidates.assert_called_once_with(20)
        expected_delete_calls = [call("hash1"), call("hash2")]
        connector._meta_mock.delete_feature.assert_has_calls(expected_delete_calls)
        for h, _ in candidates:
            assert not os.path.exists(connector._get_feature_dir(h))

    def test_eviction_stops_when_freed_enough(self, connector, tmp_path):
        connector.max_bytes = 100
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        used = 90
        connector._meta_mock.get_total_used.return_value = used
        candidates = [("hash1", 30), ("hash2", 5)]
        connector._meta_mock.get_eviction_candidates.return_value = candidates

        for h, _ in candidates:
            dir_path = connector._get_feature_dir(h)
            os.makedirs(dir_path, exist_ok=True)

        connector._evict_if_needed()

        connector._meta_mock.delete_feature.assert_called_once_with("hash1")
        assert os.path.exists(connector._get_feature_dir("hash2"))


class TestCleanupByTTL:
    def test_cleanup_old_features(self, connector, tmp_path):
        older_than = int(time.time()) - 100
        hashes = ["expired1", "expired2"]
        connector._meta_mock.delete_old_features.return_value = hashes

        for h in hashes:
            dir_path = connector._get_feature_dir(h)
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, "dummy"), "w") as f:
                f.write("test")

        connector.ttl_seconds = 100
        with patch("time.time", return_value=1000):  # 固定当前时间
            connector._cleanup_by_ttl()

        connector._meta_mock.delete_old_features.assert_called_once_with(900)
        for h in hashes:
            assert not os.path.exists(connector._get_feature_dir(h))

    def test_no_cleanup_when_no_old_features(self, connector):
        connector._meta_mock.delete_old_features.return_value = []
        connector.ttl_seconds = 100
        with patch("time.time", return_value=1000):
            connector._cleanup_by_ttl()
        connector._meta_mock.delete_old_features.assert_called_once_with(900)


# -------------------- Tests for atomic publish / atomic removal --------------------
class TestAtomicPublishAndRemoval:
    """A feature must never be observable in a partial state.

    `has_item()` only checks for metadata.json while `load()` also needs every tensor and
    prompt_updates.pkl, and a removal is not atomic either (rmtree deletes file by file),
    so a reader could otherwise observe a half-written / half-deleted directory.
    """

    @staticmethod
    def _save(connector, mm_hash):
        metadata = {"modality": "image", "tensor_keys": ["image"], "field_types": {}}
        tensors = {"image": torch.zeros(2, 4)}
        asyncio.run(connector.save(mm_hash, metadata, tensors, []))

    def test_saved_feature_is_complete_when_it_becomes_visible(self, connector, monkeypatch):
        """Publishing happens once, with every file already in place."""
        connector.is_consumer = True
        mm_hash = "atomic"
        feature_dir = connector._get_feature_dir(mm_hash)
        published = []

        real_replace = os.replace

        def spy_replace(src, dst):
            published.append((sorted(os.listdir(src)), os.path.exists(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy_replace)
        monkeypatch.setattr(connector, "is_profile_run", lambda: False)

        self._save(connector, mm_hash)

        assert published == [
            (["image.safetensors", "metadata.json", "prompt_updates.pkl"], False)
        ]
        # has_item() -> True implies the feature is loadable.
        assert connector.has_item(mm_hash) is True
        assert connector.load_item_with_updates(mm_hash) is not None

    def test_failed_save_leaves_no_partial_feature(self, connector, monkeypatch):
        """A save that fails halfway must not publish anything."""
        connector.is_consumer = True
        mm_hash = "failed"
        feature_dir = connector._get_feature_dir(mm_hash)

        def boom(*args, **kwargs):
            raise RuntimeError("pickle failed")

        monkeypatch.setattr(pickle, "dump", boom)
        monkeypatch.setattr(connector, "is_profile_run", lambda: False)

        self._save(connector, mm_hash)

        assert not os.path.exists(feature_dir)
        assert sorted(os.listdir(connector.storage_path)) == []

    def test_removal_renames_the_directory_before_deleting_it(self, connector, monkeypatch):
        """The directory is gone from its own name before rmtree starts deleting files."""
        mm_hash = "evict"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        with open(os.path.join(feature_dir, "x.safetensors"), "w") as f:
            f.write("dummy")

        observed = []
        real_rmtree = shutil.rmtree

        def spy_rmtree(path, *args, **kwargs):
            observed.append((os.path.exists(feature_dir), os.path.basename(path)))
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", spy_rmtree)
        connector._remove_feature_dir(mm_hash)

        assert len(observed) == 1
        gone_from_own_name, deleted_name = observed[0]
        assert gone_from_own_name is False
        assert deleted_name.startswith(mm_hash + ".evicting-")
        assert not os.path.exists(feature_dir)

    def test_removal_skips_a_referenced_feature(self, connector):
        """A feature that a reader just took a reference on is not removed."""
        mm_hash = "referenced"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        connector._meta_mock.get_refcount.return_value = 1

        connector._remove_feature_dir(mm_hash)

        assert os.path.exists(feature_dir)

    def test_removal_of_a_missing_feature_is_a_noop(self, connector):
        connector._remove_feature_dir("does-not-exist")  # must not raise


# -------------------- Tests for the error branches of save / removal --------------------
class TestErrorBranches:
    @staticmethod
    def _save(connector, mm_hash):
        metadata = {"modality": "image", "tensor_keys": ["image"], "field_types": {}}
        tensors = {"image": torch.zeros(2, 4)}
        asyncio.run(connector.save(mm_hash, metadata, tensors, []))

    def test_save_ignores_a_row_that_is_already_registered(self, connector, monkeypatch):
        """An IntegrityError from add_feature (row already there) must not fail the save."""
        connector.is_consumer = True
        mm_hash = "already-registered"
        monkeypatch.setattr(connector, "is_profile_run", lambda: False)
        connector._meta_mock.add_feature.side_effect = sqlite3.IntegrityError("duplicate")

        self._save(connector, mm_hash)

        assert connector.has_item(mm_hash) is True
        assert connector.load_item_with_updates(mm_hash) is not None

    def test_save_replaces_a_leftover_directory(self, connector, monkeypatch):
        """A leftover directory (e.g. from a killed save) is replaced by the new feature."""
        connector.is_consumer = True
        mm_hash = "leftover"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        with open(os.path.join(feature_dir, "stale.safetensors"), "w") as f:
            f.write("stale")
        monkeypatch.setattr(connector, "is_profile_run", lambda: False)

        self._save(connector, mm_hash)

        assert sorted(os.listdir(feature_dir)) == sorted(
            ["image.safetensors", "metadata.json", "prompt_updates.pkl"]
        )

    def test_removal_proceeds_when_the_refcount_cannot_be_read(self, connector):
        """A failing refcount lookup only warns; the feature is still removed."""
        mm_hash = "refcount-error"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)
        connector._meta_mock.get_refcount.side_effect = RuntimeError("DB error")

        connector._remove_feature_dir(mm_hash)

        assert not os.path.exists(feature_dir)

    def test_removal_gives_up_when_another_cleanup_thread_renamed_first(self, connector, monkeypatch):
        """Losing the rename race is not an error, and nothing is deleted afterwards."""
        mm_hash = "rename-race"
        feature_dir = connector._get_feature_dir(mm_hash)
        os.makedirs(feature_dir, exist_ok=True)

        def lose_the_race(*args, **kwargs):
            raise FileNotFoundError("already renamed by another thread")

        removed = []
        monkeypatch.setattr(os, "rename", lose_the_race)
        monkeypatch.setattr(shutil, "rmtree", lambda path, *a, **k: removed.append(path))

        connector._remove_feature_dir(mm_hash)

        assert removed == []
        assert os.path.exists(feature_dir)

    def test_eviction_warns_when_nothing_is_evictable(self, connector):
        """Above the high watermark but no candidates: warn and leave the cache alone."""
        connector.max_bytes = 100
        connector.high_water_ratio = 0.85
        connector.low_water_ratio = 0.70
        connector._meta_mock.get_total_used.return_value = 90
        connector._meta_mock.get_eviction_candidates.return_value = []

        connector._evict_if_needed()

        connector._meta_mock.get_eviction_candidates.assert_called_once()
        connector._meta_mock.delete_feature.assert_not_called()
