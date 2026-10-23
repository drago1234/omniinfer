# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
from unittest.mock import MagicMock, patch

import pytest

from omni_npu.lopt import (
    LOPT_AVAILABLE,
    LoptParallelTokenizer,
    maybe_get_lopt_tokenizer,
)


class TestEnsureSpecialTokenBuilder:
    @pytest.mark.parametrize("add_special_tokens", [False, True])
    def test_init_installs_builder_used_by_long_text_merge(self, monkeypatch, add_special_tokens):
        import numpy as np
        from types import SimpleNamespace
        import omni_npu.lopt.lopt_wrapper as lw
        from omni_npu.lopt import lopt_utils

        class OpenPanguV2Tokenizer:
            _add_bos_token = True
            add_eos_token = True
            bos_token_id = 101
            eos_token_id = 102

            def __call__(self, *args, **kwargs):
                raise AssertionError("The long-text merge must not fall back to native tokenization")

        tokenizer = OpenPanguV2Tokenizer()
        assert not hasattr(tokenizer, "build_inputs_with_special_tokens")
        text = "abcd" * 64
        merged_ids = [ord(char) for char in text]
        merge = MagicMock(return_value=np.array(merged_ids, dtype=np.int64))
        monkeypatch.setattr(
            lw, "Cpp_match_merge", SimpleNamespace(match=MagicMock(return_value=(1, 1)), merge=merge),
            raising=False,
        )
        for name in ("chunks", "pairs", "flatten"):
            monkeypatch.setattr(lw, name, getattr(lopt_utils, name), raising=False)

        def tokenize_chunks(fn, text_chunks):
            return [
                {
                    "input_ids": np.array([[ord(char) for char in chunk]], dtype=np.int64),
                    "offset_mapping": np.array([[[i, i + 1] for i in range(len(chunk))]]),
                }
                for chunk in text_chunks
            ]

        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
            "multiprocessing.Pool"
        ) as pool:
            pool.return_value.map.side_effect = tokenize_chunks
            tok = lw.LoptParallelTokenizer(model_path="/fake/openpangu", chunk_size=64)
            try:
                result = tok(text, add_special_tokens=add_special_tokens)
                expected = [101, *merged_ids, 102] if add_special_tokens else merged_ids
                assert result.input_ids == expected
                assert tok.tokenizer.build_inputs_with_special_tokens.__self__ is tokenizer
                pool.return_value.map.assert_called_once()
                merge.assert_called_once()
            finally:
                tok.close()

    def test_prefers_native_builder_for_openpangu(self):
        from omni_npu.lopt.lopt_wrapper import LoptParallelTokenizer

        _ensure_special_token_builder = LoptParallelTokenizer._ensure_special_token_builder

        class OpenPanguV2Tokenizer:
            build_inputs_with_special_tokens = MagicMock(return_value=[9, 1, 2, 8])

        tokenizer = OpenPanguV2Tokenizer()
        original = tokenizer.build_inputs_with_special_tokens
        _ensure_special_token_builder(tokenizer)
        assert tokenizer.build_inputs_with_special_tokens is original
        ids = [1, 2]
        assert tokenizer.build_inputs_with_special_tokens(ids) == [9, 1, 2, 8]
        tokenizer.build_inputs_with_special_tokens.assert_called_once_with(ids)

    @pytest.mark.parametrize(
        "add_bos,add_eos,expected",
        [
            (False, False, [1, 2]),
            (True, False, [101, 1, 2]),
            (False, True, [1, 2, 102]),
            (True, True, [101, 1, 2, 102]),
        ],
    )
    def test_openpangu_backend_preserves_bos_eos(self, add_bos, add_eos, expected):
        from omni_npu.lopt.lopt_wrapper import LoptParallelTokenizer

        _ensure_special_token_builder = LoptParallelTokenizer._ensure_special_token_builder

        class OpenPanguV2Tokenizer:
            add_bos_token = add_bos
            _add_bos_token = add_bos
            add_eos_token = add_eos
            bos_token_id = 101
            eos_token_id = 102

        ids = [1, 2]
        tokenizer = OpenPanguV2Tokenizer()
        assert not hasattr(tokenizer, "build_inputs_with_special_tokens")
        _ensure_special_token_builder(tokenizer)
        bound = tokenizer.build_inputs_with_special_tokens
        assert bound.__self__ is tokenizer
        _ensure_special_token_builder(tokenizer)
        assert tokenizer.build_inputs_with_special_tokens is bound
        result = tokenizer.build_inputs_with_special_tokens(ids)
        assert result == expected
        assert ids == [1, 2]
        assert result is not ids
        assert tokenizer.build_inputs_with_special_tokens(ids, [3]) == (
            ([101] if add_bos else []) + [1, 2, 3] + ([102] if add_eos else [])
        )

    def test_other_tokenizers_do_not_get_openpangu_special_tokens(self):
        from types import SimpleNamespace
        from omni_npu.lopt.lopt_wrapper import LoptParallelTokenizer

        _ensure_special_token_builder = LoptParallelTokenizer._ensure_special_token_builder

        tokenizer = SimpleNamespace(
            add_bos_token=True, add_eos_token=True, bos_token_id=101, eos_token_id=102
        )
        _ensure_special_token_builder(tokenizer)
        assert not hasattr(tokenizer, "build_inputs_with_special_tokens")


class TestMaybeGetLoptTokenizer:
    def test_disabled_returns_none(self):
        result = maybe_get_lopt_tokenizer(
            model_path="/fake/path",
            enable_lopt=False,
        )
        assert result is None

    def test_enabled_but_cpp_not_available(self):
        if LOPT_AVAILABLE:
            pytest.skip("Cpp_match_merge is available in this environment")
        result = maybe_get_lopt_tokenizer(
            model_path="/fake/path",
            enable_lopt=True,
        )
        assert result is None


class TestLoptParallelTokenizerWithMock:
    def test_init_basic(self):
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch(
            "multiprocessing.Pool",
        ) as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            tok = LoptParallelTokenizer(
                model_path="/fake/model",
                pool_size=4,
                chunk_size=2048,
                overlap_ratio=0.125,
            )

            assert tok.model_path == "/fake/model"
            assert tok.pool_size == 4
            assert tok.chunk_size == 2048
            assert tok.overlap == int(2048 * 0.125)
            assert tok.tokenizer is mock_tokenizer
            mock_pool_cls.assert_called_once_with(
                4,
                initializer=LoptParallelTokenizer._init_worker,
                initargs=("/fake/model",),
            )

    def test_tokenize_chunk_static(self):
        import omni_npu.lopt.lopt_wrapper as lw

        mock_tokenizer = MagicMock()
        lw._worker_tokenizer = mock_tokenizer
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=MagicMock(),
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(
                model_path="/fake/model",
                pool_size=2,
                chunk_size=512,
            )

        result = tok._tokenize_chunk("hello world")
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["return_tensors"] == "np"
        assert call_kwargs["return_offsets_mapping"] is True
        assert call_kwargs["add_special_tokens"] is False
        assert result == {"input_ids": [1, 2, 3]}

    def test_call_short_text(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(
                model_path="/fake/model",
                pool_size=2,
                chunk_size=4096,
            )

        tok("short text")
        mock_tokenizer.assert_called_once_with(
            "short text", add_special_tokens=False
        )

    def test_call_short_text_with_special_tokens(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(
                model_path="/fake/model",
                pool_size=2,
                chunk_size=4096,
            )

        tok("short text", add_special_tokens=True)
        mock_tokenizer.assert_called_once_with(
            "short text", add_special_tokens=True
        )

    def test_encode(self):
        mock_tokenizer = MagicMock()
        mock_result = MagicMock()
        mock_result.input_ids = [101, 102, 103]
        mock_tokenizer.return_value = mock_result

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(
                model_path="/fake/model",
                pool_size=2,
                chunk_size=4096,
            )

        result = tok.encode("test", add_special_tokens=False)
        assert result == [101, 102, 103]

    def test_init_with_default_values(self):
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch(
            "multiprocessing.Pool",
        ) as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            tok = LoptParallelTokenizer(model_path="/fake/model")

            assert tok.pool_size == 8
            assert tok.chunk_size == 2048
            assert tok.overlap == int(2048 * 0.125)

    def test_maybe_get_lopt_tokenizer_success_path(self, monkeypatch):
        mock_cpp = MagicMock()
        monkeypatch.setitem(sys.modules, "Cpp_match_merge", mock_cpp)

        import importlib
        import omni_npu.lopt.lopt_wrapper as lw

        importlib.reload(lw)

        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch(
            "multiprocessing.Pool",
        ):
            result = lw.maybe_get_lopt_tokenizer(
                model_path="/fake/model",
                enable_lopt=True,
                lopt_pool_size=8,
                lopt_chunk_size=2048,
            )
            assert result is not None
            assert isinstance(result, lw.LoptParallelTokenizer)
            assert result.model_path == "/fake/model"

    def test_maybe_get_lopt_tokenizer_exception_path(self, monkeypatch):
        mock_cpp = MagicMock()
        monkeypatch.setitem(sys.modules, "Cpp_match_merge", mock_cpp)

        import importlib
        import omni_npu.lopt.lopt_wrapper as lw

        importlib.reload(lw)

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("tokenizer load failed"),
        ), patch("multiprocessing.Pool"):
            result = lw.maybe_get_lopt_tokenizer(
                model_path="/bad/path",
                enable_lopt=True,
            )
            assert result is None

    def test_init_worker(self):
        import omni_npu.lopt.lopt_wrapper as lw

        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            lw.LoptParallelTokenizer._init_worker("/fake/model")
        assert lw._worker_tokenizer is mock_tokenizer

    def test_call_long_text(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(
                model_path="/fake/model",
                pool_size=2,
                chunk_size=10,
            )

        tok._parallel_encode = MagicMock(return_value=MagicMock())
        long_text = "a" * 30
        tok(long_text)
        tok._parallel_encode.assert_called_once()

    def test_init_threshold(self):
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            tok = LoptParallelTokenizer(model_path="/fake/model")
            assert tok.threshold == 2

    def test_init_finalizer(self):
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            tok = LoptParallelTokenizer(model_path="/fake/model")
            assert tok._finalizer is not None

    def test_close(self):
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            tok = LoptParallelTokenizer(model_path="/fake/model")
            tok.close()
            mock_pool.close.assert_called_once()
            mock_pool.join.assert_called_once()
            assert tok.pool is None

    def test_close_when_pool_already_none(self):
        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            tok = LoptParallelTokenizer(model_path="/fake/model")
            tok.pool = None
            tok.close()  # Should not raise

    def test_cpp_match_wrapper_success(self):
        import numpy as np
        import omni_npu.lopt.lopt_wrapper as lw

        mock_cpp = MagicMock()
        mock_cpp.match = MagicMock(return_value=(5, 3))

        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(model_path="/fake/model")

        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([4, 5, 6], dtype=np.int64)
        with patch.object(lw, "Cpp_match_merge", mock_cpp, create=True):
            result = tok._cpp_match_wrapper(a, b, 100)
        assert result == (5, 3)
        mock_cpp.match.assert_called_once()

    def test_parallel_encode_success(self, monkeypatch):
        import importlib
        import numpy as np
        import omni_npu.lopt.lopt_wrapper as lw

        mock_cpp = MagicMock()
        mock_cpp.match = MagicMock(return_value=(3, 3))
        mock_cpp.merge = MagicMock(
            return_value=np.array([1, 2, 3, 7, 8, 9])
        )
        monkeypatch.setitem(sys.modules, "Cpp_match_merge", mock_cpp)
        importlib.reload(lw)

        mock_tokenizer = MagicMock()
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = lw.LoptParallelTokenizer(
                model_path="/fake/model", chunk_size=10
            )

        shard_a = {
            "input_ids": np.array([[1, 2, 3]]),
            "offset_mapping": np.array(
                [[[0, 1], [0, 2], [0, 3]]]
            ),
        }
        shard_b = {
            "input_ids": np.array([[3, 7, 8, 9]]),
            "offset_mapping": np.array(
                [[[0, 1], [0, 2], [0, 3], [0, 4]]]
            ),
        }
        tok.pool.map = MagicMock(return_value=[shard_a, shard_b])

        long_text = "a" * 30
        result = tok._parallel_encode(long_text, add_special_tokens=False)
        assert result is not None
        mock_cpp.match.assert_called()
        mock_cpp.merge.assert_called()

    def test_parallel_encode_with_special_tokens(self, monkeypatch):
        import importlib
        import numpy as np
        import omni_npu.lopt.lopt_wrapper as lw

        mock_cpp = MagicMock()
        mock_cpp.match = MagicMock(return_value=(3, 3))
        mock_cpp.merge = MagicMock(
            return_value=np.array([1, 2, 3, 7, 8, 9])
        )
        monkeypatch.setitem(sys.modules, "Cpp_match_merge", mock_cpp)
        importlib.reload(lw)

        mock_tokenizer = MagicMock()
        mock_tokenizer.build_inputs_with_special_tokens = MagicMock(
            return_value=[0, 1, 2, 3, 7, 8, 9, 10]
        )
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = lw.LoptParallelTokenizer(
                model_path="/fake/model", chunk_size=10
            )

        shard = {
            "input_ids": np.array([[1, 2, 3]]),
            "offset_mapping": np.array(
                [[[0, 1], [0, 2], [0, 3]]]
            ),
        }
        tok.pool.map = MagicMock(return_value=[shard, shard])

        long_text = "a" * 30
        tok._parallel_encode(long_text, add_special_tokens=True)
        mock_tokenizer.build_inputs_with_special_tokens.assert_called_once()

    def test_call_boundary_long_text(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = LoptParallelTokenizer(
                model_path="/fake/model", chunk_size=10
            )

        tok._parallel_encode = MagicMock(return_value=MagicMock())
        boundary_text = "a" * 20  # Exactly chunk_size * 2
        tok(boundary_text)
        tok._parallel_encode.assert_called_once()

    def test_parallel_encode_fallback_returns_1d_ids(self, monkeypatch):
        """Match failure must fall back to HF list ids, not numpy (1, N)."""
        import importlib
        import numpy as np
        import omni_npu.lopt.lopt_wrapper as lw
        from transformers import BatchEncoding

        mock_cpp = MagicMock()
        monkeypatch.setitem(sys.modules, "Cpp_match_merge", mock_cpp)
        importlib.reload(lw)

        fallback = BatchEncoding({"input_ids": [11, 22, 33]})
        mock_tokenizer = MagicMock(return_value=fallback)
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ), patch("multiprocessing.Pool"):
            tok = lw.LoptParallelTokenizer(
                model_path="/fake/model", chunk_size=10
            )

        shard = {
            "input_ids": np.array([[1, 2, 3]]),
            "offset_mapping": np.array([[[0, 1], [0, 2], [0, 3]]]),
        }
        tok.pool.map = MagicMock(return_value=[shard, shard])
        tok._cpp_match_wrapper = MagicMock(
            side_effect=RuntimeError("C++ match returned error")
        )

        result = tok._parallel_encode("a" * 30, add_special_tokens=False)
        mock_tokenizer.assert_called_with("a" * 30, add_special_tokens=False)
        assert "return_tensors" not in mock_tokenizer.call_args.kwargs
        assert result.input_ids == [11, 22, 33]
