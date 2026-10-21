# SPDX-License-Identifier: MIT
"""Independent CPU window reference for Engram hardware tests."""

import numpy as np
import torch


class PanGuNGramContextProvider:
    """CPU windows from vLLM-style batch boundaries and nearest-first lookbacks."""

    def __init__(self, max_ngram_size: int, pad_token_id: int) -> None:
        self.max_ngram_size = int(max_ngram_size)
        self.pad_token_id = int(pad_token_id)
        self._distances = np.arange(self.max_ngram_size, dtype=np.int64)

    def build_from_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor,
        lookback_dead_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
        )
        if any(tensor.device.type != "cpu" for tensor in tensors):
            raise ValueError("Engram context inputs must stay on CPU")
        ids = input_ids.detach().numpy().reshape(-1).astype(np.int64, copy=False)
        pos = positions.detach().numpy().reshape(-1)
        starts = query_start_loc.detach().numpy()
        dead = dead_mask.detach().numpy()
        lookbacks = lookback_token_ids.detach().numpy()
        lookback_dead = lookback_dead_mask.detach().numpy()
        if pos.shape != ids.shape or dead.shape != ids.shape:
            raise ValueError("Engram positions and dead_mask must match input_ids")
        if starts.ndim != 1 or starts.size == 0 or starts[0] != 0 or starts[-1] != ids.size:
            raise ValueError("Engram query_start_loc must span the current token batch")
        if np.any(starts[1:] < starts[:-1]):
            raise ValueError("Engram query_start_loc must be nondecreasing")
        expected = (starts.size - 1, self.max_ngram_size - 1)
        if lookbacks.shape != expected or lookback_dead.shape != expected:
            raise ValueError(f"Engram lookbacks must have shape {expected}")
        windows = np.full((ids.size, self.max_ngram_size), self.pad_token_id, dtype=np.int64)
        if ids.size == 0:
            return torch.from_numpy(windows)

        token_indices = np.arange(ids.size)
        requests = np.searchsorted(starts[1:], token_indices, side="right")
        previous = token_indices[:, None] - self._distances
        chunk_begin = starts[requests, None]
        in_batch = previous >= chunk_begin
        batch_indices = np.maximum(previous, 0)
        lookback_columns = np.clip(chunk_begin - 1 - previous, 0, self.max_ngram_size - 2)
        source = np.where(in_batch, ids[batch_indices], lookbacks[requests[:, None], lookback_columns])
        source_dead = np.where(in_batch, dead[batch_indices], lookback_dead[requests[:, None], lookback_columns])
        source_dead &= source >= 0
        # A dead token terminates the n-gram prefix, as in the vLLM kernel.
        # Missing history (-1) uses padding; it is distinct from a known dead token.
        blocked = np.logical_or.accumulate(source_dead | (pos[:, None] < self._distances), axis=1)
        np.copyto(windows, source, where=(source >= 0) & ~blocked)
        return torch.from_numpy(windows)


class PanGuNGramHashReference:
    """CPU arithmetic reference; never used by the inference execution path."""

    def __init__(self, layout, layer_idx):
        self.config = layout.config
        self.head_sizes = np.asarray(layout.head_sizes, dtype=np.int64).reshape(
            self.config.num_ngram_orders, self.config.num_heads_per_ngram
        )
        bound = int(np.iinfo(np.int64).max // max(layout.head_sizes))
        rng = np.random.default_rng(self.config.seed + 10007 * (layer_idx + 1))
        self.multipliers = rng.integers(0, max(1, bound // 2), size=self.config.max_ngram_size, dtype=np.int64) * 2 + 1

    def __call__(self, windows):
        weighted = windows.numpy().astype(np.int64, copy=False) * self.multipliers
        # Use explicit prefix reductions rather than the device kernel's loop.
        hashes = [
            np.bitwise_xor.reduce(weighted[:, : order + 2], axis=1)[:, None] % sizes[None, :]
            for order, sizes in enumerate(self.head_sizes)
        ]
        return torch.from_numpy(np.concatenate(hashes, axis=1))
