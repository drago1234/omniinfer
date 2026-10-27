# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Graph-static zigzag query layout; weights and caches remain unchanged."""
import torch


class ZigzagQueryLayout:
    """Assign each rank two mirrored query-token fragments.

    Split graph_width into 2 * tp_size equal fragments. Rank r owns
    fragment r and fragment (2 * tp_size - 1 - r), balancing causal work.
    """
    def __init__(self, width, size, rank):
        if width % (2 * size):
            raise ValueError("Query graph width must divide into 2*TP fragments")
        self.width, self.size, self.rank = width, size, rank
        self.fragment = width // (2 * size)
        self.starts = (rank * self.fragment,
                       (2 * size - 1 - rank) * self.fragment)

    def select(self, x):
        return torch.cat([x.narrow(0, start, self.fragment)
                          for start in self.starts], dim=0)

    def metadata(self, cumulative, kv_lengths, block_table):
        # Tensor arithmetic stays inside capture/replay. No .item(), CPU
        # request-boundary snapshot or per-request Python branching.
        ends = cumulative.to(torch.int32)
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        lengths, histories = [], []
        for lo in self.starts:
            left = starts.clamp(min=lo, max=lo + self.fragment)
            right = ends.clamp(min=lo, max=lo + self.fragment)
            lengths.append(right - left)
            prefix = (ends.clamp(max=lo + self.fragment) - starts).clamp(min=0)
            histories.append((kv_lengths.to(torch.int32) - (ends - starts) + prefix).clamp(min=0))
        q_ends = torch.cat(lengths).cumsum(0, dtype=torch.int32)
        return q_ends, torch.cat(histories), torch.cat([block_table, block_table])

    def restore_rank_major(self, x):
        # Source rank r sent [fragment r, fragment 2*TP-1-r].
        shaped = x.reshape(self.size, 2, self.fragment, -1)
        return torch.cat([shaped[:, 0].reshape(-1, x.size(-1)),
                          shaped.flip(0)[:, 1].reshape(-1, x.size(-1))])

    def gather_full_tokens(self, x):
        """Restore natural token rows, retaining replicated o-convolution state."""
        from vllm.distributed import get_tp_group
        return self.restore_rank_major(get_tp_group().all_gather(x, dim=0))
