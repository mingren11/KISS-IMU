"""Evaluation/inference dataset.

The eval/inference/t-SNE scripts import ``SeqDataset`` and ``collate_fn`` from
here. The windowing contract is identical to training, so this is a thin
re-export of the training dataset (which already accepts an optional
``window_size`` positional argument for API parity).
"""

from data.seq_dataset import SeqDataset
from data.collate import collate_fn

__all__ = ["SeqDataset", "collate_fn"]
