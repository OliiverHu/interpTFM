from __future__ import annotations

import os
import random
from typing import Dict, Iterator, List, Tuple

import pandas as pd
import torch
from torch.utils.data import IterableDataset

from interp_pipeline.io.activation_store import ActivationStore


def build_id_to_gene(vocab: Dict[str, int]) -> Dict[str, str]:
    """
    Invert a scGPT tokenizer vocab {gene_name: int_id} into
    {str(int_id): gene_name}, matching the string token_ids stored in
    ActivationStore index files.

    Usage:
        id_to_gene = build_id_to_gene(handle.tokenizer.vocab)
    """
    return {str(v): k for k, v in vocab.items()}


class ConceptFilteredDataset(IterableDataset):
    """
    Streams token-level activations from an ActivationStore one shard at a
    time, filtering to only the tokens (genes) that appear in the concept
    matrix.  Never materialises the full dataset in RAM.

    For each retained token the label vector is the corresponding column of
    concept_matrix (a binary [n_concepts] float32 tensor).

    Shard list is shuffled once at construction; within each shard, filtered
    samples are yielded in storage order.  Use DataLoader with shuffle=False
    (the IterableDataset contract).

    Args:
        store:          ActivationStore to read from.
        layer:          Layer name (e.g. "layer_4").
        concept_matrix: DataFrame of shape [n_concepts, n_genes].
                        Columns are gene name strings; rows are concept terms.
                        Produced by get_annotation.empirical_gt.build_binary_empirical_gt().
        id_to_gene:     Mapping from str(token_id) → gene_name.
                        Produced by build_id_to_gene(tokenizer.vocab).
        split:          "train" or "test".
        test_fraction:  Fraction of shards held out for the test split.
        seed:           Random seed for shard shuffling.
    """

    def __init__(
        self,
        store: ActivationStore,
        layer: str,
        concept_matrix: pd.DataFrame,
        id_to_gene: Dict[str, str],
        split: str = "train",
        test_fraction: float = 0.2,
        seed: int = 42,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.n_concepts = concept_matrix.shape[0]

        # Pre-build gene → label tensor for O(1) lookup per token.
        self._known_genes: set = set(concept_matrix.columns)
        self._gene_to_label: Dict[str, torch.Tensor] = {
            gene: torch.tensor(concept_matrix[gene].values, dtype=torch.float32)
            for gene in concept_matrix.columns
        }
        self._id_to_gene = id_to_gene

        # Shard-level split — shuffle once, fix order for reproducibility.
        shards = store.list_shards(layer)
        if not shards:
            raise ValueError(f"No shards found for layer '{layer}' in store at '{store.spec.root}'")

        rng = random.Random(seed)
        shuffled = list(shards)
        rng.shuffle(shuffled)

        n_test = max(1, int(len(shuffled) * test_fraction))
        self._shards: List[str] = shuffled[n_test:] if split == "train" else shuffled[:n_test]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for shard_dir in self._shards:
            acts = torch.load(
                os.path.join(shard_dir, "activations.pt"), map_location="cpu"
            )
            index = torch.load(os.path.join(shard_dir, "index.pt"))
            token_ids: List = index.get("token_ids") or []

            for i, tid in enumerate(token_ids):
                gene = self._id_to_gene.get(str(tid))
                if gene and gene in self._known_genes:
                    yield acts[i], self._gene_to_label[gene]
