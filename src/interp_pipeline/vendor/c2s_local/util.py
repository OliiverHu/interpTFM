"""
Utility functions for C2S-Scale local implementation.
Adapted from cell2sentence library.
"""

from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.utils import shuffle
import torch


def generate_vocabulary(adata) -> OrderedDict:
    """
    Create a vocabulary dictionary where each key is a gene name (uppercase)
    and the value is the number of non-zero cells for that gene.

    Arguments:
        adata: AnnData object where obs=cells and vars=genes

    Returns:
        OrderedDict mapping gene names to expression counts
    """
    vocabulary = OrderedDict()

    if sparse.issparse(adata.X):
        gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))
    else:
        gene_sums = np.ravel(np.sum(adata.X > 0, axis=0))

    for i, name in enumerate(adata.var_names):
        vocabulary[name.upper()] = gene_sums[i]

    return vocabulary


def generate_sentences(
    adata,
    vocab: OrderedDict,
    delimiter: str = ' ',
    random_state: int = 42,
) -> List[str]:
    """
    Transform expression matrix to cell sentences. Genes are ordered from
    highest expression to lowest expression.

    Arguments:
        adata: AnnData object where obs=cells and vars=genes
        vocab: OrderedDict with gene names as keys
        delimiter: separator for cell sentence strings (default: ' ')
        random_state: random seed for tie-breaking

    Returns:
        List of cell sentence strings
    """
    np.random.seed(random_state)

    mat = sparse.csr_matrix(adata.X) if not sparse.issparse(adata.X) else adata.X.tocsr()
    enc_map = list(vocab.keys())

    sentences = []
    for i in range(mat.shape[0]):
        cols = mat.indices[mat.indptr[i]: mat.indptr[i + 1]]
        vals = mat.data[mat.indptr[i]: mat.indptr[i + 1]]
        # Shuffle to break ties randomly
        cols, vals = shuffle(cols, vals, random_state=random_state + i)
        # Sort by descending expression
        sorted_indices = np.argsort(-vals, kind="stable")
        sentence = delimiter.join([enc_map[cols[idx]] for idx in sorted_indices])
        sentences.append(sentence)

    return sentences


def to_device(data, device: Union[str, torch.device]):
    """
    Recursively move tensors in nested data structure to device.
    """
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
