from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class StandardDataset:
    """
    Thin wrapper around AnnData (or AnnData-like).
    Keeps dataset + obs-key mapping consistent across pipeline stages.
    """
    adata: Any  # anndata.AnnData
    obs_key_map: Dict[str, str]

    def validate(self) -> None:
        if self.adata is None:
            raise ValueError("adata is None")
        if not hasattr(self.adata, "obs") or not hasattr(self.adata, "var"):
            raise ValueError("Expected AnnData-like object with .obs and .var")
        if not hasattr(self.adata, "X"):
            raise ValueError("Expected AnnData-like object with .X")
