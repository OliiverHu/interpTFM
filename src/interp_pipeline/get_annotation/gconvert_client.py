from __future__ import annotations

import hashlib
import json
import os
import requests
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient


def _sanitize_uniprot_id(x: str) -> str:
    x = str(x).strip()
    if x.startswith("UniProtKB:"):
        x = x.split(":", 1)[1]
    # QuickGO rejects dot-suffixed IDs like O00182.208
    if "." in x:
        x = x.split(".", 1)[0]
    return x


def _hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class GConvertSpec:
    organism: str = "hsapiens"
    target: str = "UNIPROTSWISSPROT"  # high precision; switch to "UNIPROT" for more coverage
    filter_na: bool = True


class GConvertClient:
    """
    Thin wrapper around g:Profiler g:Convert to map IDs in batch.
    g:Profiler provides identifier conversion (g:Convert) via API and official clients. :contentReference[oaicite:2]{index=2}
    """

    def __init__(self, gp: GProfilerClient, cache_dir: str):
        self.gp = gp
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def ensg_to_uniprot(self, ensg_ids: Sequence[str], spec: GConvertSpec, force: bool = False) -> Dict[str, List[str]]:
        # cache key
        key = _hash_key(f"{spec.organism}|{spec.target}|{len(ensg_ids)}|" + "|".join(ensg_ids[:20]))
        path = os.path.join(self.cache_dir, f"gconvert_{key}.csv")

        if (not force) and os.path.exists(path):
            df = pd.read_csv(path)
        else:
            # Your GProfilerClient likely already wraps this; if not, we can add it there.
            # Many g:Profiler clients return columns: incoming, converted, name, description
            df = self.gp.convert(list(ensg_ids), organism=spec.organism, target=spec.target)
            if not hasattr(df, "to_csv"):
                df = pd.DataFrame(df)
            df.to_csv(path, index=False)

        cols = set(df.columns)
        if "incoming" not in cols or "converted" not in cols:
            raise ValueError(f"Unexpected g:Convert columns: {list(df.columns)}")

        out: Dict[str, List[str]] = {g: [] for g in ensg_ids}

        for inc, conv in zip(df["incoming"].astype(str), df["converted"].astype(str)):
            conv = conv.strip()
            if conv.lower() in {"nan", "none", ""}:
                continue

            conv = _sanitize_uniprot_id(conv)
            if conv.lower() in {"nan", "none", ""}:
                continue

            out.setdefault(str(inc), []).append(conv)

        # de-dupe per gene
        for k, vs in out.items():
            seen = set()
            out[k] = [v for v in vs if not (v in seen or seen.add(v))]

        if spec.filter_na:
            # keep keys but allow empty lists; caller can decide coverage threshold
            pass

        return out
    
    def convert(
        self,
        query,
        organism: str = "hsapiens",
        target: str = "UNIPROTSWISSPROT",
    ):
        """
        Wrapper for g:Convert API.
        Converts gene identifiers (e.g., ENSG IDs) to another namespace (e.g., UniProt).

        Official g:Profiler API supports ID conversion (g:Convert).
        """

        url = "https://biit.cs.ut.ee/gprofiler/api/convert"

        payload = {
            "organism": organism,
            "query": query,
            "target": target,
        }

        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # Expected structure:
        # {
        #   "result": [
        #       {"incoming": "...", "converted": "...", ...},
        #       ...
        #   ]
        # }
        results = data.get("result", [])
        return pd.DataFrame(results)