from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

import pandas as pd

from gprofiler import GProfiler  # gprofiler-official


def _stable_hash(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


@dataclass(frozen=True)
class GProfilerSpec:
    organism: str = "hsapiens"
    user_threshold: float = 0.05
    significance_threshold_method: str = "fdr"  # fdr, bonferroni, gSCS
    sources: Optional[Sequence[str]] = None
    background: Optional[Sequence[str]] = None
    no_evidences: bool = False
    ordered: bool = False
    return_dataframe: bool = True
    user_agent: str = "interp_pipeline"


class GProfilerClient:
    """
    g:Profiler helper with disk caching for:
      - g:GOSt enrichment (profile)
      - g:Convert identifier mapping (convert)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_paths(self, key: str) -> Dict[str, str]:
        return {
            "json": os.path.join(self.cache_dir, f"{key}.json"),
            "csv": os.path.join(self.cache_dir, f"{key}.csv"),
        }

    # ------------------------
    # g:GOSt enrichment
    # ------------------------
    def profile(
        self,
        gene_list: Sequence[str],
        spec: GProfilerSpec,
        query_name: Optional[str] = None,
        force: bool = False,
    ) -> Union["pd.DataFrame", List[Dict[str, Any]]]:
        payload: Dict[str, Any] = {
            "kind": "gost",
            "organism": spec.organism,
            "query": list(gene_list),
            "query_name": query_name,
            "user_threshold": spec.user_threshold,
            "significance_threshold_method": spec.significance_threshold_method,
            "sources": list(spec.sources) if spec.sources else None,
            "background": list(spec.background) if spec.background else None,
            "no_evidences": spec.no_evidences,
            "ordered": spec.ordered,
        }
        key = _stable_hash(payload)
        paths = self._cache_paths(key)

        if (not force) and os.path.exists(paths["json"]):
            with open(paths["json"], "r") as f:
                raw = json.load(f)
            if spec.return_dataframe and pd is not None and isinstance(raw, list):
                return pd.DataFrame(raw)
            return raw

        gp = GProfiler(
            user_agent=spec.user_agent,
            return_dataframe=bool(spec.return_dataframe and pd is not None),
        )

        res = gp.profile(
            organism=spec.organism,
            query=list(gene_list),
            user_threshold=spec.user_threshold,
            significance_threshold_method=spec.significance_threshold_method,
            sources=list(spec.sources) if spec.sources else None,
            background=list(spec.background) if spec.background else None,
            no_evidences=spec.no_evidences,
            ordered=spec.ordered,
        )

        # Normalize for caching
        if pd is not None and spec.return_dataframe and hasattr(res, "to_dict"):
            raw = res.to_dict(orient="records")
        else:
            raw = res

        with open(paths["json"], "w") as f:
            json.dump(raw, f, indent=2)

        if pd is not None and spec.return_dataframe and hasattr(res, "to_csv"):
            res.to_csv(paths["csv"], index=False)

        return res

    # ------------------------
    # g:Convert ID mapping
    # ------------------------
    def convert(
        self,
        query: Sequence[str],
        organism: str = "hsapiens",
        target: str = "UNIPROTSWISSPROT",
        force: bool = False,
    ) -> "pd.DataFrame":
        """
        g:Convert API wrapper (batched ID conversion).

        Correct endpoint (per g:Profiler API docs):
        https://biit.cs.ut.ee/gprofiler/api/convert/convert/
        """

        if pd is None:
            raise ImportError("pandas is required for GProfilerClient.convert(). Please `pip install pandas`.")

        payload: Dict[str, Any] = {
            "kind": "convert",
            "organism": organism,
            "target": target,
            "query": list(query),
        }
        key = _stable_hash(payload)
        paths = self._cache_paths(key)

        if (not force) and os.path.exists(paths["json"]):
            raw = json.load(open(paths["json"]))
            return pd.DataFrame(raw)

        # Try the documented endpoint first; fall back to a couple variants.
        endpoints = [
            "https://biit.cs.ut.ee/gprofiler/api/convert/convert/",
            "https://biit.cs.ut.ee/gprofiler/api/convert/convert",   # sometimes works without trailing slash
            "https://biit.cs.ut.ee/gprofiler/api/convert",           # legacy/redirects (may 404)
        ]

        last_err = None
        for url in endpoints:
            try:
                r = requests.post(
                    url,
                    json={"organism": organism, "target": target, "query": list(query)},
                    timeout=120,
                )
                if r.status_code == 404:
                    last_err = RuntimeError(f"404 from {url}")
                    continue
                r.raise_for_status()
                data = r.json()
                raw = data.get("result", data if isinstance(data, list) else [])
                json.dump(raw, open(paths["json"], "w"), indent=2)
                return pd.DataFrame(raw)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"g:Convert failed for all endpoints. Last error: {last_err}")