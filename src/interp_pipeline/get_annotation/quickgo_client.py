from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import requests

QUICKGO_BASE = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"


def _hash_payload(payload: Dict) -> str:
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


@dataclass(frozen=True)
class QuickGOSpec:
    """
    QuickGO annotation query settings.

    Notes:
    - QuickGO annotations are keyed by gene products (usually UniProt accessions).
    - We query UniProt accessions (from Ensembl xrefs).
    """
    taxon_id: str = "9606"                 # human
    aspects: Optional[Sequence[str]] = None  # ["P","F","C"] optional filter
    evidence_codes: Optional[Sequence[str]] = None  # optional server-side evidence filter
    page_size: int = 200
    sleep_s: float = 0.05
    timeout_s: int = 60


class QuickGOClient:
    """
    Fetch GO annotations via QuickGO REST API.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_annotations_for_uniprot(
        self,
        uniprot_accessions: Sequence[str],
        spec: QuickGOSpec,
        force: bool = False,
    ) -> List[Dict]:
        chunk_size = 50
        all_results: List[Dict] = []

        for i in range(0, len(uniprot_accessions), chunk_size):
            chunk = list(uniprot_accessions[i : i + chunk_size])

            payload: Dict[str, object] = {
                "taxonId": spec.taxon_id,
                "geneProductId": ",".join(chunk),
                "limit": spec.page_size,
            }
            if spec.aspects:
                payload["aspect"] = ",".join(spec.aspects)
            if spec.evidence_codes:
                payload["evidenceCode"] = ",".join(spec.evidence_codes)

            key = _hash_payload(payload)
            path = os.path.join(self.cache_dir, f"quickgo_{key}.json")

            if (not force) and os.path.exists(path):
                all_results.extend(json.load(open(path)))
                continue

            page = 1
            chunk_results: List[Dict] = []
            while True:
                params = dict(payload)
                params["page"] = page

                r = requests.get(
                    QUICKGO_BASE,
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=spec.timeout_s,
                )
                r.raise_for_status()
                data = r.json()

                results = data.get("results", [])
                chunk_results.extend(results)

                page_info = data.get("pageInfo", {})
                total_pages = int(page_info.get("totalPages", page))
                if page >= total_pages:
                    break

                page += 1
                time.sleep(spec.sleep_s)

            json.dump(chunk_results, open(path, "w"), indent=2)
            all_results.extend(chunk_results)
            time.sleep(spec.sleep_s)

        return all_results