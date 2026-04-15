from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

import requests

# Ensembl REST API base
ENSEMBL_REST = "https://rest.ensembl.org"


def _hash_key(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EnsemblXrefSpec:
    """
    Ensembl xref query settings.

    external_db examples:
      - "UniProtKB/Swiss-Prot" (reviewed; lower coverage)
      - "UniProtKB"           (includes TrEMBL; higher coverage)
    """
    external_db: str = "UniProtKB/Swiss-Prot"
    timeout_s: int = 60


class EnsemblXrefClient:
    """
    Maps Ensembl stable IDs (ENSG...) to external references like UniProt accessions
    using Ensembl REST /xrefs/id endpoint.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def ensg_to_uniprot(self, ensg_id: str, spec: EnsemblXrefSpec, force: bool = False) -> List[str]:
        key = _hash_key(f"{ensg_id}|{spec.external_db}")
        path = os.path.join(self.cache_dir, f"ensxref_{key}.json")

        if (not force) and os.path.exists(path):
            return json.load(open(path))

        url = f"{ENSEMBL_REST}/xrefs/id/{ensg_id}"
        params = {"external_db": spec.external_db}

        r = requests.get(
            url,
            params=params,
            headers={"Content-Type": "application/json"},
            timeout=spec.timeout_s,
        )
        r.raise_for_status()
        data = r.json()

        accs: List[str] = []
        for x in data:
            pid = x.get("primary_id")
            if pid:
                accs.append(str(pid))

        # De-dupe, preserve order
        seen: Set[str] = set()
        out: List[str] = []
        for a in accs:
            if a not in seen:
                seen.add(a)
                out.append(a)

        json.dump(out, open(path, "w"), indent=2)
        return out

    def batch_ensg_to_uniprot(
        self,
        ensg_ids: Sequence[str],
        spec: EnsemblXrefSpec,
        force: bool = False,
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for g in ensg_ids:
            out[g] = self.ensg_to_uniprot(g, spec=spec, force=force)
        return out