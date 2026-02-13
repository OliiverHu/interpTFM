from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

TokenUnit = Literal["cell", "gene", "text_token", "patch", "other"]

@dataclass(frozen=True)
class ActivationIndex:
    """
    Index describing how activation rows map back to dataset examples/tokens.

    For token-level storage (your scGPT case):
      - acts rows are token occurrences, so example_ids/token_ids are same length.
    For example-level storage:
      - acts rows align with example_ids; token_ids=None.
    """
    example_ids: Sequence[str]
    token_ids: Optional[Sequence[str]]
    token_unit: TokenUnit
    layer_names: Sequence[str]
