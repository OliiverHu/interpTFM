from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import PreTrainedModel

from interp_pipeline.adapters.model_base import ModelAdapter, ModelSpec
from interp_pipeline.types.activations import TokenUnit
from interp_pipeline.types.dataset import StandardDataset

from interp_pipeline.c2s_local.load_model import load_c2s_model
from interp_pipeline.c2s_local.tokenizer import C2STokenizer
from interp_pipeline.c2s_local.processor import C2SProcessor


@dataclass
class C2SHandle:
    model: PreTrainedModel
    tokenizer: C2STokenizer
    processor: C2SProcessor
    device: torch.device


class C2SScaleAdapter(ModelAdapter):
    """
    Adapter for C2S-Scale models (GPT-NeoX based, e.g., Pythia).

    C2S converts gene expression to "cell sentences" - gene names ordered by
    descending expression, space-separated. The model then processes these
    as text using a causal language model.

    Key points:
    - model.gpt_neox.layers[i] contains transformer layers
    - Input is tokenized cell sentences (gene names as text)
    - Captures hidden states from transformer layers
    """

    def load(self, spec: ModelSpec) -> C2SHandle:
        device = torch.device(spec.device)
        cache_dir = spec.options.get("cache_dir") if spec.options else None
        max_genes = spec.options.get("max_genes") if spec.options else None

        model, tokenizer = load_c2s_model(
            model_name_or_path=spec.checkpoint,
            cache_dir=cache_dir,
            device=device,
        )
        processor = C2SProcessor(max_genes=max_genes)
        return C2SHandle(model=model, tokenizer=tokenizer, processor=processor, device=device)
    
    def _unwrap_base_model(self, model_handle: C2SHandle):
        lm = model_handle.model
        return getattr(lm, "_model", lm)

    def _get_transformer_layers(self, model_handle: C2SHandle):
        base = self._unwrap_base_model(model_handle)

        if hasattr(base, "gpt_neox") and hasattr(base.gpt_neox, "layers"):
            return base.gpt_neox.layers, "gpt_neox"

        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base.model.layers, "model.layers"

        raise ValueError(
            f"Unsupported model architecture: {type(base)}. "
            "Expected either base.gpt_neox.layers or base.model.layers."
        )
    
    def _get_traceable_transformer_layers(self, model_handle: C2SHandle):
        """
        Resolve the transformer block list from the NNsight-wrapped model object,
        so `.output.save()` is available inside `model.trace(...)`.
        """
        lm = model_handle.model

        # GPT-NeoX / Pythia path on wrapped model
        if hasattr(lm, "gpt_neox") and hasattr(lm.gpt_neox, "layers"):
            return lm.gpt_neox.layers, "gpt_neox"

        # Gemma / Llama-like wrapped path
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers, "model.layers"

        # Some wrappers may keep the HF model at _model, but this usually won't
        # expose NNsight proxy attributes. Keep as fallback only.
        if hasattr(lm, "_model"):
            base = lm._model
            if hasattr(base, "gpt_neox") and hasattr(base.gpt_neox, "layers"):
                return base.gpt_neox.layers, "_model.gpt_neox"
            if hasattr(base, "model") and hasattr(base.model, "layers"):
                return base.model.layers, "_model.model.layers"

        raise ValueError(
            f"Could not resolve traceable transformer layers from wrapped model type {type(lm)}"
        )

    def list_layers(self, model_handle: C2SHandle) -> List[str]:
        layers, layer_family = self._get_transformer_layers(model_handle)
        n = len(layers)
        print(f"[C2SScaleAdapter] detected layer stack: {layer_family} ({n} layers)")
        return [f"layer_{i}" for i in range(n)]

    def infer_token_unit(self, layer_name: str) -> TokenUnit:
        # currently only initialized with "gene"
        return "gene"

    def make_batches(
        self,
        dataset: StandardDataset,
        model_handle: C2SHandle,
        batch_size: int,
        max_genes: int,
        normalize: bool = True,
    ) -> Iterable[Dict[str, Any]]:
        """
        Create batches from dataset by converting expression to cell sentences.

        Required output keys for extractor:
        - cell_ids: list[str] length B
        - tokenized: tokenizer output with input_ids / attention_mask
        - genes_ranked: list[list[str]] one ranked gene list per sample
        - gene_spans: list[list[(gene, start, end)]] per sample, pre-truncation
        """
        adata = dataset.adata
        n = adata.n_obs
        processor = model_handle.processor

        # ----- dataset-specific block; keep for now if needed ----- #
        adata.obs["cell_type"] = adata.obs["author_cell_type"]
        adata.obs["cell_name"] = adata.obs_names
        adata.var_names = adata.var["feature_name"].astype(str).values
        # ---------------------------------------------------------- #

        if normalize:
            processor.normalize_adata(adata)

        arrow_dataset, _ = processor.adata_to_arrow(adata)
        formatted_hf_ds = processor.prompts_generation(arrow_dataset, n_genes=max_genes)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            arrow_dataset_batch = arrow_dataset.select(range(start, end))
            formatted_hf_ds_batch = formatted_hf_ds.select(range(start, end))

            cell_ids = [str(x) for x in adata.obs_names[start:end]]
            batch_input = list(formatted_hf_ds_batch["model_input"])
            cell_sentences = list(arrow_dataset_batch["cell_sentence"])

            tokenized = model_handle.tokenizer(batch_input)

            # Recover ranked genes from cell_sentence.
            # Assumes cell_sentence is the ordered gene sequence separated by spaces.
            genes_ranked: List[List[str]] = [
                str(sentence).split() for sentence in cell_sentences
            ]

            gene_spans: List[List[Tuple[str, int, int]]] = []
            for prompt, genes_this_cell in zip(batch_input, genes_ranked):
                spans = build_prompt_and_spans_from_rendered_prompt(
                    tokenizer=model_handle.tokenizer,
                    prompt=prompt,
                    genes_ranked=genes_this_cell,
                )
                gene_spans.append(spans)

            yield {
                "cell_ids": cell_ids,
                "cell_sentences": cell_sentences,
                "batch_input": batch_input,
                "tokenized": tokenized,
                "genes_ranked": genes_ranked,
                "gene_spans": gene_spans,   # pre-truncation spans
            }

    def forward_and_capture(
        self,
        model_handle: C2SHandle,
        batch: Dict[str, Any],
        layers: Sequence[str],
        capture_cfg: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and capture hidden states from specified layers.
        Uses the NNsight-wrapped model path so `.output.save()` works.
        """
        captured: Dict[str, torch.Tensor] = {}
        layer_idxs = [int(lname.split("_")[1]) for lname in layers]

        transformer_layers, layer_family = self._get_traceable_transformer_layers(model_handle)
        print(f"[C2SScaleAdapter] tracing layer stack: {layer_family}")

        with torch.no_grad(), model_handle.model.trace(batch["tokenized"]):
            for layer_name, idx in zip(layers, layer_idxs):
                captured[layer_name] = transformer_layers[idx].output.save()

        return captured
    
    def process_captured(
        self,
        captured: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert captured token-level hidden states into per-gene pooled activations.

        Returns:
            {
                layer_name: {
                    "acts": Tensor[N_genes_kept, H],
                    "tok": List[str],              # gene names
                    "ex": List[str],               # cell ids aligned to acts
                    "cell_acts": Tensor[N_cells_kept, H],
                    "cell_ids": List[str],         # aligned to cell_acts
                    "token_unit": "gene",
                },
                ...
            }
        """
        tokenized = batch["tokenized"]
        cell_ids: List[str] = batch["cell_ids"]
        gene_spans: List[List[Tuple[str, int, int]]] = batch["gene_spans"]

        pooling = batch.get("pooling", "mean")
        save_dtype = batch.get("save_dtype", "fp16")
        pool_dtype = batch.get("pool_dtype", "fp32")

        attention_mask = tokenized["attention_mask"]
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("tokenized['attention_mask'] must be a torch.Tensor")

        seq_lens = attention_mask.sum(dim=1).tolist()
        processed: Dict[str, Dict[str, Any]] = {}

        for layer_name, layer_out in captured.items():
            hs = layer_out
            if hasattr(hs, "value"):
                hs = hs.value
            if isinstance(hs, (tuple, list)):
                hs = hs[0]
            if not isinstance(hs, torch.Tensor):
                raise TypeError(f"{layer_name} is not a tensor after unwrapping; got {type(hs)}")

            hs_cpu = hs.detach().to("cpu")

            if pool_dtype == "fp32":
                hs_cpu = hs_cpu.float()
            elif pool_dtype == "fp16":
                hs_cpu = hs_cpu.half()
            else:
                raise ValueError(f"pool_dtype must be 'fp16' or 'fp32', got {pool_dtype}")

            padded_seq_len = hs_cpu.shape[1]

            pooled_rows: List[torch.Tensor] = []
            tok: List[str] = []
            ex: List[str] = []

            per_cell_vecs: List[torch.Tensor] = []
            per_cell_ids: List[str] = []

            for sample_idx, cell_id in enumerate(cell_ids):
                seq_len = int(seq_lens[sample_idx])
                pad_len = padded_seq_len - seq_len

                cell_gene_vecs: List[torch.Tensor] = []

                for gene, start_tok, end_tok in gene_spans[sample_idx]:
                    if end_tok > seq_len:
                        continue

                    start_tok_shifted = start_tok + pad_len
                    end_tok_shifted = end_tok + pad_len

                    gene_hs = hs_cpu[sample_idx, start_tok_shifted:end_tok_shifted, :]
                    if gene_hs.shape[0] == 0:
                        continue

                    if pooling == "mean":
                        vec = gene_hs.mean(dim=0)
                    elif pooling == "max":
                        vec = gene_hs.max(dim=0).values
                    elif pooling == "last":
                        vec = gene_hs[-1]
                    else:
                        raise ValueError(f"Unknown pooling={pooling}")

                    # keep an fp32 copy for stable cell averaging before save cast
                    cell_gene_vecs.append(vec.float())

                    if save_dtype == "fp16":
                        vec_to_store = vec.half()
                    elif save_dtype == "fp32":
                        vec_to_store = vec.float()
                    else:
                        raise ValueError(f"save_dtype must be 'fp16' or 'fp32', got {save_dtype}")

                    pooled_rows.append(vec_to_store)
                    tok.append(gene)
                    ex.append(cell_id)

                # mean over this cell's kept gene vectors
                if cell_gene_vecs:
                    cell_vec = torch.stack(cell_gene_vecs, dim=0).mean(dim=0)
                    if save_dtype == "fp16":
                        cell_vec = cell_vec.half()
                    elif save_dtype == "fp32":
                        cell_vec = cell_vec.float()

                    per_cell_vecs.append(cell_vec)
                    per_cell_ids.append(cell_id)

            if pooled_rows:
                acts = torch.stack(pooled_rows, dim=0)
            else:
                hidden_size = hs_cpu.shape[-1]
                dtype = torch.float16 if save_dtype == "fp16" else torch.float32
                acts = torch.empty((0, hidden_size), dtype=dtype)

            if per_cell_vecs:
                cell_acts = torch.stack(per_cell_vecs, dim=0)
            else:
                hidden_size = hs_cpu.shape[-1]
                dtype = torch.float16 if save_dtype == "fp16" else torch.float32
                cell_acts = torch.empty((0, hidden_size), dtype=dtype)

            processed[layer_name] = {
                "acts": acts,
                "tok": tok,
                "ex": ex,
                "cell_acts": cell_acts,
                "cell_ids": per_cell_ids,
                "token_unit": "gene",
            }

        return processed


def tokenize_with_leading_space(tok, s: str) -> List[int]:
    return tok.hf_tokenizer.encode(" " + s, add_special_tokens=False)

def find_subsequence(haystack: List[int], needle: List[int], start: int) -> int:
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

def build_prompt_and_spans_from_rendered_prompt(
    tokenizer: C2STokenizer,
    prompt: str,
    genes_ranked: List[str],
) -> List[Tuple[str, int, int]]:
    """
    Returns list of (gene, start_token_idx, end_token_idx), end exclusive.
    Uses sequential subsequence matching in the tokenized rendered prompt.
    """
    input_ids = tokenizer.hf_tokenizer.encode(prompt, add_special_tokens=True)

    spans: List[Tuple[str, int, int]] = []
    cursor = 0

    for g in genes_ranked:
        gid = tokenize_with_leading_space(tokenizer, g)
        pos = find_subsequence(input_ids, gid, cursor)

        if pos == -1:
            gid2 = tokenizer.hf_tokenizer.encode(g, add_special_tokens=False)
            pos = find_subsequence(input_ids, gid2, cursor)
            if pos == -1:
                raise RuntimeError(
                    f"Could not locate gene '{g}' in tokenized prompt.\n"
                    f"Prompt:\n{prompt}"
                )
            gid = gid2

        spans.append((g, pos, pos + len(gid)))
        cursor = pos + len(gid)

    return spans