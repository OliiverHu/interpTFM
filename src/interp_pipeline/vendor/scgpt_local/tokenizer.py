import json
from typing import Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch

from .util import to_device

# %%
class Tokenizer:
    def __init__(
        self,
        vocab_file: str,
        pad_token: str = "<pad>",
        pad_value: float = -2,
        cls_token: Optional[str] = "<cls>",
        mask_value: Optional[float] = -1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_token = pad_token
        self.pad_token_id = self.vocab[pad_token]
        self.pad_value = pad_value
        self.cls_token = cls_token
        self.mask_value = mask_value
        self.device = device if device is not None else "cpu"

    def __len__(self):
        return len(self.vocab)

    def __call__(
        self,
        count_matrix: np.ndarray,
        gene_names: np.ndarray,
        max_length: Optional[int] = None,
        include_zero_genes: bool = False,
        normalize: bool = True,
        add_cls: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if type(count_matrix) == torch.Tensor:
            examples = self._preprocess_torch(count_matrix, gene_names, include_zero_genes, normalize, add_cls)
        elif type(count_matrix) == np.ndarray:
            examples = self._preprocess_np(count_matrix, gene_names, include_zero_genes, normalize, add_cls)
        else:
            raise ValueError("count_matrix should be either numpy array or torch tensor")

        # examples = self._preprocess(count_matrix, gene_names, include_zero_genes, normalize, add_cls)
        return to_device(self._encode(examples, max_length, include_zero_genes), self.device)

    def _preprocess_np(
        self, count_matrix, gene_names, include_zero_genes, normalize, add_cls
    ):
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}
        """
        gene_ids = [self.vocab.get(gene, -1) for gene in gene_names]
        if not gene_ids.count(-1) == 0:
            print("Some gene names not in vocab")
        examples = []
        assert len(count_matrix.shape) == 2, "count_matrix should be 2D"
        assert count_matrix.shape[1] == len(gene_ids), "count_matrix shape does not match gene_names length"

        for i, expr in enumerate(count_matrix):
            genes = np.array(gene_ids)
            assert expr.sum(), f"Empty cell found at index {i}"

            if not include_zero_genes:
                nonzero_idx = np.nonzero(expr)[0]
                genes = genes[nonzero_idx]
                expr = expr[nonzero_idx]
                        
            if normalize:
                expr = np.log1p(expr)

            if add_cls and self.cls_token is not None:
                genes = np.insert(genes, 0, self.vocab[self.cls_token])
                expr = np.insert(expr, 0, self.pad_value)

            genes = torch.from_numpy(genes).long()
            expr = torch.from_numpy(expr).float()

            examples.append({
                "id": i,
                "genes": genes,
                "expressions": expr,
            })

        return examples

    def _preprocess_torch(
        self, count_matrix, gene_names, include_zero_genes, normalize, add_cls
    ):
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}
        """
        gene_ids = [self.vocab.get(gene, -1) for gene in gene_names]
        if not gene_ids.count(-1) == 0:
            print("Some gene names not in vocab")
        examples = []
        assert len(count_matrix.shape) == 2, "count_matrix should be 2D"
        assert count_matrix.shape[1] == len(gene_ids), "count_matrix shape does not match gene_names length"

        for i, expr in enumerate(count_matrix):
            genes = torch.tensor(gene_ids, device=expr.device, dtype=torch.long)
            assert expr.sum(), f"Empty cell found at index {i}"

            if not include_zero_genes:
                # get indices where expr != 0
                nonzero_idx = torch.nonzero(expr, as_tuple=False).squeeze(1)  # shape [n_nonzero]
                genes = genes[nonzero_idx]
                expr = expr[nonzero_idx]

            if normalize:
                # torch equivalent of np.log1p
                expr = torch.log1p(expr)

            if add_cls and self.cls_token is not None:
                cls_id = torch.tensor([self.vocab[self.cls_token]], device=genes.device, dtype=genes.dtype)
                pad_val = torch.tensor([self.pad_value], device=expr.device, dtype=expr.dtype)
                genes = torch.cat([cls_id, genes], dim=0)
                expr = torch.cat([pad_val, expr], dim=0)

            examples.append({
                "id": i,
                "genes": genes,
                "expressions": expr,
            })

        return examples
        
    def _encode(self, examples, max_length, include_zero_genes):
        max_ori_len = max(len(example["genes"]) for example in examples)
        if max_length is None:
            max_length = max_ori_len
        else:
            max_length = min(max_length, max_ori_len)
        
        padded_genes = []
        padded_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, max_length
            )  # torch tensors of length max_length
            padded_genes.append(genes)
            padded_expressions.append(expressions)
        
        padded_genes = torch.stack(padded_genes, dim=0)
        padded_expressions = torch.stack(padded_expressions, dim=0)

        if include_zero_genes:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = padded_genes.eq(self.pad_token_id)
            
        return padded_genes, padded_expressions, src_key_padding_mask
    
    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:  # sample or truncate
            return genes[:max_length], expressions[:max_length]
        else:  # pad
            return self._padding(genes, expressions, max_length)

    def _padding(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        device = genes.device
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token_id,
                    dtype=genes.dtype,
                    device=device,
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions

    def decode(self, gene_ids: List[int]) -> List[str]:
        return [
            [
                # self.id_to_token.get(gid, self.pad_token)
                self.id_to_token[gid]
                for gid in g_list if gid != self.pad_token_id
            ]
            for g_list in gene_ids
        ]
