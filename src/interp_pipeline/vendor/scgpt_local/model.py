# %%
import os
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from tqdm import trange
import numpy as np

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import PretrainedConfig, PreTrainedModel


# %%
class scGPTConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=60697,
            embsize=512,
            d_hid=512,
            nlayers=12,
            nheads=8,
            max_seq_len=1200,
            dropout=0.2,
            pad_token_id=60694,
            input_emb_style="continuous",
            cell_emb_style="cls",  # output embedding vector with
            explicit_zero_prob=False,
            **kwargs):
        self.vocab_size = vocab_size
        self.embsize = embsize
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.nheads = nheads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        if input_emb_style not in ["continuous"]:
            raise ValueError(
                f"Invalid input_emb_style: {input_emb_style}. Only continuous embeddings currently supported."
            )
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        super().__init__(pad_token_id=pad_token_id, **kwargs)

# %%
class ScGPTPreTrainedModel(PreTrainedModel):
    config_class = scGPTConfig
    base_model_prefix = "scgpt"

# %%
class scGPTModelClean(ScGPTPreTrainedModel):
    def __init__(self, config: scGPTConfig):
        super().__init__(config)
        # self.config = config
        d_model = config.embsize
        # self.ecs_threshold = ecs_threshold
        if config.input_emb_style not in ["continuous"]:
            raise ValueError(
                f"only continuous is supported for input_emb_style but got {config.input_emb_style}"
            )
        if config.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {config.cell_emb_style}")

        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(config.vocab_size, d_model, padding_idx=config.pad_token_id)

        ## Input Value Encoder
        self.value_encoder = ContinuousValueEncoder(d_model, config.dropout)

        # self.pert_encoder = nn.Embedding(3, d_model, padding_idx=0)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model,
                config.nheads,
                config.d_hid,
                config.dropout,
                batch_first=True
            ),
            config.nlayers
        )

        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=config.explicit_zero_prob,
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        if self.config.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.config.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.config.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.config.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def predict(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(
            src, values, src_key_padding_mask
        )

        output = {}
        decoder_output = self.decoder(transformer_output)
        output.update(decoder_output)
        
        if self.config.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=decoder_output["zero_probs"])
            output["bernoulli_pred"] = bernoulli.sample() * decoder_output["pred"]
        else:
            output["pred"] = decoder_output["pred"]  # (batch, seq_len)
        
        if self.config.explicit_zero_prob:
            output["zero_probs"] = decoder_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb

        return output


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

