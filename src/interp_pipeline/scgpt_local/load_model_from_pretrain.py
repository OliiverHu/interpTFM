# %%
import os
import torch

from .model import scGPTModelClean, scGPTConfig
from .tokenizer import Tokenizer
from .util import load_pretrained


MAIN = __name__ == "__main__"
# %%
def create_clean_model_from_pretrain(model_dir, device="cpu"):
    model_file = os.path.join(model_dir, "best_model.pt")
    pretrained_params = torch.load(model_file, map_location=device)

    scgptcfg = scGPTConfig()
    scgptmodel = scGPTModelClean(scgptcfg)

    load_pretrained(scgptmodel, pretrained_params, verbose=True)
    scgptmodel.to(device)
    scgptmodel.eval()

    tokenizer = Tokenizer(os.path.join(model_dir, "vocab.json"), device=device)

    return scgptmodel, tokenizer

# %%
if MAIN:
    import sys
    import scanpy as sc
    import numpy as np

    BASE_PATH = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(BASE_PATH)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model_dir = "/maiziezhou_lab/zihang/interpTFM/scgpt_clean/whole-human-pretrain"

    model_file = os.path.join(model_dir, "best_model.pt")
    pretrained_params = torch.load(model_file, map_location=device)

    scgptcfg = scGPTConfig()
    scgptmodel = scGPTModelClean(scgptcfg)

    load_pretrained(scgptmodel, pretrained_params, verbose=True)
    scgptmodel.to(device)
    scgptmodel.eval()

    tokenizer = Tokenizer(os.path.join(model_dir, "vocab.json"), device=device)

    adata = sc.read_h5ad('/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad')
    adata.X = adata.X.toarray().astype(np.float32)

    # X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

    # # Identify rows where all elements are zero
    # all_zero_rows = np.all(X == 0, axis=1)

    # # Get indices of all-zero rows
    # zero_row_indices = np.where(all_zero_rows)[0]

    adata_64 = adata[:64, :].copy()
    gene_names = adata.var['feature_name'].to_numpy()

    max_length = 100
    batch_size = 64
   

    output = tokenizer(adata_64.X, gene_names, max_length=max_length)
    scgptmodel(*output)



# %%
# scgptmodel.state_dict()["transformer_encoder.layers.11.self_attn.in_proj_weight"] == pretrained_params["transformer_encoder.layers.11.self_attn.Wqkv.weight"]

# %%