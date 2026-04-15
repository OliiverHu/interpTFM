# %%
import os
import sys
import pickle
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import einops
from tqdm import tqdm
from collections import defaultdict

import scanpy as sc
# import anndata as ad
import numpy as np
import torch
from nnsight import NNsight
from gears import PertData, GEARS

BASE_PATH = Path(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_PATH.as_posix())

from scgpt_clean.load_model_from_pretrain import create_clean_model_from_pretrain

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# %%
data_set_name = "norman"
SAVE_PATH = Path(f"/maiziezhou_lab2/zihang/interpTFM/data/lp_training/filtered_data_{data_set_name}/")

model_dir = BASE_PATH / "data/whole-human-pretrain"
scgptmodel, tokenizer = create_clean_model_from_pretrain(model_dir, device=device)
scgpt_nn = NNsight(scgptmodel)

# %%
# concept_matrix_dir = Path("/maiziezhou_lab2/yunfei/Projects/interpTFM/") / "gprofiler_annotation" / "gene_concepts.csv"
# concept_gene_map = pd.read_csv(concept_matrix_dir, index_col=0)
# concept_gene_map = concept_gene_map.sort_index()

# current_genes = concept_gene_map.columns.to_numpy()

# adata_cosmx = sc.read_h5ad('/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad')

# adata_part1 = ad.read_h5ad("/maiziezhou_lab/zihang/interpTFM/data/perturbation/replogle_2022_part1.h5ad")
# adata_part2 = ad.read_h5ad("/maiziezhou_lab/zihang/interpTFM/data/perturbation/replogle_2022_part2.h5ad")
# adata_part3 = ad.read_h5ad("/maiziezhou_lab/zihang/interpTFM/data/perturbation/replogle_2022_part3.h5ad")
# adata_full = ad.concat([adata_part1, adata_part2, adata_part3], axis=0, join='outer')
# adata_full.write_h5ad('/maiziezhou_lab/zihang/interpTFM/data/perturbation/replogle_2022.h5ad')

# adata_full = ad.read_h5ad('/maiziezhou_lab/zihang/interpTFM/data/perturbation/replogle_2022.h5ad')

# %%
# pert_data = PertData(BASE_PATH / "data/perturbation_gears")
# data_set_name = "norman"
# pert_data.load(data_name=data_set_name)

# adata = pert_data.adata

adata = sc.read_h5ad(BASE_PATH / f"data/perturbation_gears/{data_set_name}/perturb_processed.h5ad")

# print("KLF1+MAP2K6" in pert_data.adata.obs["condition"].tolist())
# print("KLF1+ctrl" in pert_data.adata.obs["condition"].tolist())
# print("MAP2K6+ctrl" in pert_data.adata.obs["condition"].tolist())

# %% data info
np.random.seed(42)
n_cells = adata.n_obs
n_shards = 60

idx = np.random.permutation(n_cells)

# split into roughly equal shards
shards = np.array_split(idx, n_shards)

# create list of shard AnnData objects
# adata_shards = [adata[shard_idx].copy() for shard_idx in shards]

gene_names = adata.var['gene_name'].to_numpy()
gene_ids = [tokenizer.vocab.get(gene, -1) for gene in gene_names]
missing_genes = gene_names[np.array(gene_ids) == -1]
remaining_genes = gene_names[np.array(gene_ids) != -1]

# %% construct concept matrix from gprofiler annotation
# df = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/{data_set_name}_go_kegg_reac.csv")
# df_sorted = df.sort_values(by='term_name', ascending=True)
# high_conf_evidence = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}

# def has_high_conf(cell):
#     if pd.isna(cell):
#         return 0
#     codes = [code.strip() for code in str(cell).split(',')]
#     return int(any(code in high_conf_evidence for code in codes))

# # Apply the function to annotation columns
# binary_matrix = df_sorted.iloc[:, 10:].applymap(has_high_conf)
# # concepts = df_sorted['term_name'].to_list()
# df_sorted['term_name'].to_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gprofiler_gene_concepts_columns.txt")
# binary_matrix.to_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts_ens.csv")

# #####
# mapping = dict(zip(adata.var.index, adata.var["gene_name"]))
# binary_matrix_renamed = binary_matrix.rename(columns=mapping)
# binary_matrix_renamed.to_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts.csv")

# binary_matrix_renamed_nonzero = binary_matrix_renamed.loc[(binary_matrix_renamed != 0).any(axis=1), (binary_matrix_renamed != 0).any(axis=0)]
# binary_matrix_renamed_nonzero.to_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts_nonzero.csv")

# #####
# binary_matrix_mat = binary_matrix_renamed_nonzero.to_numpy()
# print(binary_matrix_mat.shape)
# total_entries = binary_matrix.size

# # Number of zero entries
# zero_entries = (binary_matrix == 0).sum().sum()

# # Zero rate
# zero_rate = zero_entries / total_entries

# print(f"Zero rate: {zero_rate:.4f} ({zero_entries} out of {total_entries})")

# %%
binary_matrix_renamed_nonzero = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts_nonzero.csv", index_col=0)

# binary_matrix_renamed.to_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts.csv")

# binary_matrix_renamed = pd.read_csv(BASE_PATH / f"data/gprofiler_annotation/{data_set_name}/gene_concepts.csv", index_col=0)

# %%
# i_shard = 0
for i_shard, shard_idx in enumerate(shards):
    print(f"Processing shard {i_shard}")
    torch.cuda.empty_cache()

    filtered_genes_save_dir = SAVE_PATH / "gene_ids" / f"shard_{i_shard}" / "filtered_genes.csv"
    filtered_concepts_save_dir = SAVE_PATH / "gene_ids" / f"shard_{i_shard}" / "filtered_concepts.pt"
    filtered_genes_number_dir = SAVE_PATH / "gene_ids" / f"shard_{i_shard}" / "filtered_genes_number.txt"

    filtered_genes_save_dir.parent.mkdir(parents=True, exist_ok=True)
    filtered_concepts_save_dir.parent.mkdir(parents=True, exist_ok=True)
    filtered_genes_number_dir.parent.mkdir(parents=True, exist_ok=True)

    gene_token_list = []
    cell_list = []
    activation_list = defaultdict(list)

    # shard_idx = np.array([_ for _ in range(0,200)])
    adata_select = adata[shard_idx, np.array(gene_ids) >= 0].copy()
    gene_names_select = adata_select.var['gene_name'].to_numpy()
    adata_select.X = adata_select.X.toarray()

    batch_size = 128
    for start in tqdm(range(0, adata_select.n_obs, batch_size)):
        end = min(start + batch_size, adata_select.n_obs)
        batch_adata = adata_select[start:end].copy()

        genes, expressions, attention_mask = tokenizer(
            batch_adata.X, gene_names_select,
            include_zero_genes=False, normalize=False, add_cls=True
        )
        activations_dict = {}
        with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
            for layer in range(scgpt_nn.config.nlayers):
                activations_dict[layer] = scgpt_nn.transformer_encoder.layers[layer].norm2.output.save()
        
        gene_token_list += tokenizer.decode(genes.cpu().tolist())
        cell_list += batch_adata.obs.index.to_list()
        for layer, activations in activations_dict.items():
            activation_list[layer] += list(activations.cpu().unbind())

    flattened = [(cell, gene) for cell, genes in zip(cell_list, gene_token_list) for gene in genes]
    meta_df = pd.DataFrame(flattened, columns=['cell_id', 'gene_token'])
    # activation_tensor = {}
    # for layer in range(scgpt_nn.config.nlayers):
    #     activation_tensor[layer] = torch.cat(activation_list[layer], dim=0)

    mask = meta_df['gene_token'].isin(binary_matrix_renamed_nonzero.columns)
    filtered_genes = meta_df[mask].reset_index(drop=True)
    filtered_concepts = torch.tensor(binary_matrix_renamed_nonzero[filtered_genes['gene_token']].values.T, dtype=torch.float32)

    with open(filtered_genes_save_dir, 'wb') as f:
        pickle.dump(filtered_genes, f)
        
    torch.save(filtered_concepts, filtered_concepts_save_dir)

    for layer in range(scgpt_nn.config.nlayers):
        activations = torch.cat(activation_list[layer], dim=0)
        filtered_activations = activations[mask.values]   # torch indexing

        filtered_save_dir = SAVE_PATH / "activations" / f"layer_{layer}" / f"shard_{i_shard}" / "filtered_activations.pt"
        filtered_save_dir.parent.mkdir(parents=True, exist_ok=True)
        torch.save(filtered_activations, filtered_save_dir)

    with open(filtered_genes_number_dir, 'wb') as f:
        pickle.dump(mask.sum().item(), f)


# %% test
# count = 0
# for i, gene_list in enumerate(gene_token_list):
#     count += len(gene_list)
#     for layer in range(scgpt_nn.config.nlayers):
#         if not (len(gene_list) == len(activation_list[layer][i])):
#             print(f"Mismatch at cell {i}, layer {layer}: gene tokens length {len(gene_list)} vs expressions length {len(activation_list[layer][i])}")

# %%
# with open(BASE_PATH / "data/lp_training/filtered_data_cosmax/gene_ids/shard_1/filtered_genes_number.txt", "rb") as f:
#     filtered_gene_number = pickle.load(f) 

# with open(BASE_PATH / "data/lp_training/filtered_data_cosmax/gene_ids/shard_1/filtered_genes.csv", 'rb') as f:
#     filtered_genes = pickle.load(f)

# concepts = torch.load(BASE_PATH / "data/lp_training/filtered_data_cosmax/gene_ids/shard_1/filtered_concepts.pt")

# old_concept_dir = BASE_PATH / "gprofiler_annotation/cosmx_lung_human_gp_go_kegg_reactome.csv"
# concept_gene_map = pd.read_csv(BASE_PATH / "gprofiler_annotation/gene_concepts.csv", index_col=0)

# %%
# with torch.no_grad(), scgpt_nn.trace(genes, expressions, attention_mask):
#     # test = scgpt_nn.transformer_encoder.layers[-1].linear1.output.save()
#     test = scgpt_nn.transformer_encoder.layers[-1].norm2.output.save()
#     # test = scgpt_nn.output.save()




# stacked = torch.stack([_ for _ in test], dim=0)









# %%
# adata_raw = ad.read_h5ad('/maiziezhou_lab/zihang/interpTFM/data/perturbation_feng_second_hand/raw_pert.h5ad')

# %%
# adata_feng = ad.read_h5ad('/maiziezhou_lab/zihang/interpTFM/data/perturbation_feng/GSE216595_180124_perturb.h5ad')

# %%
# import scanpy as sc
# import scipy.io
# import pandas as pd
# import anndata as ad

# mat = scipy.io.mmread("/maiziezhou_lab/zihang/interpTFM/data/perturbation_feng/GSE216595_RAW/GSM6681046_180124_perturb_S8_matrix.mtx.gz").T.tocsc()
# genes = pd.read_csv("/maiziezhou_lab/zihang/interpTFM/data/perturbation_feng/GSE216595_RAW/GSM6681046_180124_perturb_S8_genes.tsv.gz", sep="\t", header=None)
# barcodes = pd.read_csv("/maiziezhou_lab/zihang/interpTFM/data/perturbation_feng/GSE216595_RAW/GSM6681046_180124_perturb_S1_barcodes.tsv.gz", sep="\t", header=None)
# tf_map_fp = "/maiziezhou_lab/zihang/interpTFM/data/perturbation_feng/GSE216595_RAW/GSM6681047_180124_TFmap.csv.gz"
# tf_map = pd.read_csv(tf_map_fp, compression="gzip")
