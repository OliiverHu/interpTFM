import scanpy as sc

ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"

ad = sc.read_h5ad(ADATA_PATH)
print(ad.obs)