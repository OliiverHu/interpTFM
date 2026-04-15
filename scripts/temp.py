import scanpy as sc

adata = sc.read_h5ad("/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad")

print("var columns:")
print(list(adata.var.columns))

print("\nfirst 10 var rows:")
print(adata.var.head(10))

print("\nfirst 20 var_names:")
print(list(adata.var_names[:20]))