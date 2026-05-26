#!/usr/bin/env python3
from interp_pipeline.layer_experiments.all_layer_sae_f1 import main

if __name__ == "__main__":
    main()

# python scripts/archived/run_all_layer_sae_f1.py \
#   --out-root runs/all_layer_sae_f1_cosmx \
#   --models scgpt c2sscale geneformer \
#   --stages f1 \
#   --steps 8000 \
#   --l1 1e-3 \
#   --gt-csv /maiziezhou_lab2/yunfei/Projects/interpTFM/runs/heldout_3models_l1_3e-3_geneheldout/gprofiler/gprofiler_binary_gene_by_term.csv \
#   --adata-path /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad \
#   --f1-mode cell \
#   --force