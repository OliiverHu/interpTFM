from pathlib import Path
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from interp_pipeline.extraction.c2s_extraction import extract_c2s_shard


if __name__ == "__main__":
    model_path = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
    output_dir = "/tmp/c2s_extraction_full_shard_test"

    LAYERS = ["layer_0", "layer_6", "layer_13"]
    BATCH_SIZE = 4
    MAX_GENES = 256
    SHARD_ID = 0

    p = Path("/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad")

    if not p.exists():
        adata_path = Path("/maiziezhou_lab2/yunfei/Projects/FM_temp/datasets/cosmx/lung/cosmx_human_lung.h5ad")
        adata = sc.read_h5ad(adata_path)

        adata = adata[adata.obs["library_key"] == 7].copy()

        X = adata.X
        if sp.issparse(X):
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            non_zero_mask = row_sums != 0
        else:
            X = X if isinstance(X, np.ndarray) else np.asarray(X)
            non_zero_mask = ~(np.all(X == 0, axis=1))

        adata = adata[non_zero_mask].copy()
        adata.obs["shards"] = np.random.choice([f"shard_{i}" for i in range(60)], size=adata.n_obs)
        adata.write(p)
    else:
        print("using cached")
        adata = sc.read_h5ad(p)

    subset = adata[adata.obs["shards"] == f"shard_{SHARD_ID}"].copy()
    print(f"Running full shard_{SHARD_ID} with {subset.n_obs} cells")

    extract_c2s_shard(
        adata=subset,
        model_path=model_path,
        output_dir=output_dir,
        layers=LAYERS,
        shard=SHARD_ID,
        batch_size=BATCH_SIZE,
        max_genes=MAX_GENES,
        device="cuda:0",
        pooling="last",
        save_dtype="fp16",
        pool_dtype="fp32",
        normalize=False,
        cache_dir=None,
    )

    print("Full-shard extraction finished.")