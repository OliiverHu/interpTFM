from pathlib import Path
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.types.dataset import StandardDataset

if __name__ == "__main__":
    model_path = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
    LAYERS = ["layer_0", "layer_6", "layer_13"]
    BATCH_SIZE = 4

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

    subset = adata[adata.obs["shards"] == "shard_0"][:16].copy()
    dataset = StandardDataset(adata=subset, obs_key_map={})

    spec = ModelSpec(
        name="c2s-scale",
        checkpoint=model_path,
        device="cuda:0",
        options={
            "max_genes": 256,
            "cache_dir": None,
        },
    )

    adapter = C2SScaleAdapter()
    handle = adapter.load(spec)

    print("Detected layers:", adapter.list_layers(handle)[:5], "...")

    for batch in adapter.make_batches(
        dataset=dataset,
        model_handle=handle,
        batch_size=BATCH_SIZE,
        max_length=256,   # currently acting like n_genes in your adapter path
        normalize=True,
    ):
        batch["pooling"] = "last"
        batch["save_dtype"] = "fp16"
        batch["pool_dtype"] = "fp32"

        print("\nBatch keys:", list(batch.keys()))
        print("First 3 cell_ids:", batch["cell_ids"][:3])
        print("input_ids shape:", batch["tokenized"]["input_ids"].shape)
        print("attention_mask shape:", batch["tokenized"]["attention_mask"].shape)
        print("First prompt preview:", batch["batch_input"][0][:300])
        print("First 10 ranked genes:", batch["genes_ranked"][0][:10])
        print("First 5 spans:", batch["gene_spans"][0][:5])

        captured = adapter.forward_and_capture(
            model_handle=handle,
            batch=batch,
            layers=LAYERS,
            capture_cfg={},
        )

        print("\nCaptured layers:", list(captured.keys()))
        for layer_name, acts in captured.items():
            try:
                shape = acts.shape
            except Exception:
                shape = getattr(getattr(acts, "value", None), "shape", "unknown")
            print(f"  {layer_name}: {shape}")

        processed = adapter.process_captured(
            captured=captured,
            batch=batch,
        )

        print("\nProcessed outputs:")
        for layer_name, out in processed.items():
            acts = out["acts"]
            tok = out["tok"]
            ex = out["ex"]
            token_unit = out["token_unit"]

            print(f"\n[{layer_name}]")
            print("  acts shape:", acts.shape)
            print("  len(tok):", len(tok))
            print("  len(ex):", len(ex))
            print("  token_unit:", token_unit)
            print("  first 5 genes:", tok[:5])
            print("  first 5 cell ids:", ex[:5])

            assert acts.shape[0] == len(tok), f"{layer_name}: acts/tok mismatch"
            assert acts.shape[0] == len(ex), f"{layer_name}: acts/ex mismatch"

        print("\nSmoke test passed for first batch.")
        break