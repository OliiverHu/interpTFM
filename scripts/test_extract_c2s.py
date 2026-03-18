from pathlib import Path
import shutil

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.types.dataset import StandardDataset
from interp_pipeline.extraction.c2s_extraction import extract_c2s_shard


if __name__ == "__main__":
    model_path = "/maiziezhou_lab2/yunfei/Projects/interpTFM-legacy/c2sscale/models/C2S-Scale-Gemma-2-2B"
    LAYERS = ["layer_0", "layer_6", "layer_13"]
    BATCH_SIZE = 4
    MAX_GENES = 256

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
            "max_genes": MAX_GENES,
            "cache_dir": None,
        },
    )

    adapter = C2SScaleAdapter()
    handle = adapter.load(spec)

    print("Detected layers:", adapter.list_layers(handle)[:5], "...")

    # ------------------------------------------------------------------
    # 1) Adapter-level smoke test (same as before, with cell-level outputs)
    # ------------------------------------------------------------------
    for batch in adapter.make_batches(
        dataset=dataset,
        model_handle=handle,
        batch_size=BATCH_SIZE,
        max_genes=MAX_GENES,
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
            cell_acts = out["cell_acts"]
            cell_ids = out["cell_ids"]

            print(f"\n[{layer_name}]")
            print("  gene acts shape:", acts.shape)
            print("  len(tok):", len(tok))
            print("  len(ex):", len(ex))
            print("  cell acts shape:", cell_acts.shape)
            print("  len(cell_ids):", len(cell_ids))
            print("  first 5 genes:", tok[:5])
            print("  first 5 gene cell ids:", ex[:5])
            print("  first 5 cell ids:", cell_ids[:5])

            assert acts.shape[0] == len(tok)
            assert acts.shape[0] == len(ex)
            assert cell_acts.shape[0] == len(cell_ids)

        print("\nAdapter-level smoke test passed for first batch.")
        break

    # ------------------------------------------------------------------
    # 2) New c2s_extraction API smoke test
    # ------------------------------------------------------------------
    output_dir = Path("/tmp/c2s_extraction_smoke")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    extract_c2s_shard(
        adata=subset,
        model_path=model_path,
        output_dir=str(output_dir),
        layers=LAYERS,
        shard=0,
        batch_size=BATCH_SIZE,
        max_genes=MAX_GENES,
        device="cuda:0",
        pooling="last",
        save_dtype="fp16",
        pool_dtype="fp32",
        normalize=True,
        cache_dir=None,
    )

    print("\nExtraction API smoke test finished. Checking saved files...")

    expected_files = [
        output_dir / "metadata" / "shard_0.json",
        output_dir / "activations" / "layer_0" / "shard_0" / "batch_00000_gene_acts.pt",
        output_dir / "activations" / "layer_0" / "shard_0" / "batch_00000_cell_gene_pairs.txt",
        output_dir / "cell_activations" / "layer_0" / "shard_0" / "batch_00000_cell_acts.pt",
        output_dir / "cell_activations" / "layer_0" / "shard_0" / "batch_00000_cell_ids.txt",
    ]

    for fp in expected_files:
        print(f"  exists? {fp}: {fp.exists()}")
        assert fp.exists(), f"Missing expected output file: {fp}"

    print("\nC2S extraction API smoke test passed.")