# interpTFM: Interpretable Foundation Model Analysis for sc/ST Data

`interpTFM` is a workflow for extracting, validating, and analyzing interpretable features from single-cell or spatial transcriptomics foundation models. The current implementation is centered on a 3-model comparison across **scGPT**, **Geneformer**, and **C2S-scale**, but the design is intended to be extensible to additional models as long as their activations can be exported into the expected activation-store format.

The workflow currently supports `.h5ad` input data. Most early steps only require an AnnData object with expression data and model-compatible gene identifiers. Some downstream biological and spatial analyses require additional annotations, especially cell type labels and spatial coordinates.

## Input data requirements

At minimum, the input should be an AnnData file:

```text
/path/to/data.h5ad
```

Recommended AnnData fields:

```text
.obs[celltype_col]     cell type annotation, for example author_cell_type
.obsm[spatial_key]     spatial coordinates, for example spatial
```

Some steps can run without cell types or spatial coordinates, but the full biological/spatial workflow expects them:

| Requirement                                | Used by                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------- |
| `.X` or model-compatible expression matrix | model extraction, SAE training, feature scoring                        |
| gene names / gene IDs                      | model tokenization, GO/F1 concept scoring                              |
| cell type labels                           | TIS, F1 interpretation, niche validation, crosstalk, immune follow-ups |
| spatial coordinates                        | niche analysis, spatial crosstalk, immune infiltration follow-ups      |

## Supported models and scalability

The repository currently contains a consolidated wrapper for three models:

```text
scGPT
Geneformer
C2S-scale
```

The same workflow can be expanded to other models if three pieces are provided:

1. an activation extraction script that writes activations in the expected format;
2. an SAE training/checking configuration for those activations;
3. model-specific metadata such as layer names, token handling, and store roots.

Do not assume every model can be run inside the same Python environment. In practice, extraction often needs the model's own conda environment.

## Workflow at a glance

The full conceptual workflow is:

```text
extract activations
→ TIS pre-SAE analysis
→ SAE train/check
→ activation QC
→ heldout F1 concept scoring
→ GO reduction
→ F1 QC / downstream summaries
→ build interpretable AnnData
→ niche sweep optional
→ niche validation
→ niche term analysis
→ shuffle-control spatial crosstalk
→ grouped heatmaps
→ immune infiltration follow-ups
```

In practice, we recommend splitting this into two phases.

## Phase 1: Model-environment-specific steps

Run activation extraction and TIS separately for each model or model family, inside the environment required by that model.

This is intentionally separate from the consolidated downstream wrapper because scGPT, Geneformer, and C2S-scale may require different dependencies, Python versions, CUDA expectations, or tokenizer/model packages.

### Extract activations

Each model should write activations to a model-specific activation root. Example layout:

```text
/path/to/model_store/
  activations/
    layer_0/
      shard_00000/
        activations.pt
        index.pt
    layer_1/
      shard_00000/
        activations.pt
        index.pt
```

For the current CosMx example, activation roots may be kept outside the main workflow run directory, for example:

```text
/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_scgpt_cosmx
/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_c2s_cosmx
/maiziezhou_lab2/yunfei/Projects/interpTFM/acts_extraction_geneformer_cosmx
```

A selected layer can later be symlinked or copied into the store root expected by downstream scripts, or the config can point directly to the activation store if the rest of the path structure is compatible.

### TIS pre-SAE analysis

TIS should be run before SAE training. Its role is to characterize model/layer behavior directly from extracted activations before the sparse autoencoder imposes an additional representation.

TIS is useful for:

* checking whether layers contain meaningful cell-type or biological structure;
* comparing candidate layers across models;
* selecting layers for SAE training and downstream interpretation;
* avoiding wasted SAE training on layers with weak or unstable signal.

Example TIS command shape:

```bash
python scripts/test_tis_3models.py \
  --labels scgpt c2sscale geneformer \
  --store-roots /path/to/scgpt_store /path/to/c2sscale_store /path/to/geneformer_store \
  --layers layer_4.norm2 layer_17 layer_4 \
  --pooling token mean token \
  --token-values '<cls>' NONE '<cls>' \
  --adata-path /path/to/data.h5ad \
  --out-root /path/to/tis_outputs
```

For all-layer screening, use the appropriate TIS search/grid script for the current branch. Do not guess filenames; check what exists in `scripts/` first.

```bash
find scripts -maxdepth 1 -type f | grep -Ei 'tis|seed|grid|hp'
```

## Phase 2: Mechanistic interpretation workflow

After activations are extracted and layers are selected, the rest of the analysis can be run in the recommended shared environment for `interpTFM`.

The consolidated wrapper should usually start at:

```text
sae_train_or_check
```

not at `extract` or `tis_pre_sae`, if those have already been handled in model-specific environments.

Recommended config settings when extraction and TIS are external:

```json
{
  "extract": {
    "mode": "skip",
    "require_existing": false
  },
  "tis_pre_sae": {
    "enabled": false,
    "seed_grid_enabled": false
  }
}
```

### Running the consolidated wrapper

Then run:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --stages sae_train_or_check activation_qc f1_heldout go_reduce f1_qc build_adata niche_validation niche_terms shuffle_control_crosstalk grouped_heatmaps immune_followups
```

Use `--dry-run` first to inspect commands, but note that dry-run does not create output files. Downstream stages that infer files generated by earlier stages may fail during a full dry-run on a fresh output directory. This is a limitation of the current wrapper, not necessarily a failure of the workflow.

### Running partial workflows

A safer staged approach is:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --stages sae_train_or_check activation_qc f1_heldout \
  --dry-run
```

Then run those stages for real:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --stages sae_train_or_check activation_qc f1_heldout
```

After F1 outputs exist, continue:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --stages go_reduce f1_qc build_adata niche_validation niche_terms shuffle_control_crosstalk grouped_heatmaps immune_followups \
  --dry-run
```

Then remove `--dry-run` to execute.

## Stage reference

### Extract activations

Extracts hidden-state activations from a foundation model for one or more layers. This stage is model-specific and often requires the model's own environment.

Current recommendation: run separately, outside the consolidated wrapper.

Outputs should include activation shards for each selected layer.

### TIS pre-SAE analysis

Runs TIS analysis directly on extracted model activations before SAE training. This helps compare candidate layers and decide which layer(s) to carry forward.

Current recommendation: run separately, outside the consolidated wrapper, especially when multiple model-specific environments are needed.

### SAE train/check

Either checks for existing SAE checkpoints or trains sparse autoencoders if configured to do so.

Typical modes:

```json
"sae": {
  "mode": "check"
}
```

or:

```json
"sae": {
  "mode": "train_if_missing",
  "run_tag": "nr_on__steps_6000__l1_3e-3",
  "steps": 6000,
  "l1": 0.003,
  "no_resample": true
}
```

The SAE converts dense model activations into sparse latent features that are easier to interpret.

### Activation QC

Audits SAE activations and checks for problems such as dead or rarely active neurons. This stage helps identify whether the SAE is usable before downstream biological interpretation.

### Heldout F1 concept scoring

Scores SAE features against biological concept labels using heldout data. This is used to estimate whether a latent feature reliably corresponds to a biological gene set or concept.

Recommended wording: heldout F1 with trained SAE features and max1-compatible scoring.

### GO reduction

Reduces redundant Gene Ontology or concept annotations so downstream summaries are less dominated by near-duplicate terms.

This stage depends on F1 result tables from `f1_heldout`.

### F1 QC / downstream summaries

Summarizes F1 results and related diagnostics. Depending on the current branch, this may call a downstream F1 summary script, latent analysis script, activation analysis script, or custom commands.

Because filenames have changed during development, check the script map in the workflow config before running this stage.

### Build interpretable AnnData

Builds an interpretable AnnData object where selected SAE-derived concept features are attached to cells. This is the bridge between model interpretation and standard single-cell/spatial analysis.

Typical filters include:

```text
SAE activation threshold
minimum F1
minimum true positives
top concepts per model
```

### Niche sweep optional

Runs parameter sweeps for spatial niche discovery, such as different radii, numbers of clusters, clustering methods, and random seeds.

This stage is optional and can be disabled after a parameter choice has been validated.

### Niche validation

Constructs and validates spatial niches from interpretable features. It can relabel niches by a target cell type, such as tumor 13, to make model comparisons more consistent.

For the current CosMx workflow, the working convention is:

```text
niche 0 = tumor-rich / tumor13-rich
niche 1 = intermediate / interface-like
niche 2 = tumor-poor / immune-stromal
```

### Niche term analysis

Analyzes which interpretable terms or concept features are enriched in each niche and compares them across models.

This helps move from spatial clusters to biological interpretation.

### Shuffle-control spatial crosstalk

Runs ligand-receptor-independent spatial crosstalk analysis using neighborhood adjacency and model-derived interpretable feature signals.

This should be described as interpretable spatial crosstalk, not ligand-receptor cell-cell communication.

The shuffle-control version is the canonical crosstalk stage. The old plain crosstalk script should not be treated as the main workflow stage.

### Grouped heatmaps

Usually a no-op in the wrapper because the shuffle-control crosstalk script already writes grouped heatmaps. Custom grouped heatmap commands can be added if needed.

### Immune infiltration follow-ups

Runs follow-up analyses around immune infiltration and tumor boundaries, including tumor-contact immune gradients, boundary asymmetry, immune-hot versus immune-cold tumor cells, CD8/macrophage balance, T-cell state near stromal/endothelial regions, and cross-model consensus immune-infiltration scores.

This stage should compute boundary/asymmetry from niche-labeled h5ads directly and should not depend on old crosstalk outputs such as `combined_boundary_density.csv`.

## Running partial workflows

Any subset of stages can be run with `--stages`:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --stages activation_qc f1_heldout
```

To rerun a completed stage, add `--force`:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --stages niche_validation niche_terms \
  --force
```

To run only one layer set:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --only-layer-set cosmx_scgpt_l4__c2sscale_l17__geneformer_l4
```

## Branch and script compatibility

This repository is under active development and script filenames may differ between branches. Before running a workflow config copied from another branch or session, check that every script path exists:

```bash
python - <<'PY'
from pathlib import Path
import json, difflib

cfg = json.loads(Path("configs/layer_workflow_3models_cosmx.json").read_text())
existing = sorted(p.name for p in Path("scripts").glob("*.py"))

missing = []
for key, script in cfg["scripts"].items():
    p = Path(script)
    ok = p.exists()
    print(f"{'OK' if ok else 'MISS'}  {key:28s}  {p}")
    if not ok:
        missing.append((key, script))
        close = difflib.get_close_matches(p.name, existing, n=8, cutoff=0.35)
        if close:
            print(" " * 36 + "nearby: " + ", ".join(close))

print("\nMissing:")
for key, script in missing:
    print(f"  {key}: {script}")
PY
```

Do not silently replace filenames with approximate matches. If a script is missing, either update the config to a verified current filename or disable the corresponding stage until the intended replacement is confirmed.

<!-- ## CosMx lung example configuration

Example paths used in the current 3-model CosMx workflow:

```text
project_root:
  /maiziezhou_lab2/yunfei/Projects/interpTFM

adata_path:
  /maiziezhou_lab2/yunfei/Projects/FM_temp/interGFM/ge_shards/cosmx_human_lung_sec8.h5ad

celltype_col:
  author_cell_type

spatial_key:
  spatial
```

Validated layer set:

```text
scGPT:      layer_4.norm2
C2S-scale:  layer_17
Geneformer: layer_4
```

Example command:

```bash
python scripts/run_layer_workflow_3models.py \
  --config configs/layer_workflow_3models_cosmx.json \
  --only-layer-set cosmx_scgpt_l4__c2sscale_l17__geneformer_l4 \
  --stages sae_train_or_check activation_qc f1_heldout go_reduce f1_qc build_adata niche_validation niche_terms shuffle_control_crosstalk grouped_heatmaps immune_followups
``` -->
