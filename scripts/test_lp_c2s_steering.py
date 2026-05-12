"""
C2S activation steering evaluation — analogous to test_lp_eval_steering.py for scGPT.

Three cell embedding groups captured at the final transformer layer,
pooled over gene BPE spans (same method as C2S extraction):
  ctrl      — unsteered ctrl cells (Norman)
  pert      — GENE_SELECT-overexpressing cells (Norman)
  ctrl+LP   — ctrl cells with probe direction injected at PROBE_LAYER

Probe direction is taken from the probe trained at PROBE_LAYER (geometrically
meaningful in that feature space).  The final transformer layer is the C2S
analogue of scGPT's CLS token from model.output.

Requires:
  - scripts/test_lp_c2s_acts.py     : activations + concept matrix
  - scripts/test_lp_c2s_training.py : trained probe at PROBE_LAYER
"""
# %% ── Config ─────────────────────────────────────────────────────────────────
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)

PROBE_PATH_TMPL = "runs/lp_c2s_norman/linear_probes/probe_layer_{layer}.pt"
CONCEPT_CSV   = "runs/lp_c2s_norman/gprofiler/concept_matrix.csv"
ADATA_PATH    = "/maiziezhou_lab2/zihang/interpTFM/temp/norman/perturb_c2s_prepped.h5ad"
C2S_MODEL     = "vandijklab/C2S-Pythia-410m-cell-type-conditioned-cell-generation"
CACHE_DIR     = "/maiziezhou_lab2/zihang/interpTFM/cache/c2s"
OUT_DIR       = "runs/lp_c2s_norman/eval_steering"
DEVICE        = "cuda"

GENE_SELECT    = "CEBPA"
PROBE_LAYERS   = [12, 17]   # iterate over both trained probe layers
CONDITION_COL  = "condition"
CONDITION_PERT = None     # auto-detect ("CEBPA+ctrl" / "ctrl+CEBPA" / "CEBPA")
SCALE_LIST_OPT1 = [-5.0, -4.0, -3.0, -2.0, -1.0]  # negative only: suppress pert → ctrl
SCALE_LIST_OPT3 = [1.0, 2.0, 3.0, 4.0, 5.0]        # positive only: amplify ctrl → pert
MAX_GENES      = 256      # must match extraction
POOLING        = "last"   # BPE-span pooling (must match extraction)
STEERING_MODE  = "add"      # default for options 1/3/4; option 5 uses "amplify"

# %% ── 1. Load probe and concept matrix ───────────────────────────────────────
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm

from interp_pipeline.adapters.model_base import ModelSpec
from interp_pipeline.adapters.models.c2s_scale import C2SScaleAdapter
from interp_pipeline.linear_probe import load_probe
from interp_pipeline.types.dataset import StandardDataset
from interp_pipeline.steering.analysis import score_steering_regression, plot_steering_umap
from interp_pipeline.steering.intervene import apply_scale_2D

concept_df = pd.read_csv(CONCEPT_CSV, index_col=0)
print(f"Concept matrix: {concept_df.shape[0]} terms × {concept_df.shape[1]} genes")

# %% ── 2. Build probe direction ───────────────────────────────────────────────

if GENE_SELECT not in concept_df.columns:
    raise ValueError(
        f"'{GENE_SELECT}' not in concept matrix. Sample cols: {list(concept_df.columns[:10])}"
    )

gene_col          = concept_df[GENE_SELECT]
concept_idx_union = list(np.where(gene_col.values == 1)[0])
concept_names     = list(concept_df.index[concept_idx_union])
print(f"{GENE_SELECT}: {len(concept_idx_union)} GO terms  sample: {concept_names[:5]}")

# Load each probe and slice out the GO-concept columns for GENE_SELECT.
# Store the raw [H, n_G_concepts] matrix so apply_scale_2D normalizes each
# column independently (consistent with the scGPT intervene.py approach).
probe_dirs_list: list[torch.Tensor] = []
for pl in PROBE_LAYERS:
    _probe = load_probe(PROBE_PATH_TMPL.format(layer=pl), device=DEVICE)
    _dirs  = _probe.weight[:, concept_idx_union].detach().to(DEVICE)   # [H, n_G_concepts]
    probe_dirs_list.append(_dirs)
    print(f"  Layer {pl} probe  weight={_probe.weight.shape}  dirs={_dirs.shape}  n_concepts={_dirs.shape[1]}")

# %% ── 3. Load C2S model ──────────────────────────────────────────────────────

adapter = C2SScaleAdapter()
handle  = adapter.load(ModelSpec(
    name="c2s-scale",
    checkpoint=C2S_MODEL,
    device=DEVICE,
    options={"max_genes": MAX_GENES, "cache_dir": CACHE_DIR},
))
transformer_layers, layer_family = adapter._get_traceable_transformer_layers(handle)
n_layers    = len(list(transformer_layers))
FINAL_LAYER = n_layers - 1
print(f"Model loaded — {n_layers} layers ({layer_family})  FINAL_LAYER={FINAL_LAYER}")
assert all(pl < n_layers for pl in PROBE_LAYERS), f"All PROBE_LAYERS must be < {n_layers}"

# %% ── 4. Load Norman adata ───────────────────────────────────────────────────

adata      = sc.read_h5ad(ADATA_PATH)
adata_ctrl = adata[adata.obs[CONDITION_COL] == "ctrl"].copy()

_cond_pert = CONDITION_PERT
if _cond_pert is None:
    all_conds = set(adata.obs[CONDITION_COL].unique())
    for candidate in (f"{GENE_SELECT}+ctrl", f"ctrl+{GENE_SELECT}", GENE_SELECT):
        if candidate in all_conds:
            _cond_pert = candidate
            break
if _cond_pert is None:
    matching = [c for c in adata.obs[CONDITION_COL].unique() if GENE_SELECT in c]
    raise ValueError(
        f"Cannot find a condition for '{GENE_SELECT}'. Matching conditions: {matching}. "
        "Set CONDITION_PERT explicitly."
    )
print(f"Perturbation condition: '{_cond_pert}'")

adata_pert = adata[adata.obs[CONDITION_COL] == _cond_pert].copy()

# Subsample ctrl to pert count for a balanced comparison (matches scGPT approach).
n_pert = adata_pert.n_obs
rng = np.random.default_rng(42)
ctrl_names = list(adata_ctrl.obs_names)
rng.shuffle(ctrl_names)
ctrl_names = ctrl_names[:n_pert]
adata_ctrl = adata_ctrl[ctrl_names].copy()
print(f"ctrl: {adata_ctrl.n_obs}  |  pert ({_cond_pert}): {adata_pert.n_obs}")

# %% ── 5. Cell embedding helper ───────────────────────────────────────────────

def _run_cell_emb(
    cell_adata,
    alpha: float,
    probe_layers: list = (),
    probe_dirs: list = (),
    injection_pos: str = "gene",
    mode: str = "add",
) -> torch.Tensor | None:
    """
    Run C2S forward for one cell, optionally injecting alpha * probe_dir at
    each layer in probe_layers simultaneously.

    injection_pos="gene": inject at GENE_SELECT's last BPE token.
                          Returns None if GENE_SELECT is absent from the cell.
    injection_pos="last": inject at the last valid token in the cell sentence.
                          Works for any cell regardless of GENE_SELECT presence.
    injection_pos="all":  inject at every valid (non-padding) token position.
                          Broadest intervention; always succeeds.

    Returns cell embedding [H] = mean of per-gene BPE-span-pooled hidden states
    at FINAL_LAYER, identical to C2S extraction (process_captured).
    """
    dataset = StandardDataset(adata=cell_adata, obs_key_map={})
    batches = list(adapter.make_batches(
        dataset, handle, batch_size=1, max_genes=MAX_GENES, normalize=False
    ))
    if not batches:
        return None

    batch      = batches[0]
    gene_spans = batch["gene_spans"][0]   # [(gene, start_tok, end_tok), ...] unpadded
    tokenized  = batch["tokenized"]

    seq_len    = int(tokenized["attention_mask"].sum(dim=1)[0].item())
    padded_len = tokenized["attention_mask"].shape[1]
    pad_len    = padded_len - seq_len  # C2S uses left-padding

    # Resolve injection position before entering the trace context.
    tok_pos = None
    if alpha != 0.0 and injection_pos != "all":
        if injection_pos == "gene":
            for gene, start, end in gene_spans:
                if gene == GENE_SELECT and end <= seq_len:
                    tok_pos = (end - 1) + pad_len
                    break
            if tok_pos is None:
                return None
        else:  # "last" — last valid gene's last BPE token, always present
            for gene, start, end in reversed(gene_spans):
                if end <= seq_len:
                    tok_pos = (end - 1) + pad_len
                    break
            if tok_pos is None:
                return None

    with torch.no_grad(), handle.model.trace(tokenized):
        if alpha != 0.0:
            for pl, dirs in zip(probe_layers, probe_dirs):
                hs = transformer_layers[pl].output   # [B, T, H] proxy
                if injection_pos == "all":
                    for _pos in range(pad_len, padded_len):
                        apply_scale_2D(hs, dirs, alpha, _pos, mode=mode)
                else:
                    apply_scale_2D(hs, dirs, alpha, tok_pos, mode=mode)
        final_out = transformer_layers[FINAL_LAYER].output.save()

    hs_saved = final_out.value if hasattr(final_out, "value") else final_out
    if isinstance(hs_saved, (tuple, list)):
        hs_saved = hs_saved[0]
    hs_cpu = hs_saved.detach().cpu().float()  # [B, T, H]

    # Pool per-gene using BPE spans (matches process_captured in c2s_scale.py).
    gene_vecs = []
    for gene, start, end in gene_spans:
        if end > seq_len:
            continue
        s, e = start + pad_len, end + pad_len
        token_hs = hs_cpu[0, s:e, :]
        if token_hs.shape[0] == 0:
            continue
        vec = token_hs[-1] if POOLING == "last" else token_hs.mean(0)
        gene_vecs.append(vec)

    if not gene_vecs:
        return None

    return torch.stack(gene_vecs, 0).mean(0)  # mean over genes → [H]

# %% ── 6. Collect unsteered ctrl and pert embeddings ─────────────────────────

_acts_dir   = os.path.join(OUT_DIR, "activations", GENE_SELECT)
_safe       = lambda s: s.replace("/", "_").replace("+", "_")
_ctrl_cache = os.path.join(_acts_dir, "ctrl_cell_embs.pt")
_pert_cache = os.path.join(_acts_dir, f"{_safe(_cond_pert)}_cell_embs.pt")

if os.path.exists(_ctrl_cache) and os.path.exists(_pert_cache):
    print("\n[6] Loading cached unsteered embeddings...")
    ctrl_act = torch.load(_ctrl_cache, map_location="cpu")
    pert_act = torch.load(_pert_cache, map_location="cpu")
else:
    print("\n[6] Collecting unsteered ctrl embeddings...")
    ctrl_vecs = []
    for name in tqdm(ctrl_names, desc="ctrl"):
        emb = _run_cell_emb(adata_ctrl[[name]].copy(), alpha=0.0)
        if emb is not None:
            ctrl_vecs.append(emb)
    ctrl_act = torch.stack(ctrl_vecs, 0)

    print("Collecting unsteered pert embeddings...")
    pert_vecs = []
    for name in tqdm(list(adata_pert.obs_names), desc="pert"):
        emb = _run_cell_emb(adata_pert[[name]].copy(), alpha=0.0)
        if emb is not None:
            pert_vecs.append(emb)
    pert_act = torch.stack(pert_vecs, 0)

    os.makedirs(_acts_dir, exist_ok=True)
    torch.save(ctrl_act, _ctrl_cache)
    torch.save(pert_act, _pert_cache)

print(f"    ctrl: {ctrl_act.shape}  |  pert: {pert_act.shape}")

# %% ── 6.5. Norm diagnostics — calibrate alpha relative to hidden state scale ──
# For each probe layer: compare hidden state norm vs. injection norm (alpha * probe_dir).
# A meaningful intervention needs alpha * ||probe_dir|| to be non-negligible relative
# to the hidden state norm at that layer.

_diag_dataset  = StandardDataset(adata=adata_ctrl[ctrl_names[:1]].copy(), obs_key_map={})
_diag_batches  = list(adapter.make_batches(_diag_dataset, handle, batch_size=1,
                                           max_genes=MAX_GENES, normalize=False))
if _diag_batches:
    _db         = _diag_batches[0]
    _dtok       = _db["tokenized"]
    _dseq_len   = int(_dtok["attention_mask"].sum(dim=1)[0].item())
    _dpad_len   = _dtok["attention_mask"].shape[1] - _dseq_len

    print(f"\n[6.5] Hidden state norm diagnostics (one ctrl cell, mode={STEERING_MODE}):")
    for pl, dirs in zip(PROBE_LAYERS, probe_dirs_list):
        with torch.no_grad(), handle.model.trace(_dtok):
            _hs = transformer_layers[pl].output              # [B, T, H]
            _norm_mean = _hs[:, _dpad_len:, :].norm(dim=-1).mean().save()
            _norm_last = _hs[:, -1, :].norm().save()
            # Projection of last valid token onto concept directions (amplify baseline).
            _dirs_n = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=1e-8)
            _proj   = (_hs[:, -1, :] @ _dirs_n).abs().mean().save()

        _nm   = float(_norm_mean.value if hasattr(_norm_mean, "value") else _norm_mean)
        _nl   = float(_norm_last.value if hasattr(_norm_last, "value") else _norm_last)
        _proj_val = float(_proj.value if hasattr(_proj, "value") else _proj)
        print(f"  Layer {pl}:")
        print(f"    hs norm  mean={_nm:.3f}  last={_nl:.3f}")
        print(f"    mean |projection onto concept dirs|={_proj_val:.4f}  (amplify baseline per direction)")
        for _a in SCALE_LIST_OPT1 + SCALE_LIST_OPT3:
            print(f"    scale={_a:+.1f}  amplify adds ≈ {abs(_a)*_proj_val:.4f} per dir to last token")

# Cache tag encodes the layers being intervened on simultaneously.
_layer_tag   = "l" + "_".join(str(pl) for pl in PROBE_LAYERS)  # e.g. "l12_17"

# %% ── 7a. Option 1: steer pert cells (GENE_SELECT guaranteed present) ────────
# Inject at GENE_SELECT's position at each probe layer; negative alpha suppresses toward ctrl.

_intv1_cache = os.path.join(_acts_dir, f"intv1_pert_{GENE_SELECT}_{_layer_tag}.pt")

if os.path.exists(_intv1_cache):
    print(f"\n[7a] Loading cached option-1 steered embeddings...")
    intv_acts_pert = torch.load(_intv1_cache, map_location="cpu")
else:
    print(f"\n[7a] Option 1 — steering pert cells at scales {SCALE_LIST_OPT1} (layers {PROBE_LAYERS})...")
    intv_acts_pert: dict[float, torch.Tensor] = {}
    for scale in SCALE_LIST_OPT1:
        vecs = []
        for name in tqdm(list(adata_pert.obs_names), desc=f"pert scale={scale:+.1f}"):
            emb = _run_cell_emb(adata_pert[[name]].copy(), alpha=scale,
                                probe_layers=PROBE_LAYERS, probe_dirs=probe_dirs_list,
                                injection_pos="gene")
            if emb is not None:
                vecs.append(emb)
        if vecs:
            intv_acts_pert[scale] = torch.stack(vecs, 0)
        print(f"    scale={scale:+.1f}  →  {intv_acts_pert.get(scale, torch.empty(0)).shape}")
    torch.save(intv_acts_pert, _intv1_cache)
    print(f"    Saved: {_intv1_cache}")

# %% ── 7b. Option 3: steer all ctrl cells (inject at last token) ──────────────
# No gene-presence requirement; always injects at the last token in the sentence.

_intv3_cache = os.path.join(_acts_dir, f"intv3_ctrl_last_{GENE_SELECT}_{_layer_tag}.pt")

if os.path.exists(_intv3_cache):
    print(f"\n[7b] Loading cached option-3 steered embeddings...")
    intv_acts_ctrl = torch.load(_intv3_cache, map_location="cpu")
else:
    print(f"\n[7b] Option 3 — steering all ctrl cells (last token) at scales {SCALE_LIST_OPT3} (layers {PROBE_LAYERS})...")
    intv_acts_ctrl: dict[float, torch.Tensor] = {}
    for scale in SCALE_LIST_OPT3:
        vecs = []
        for name in tqdm(ctrl_names, desc=f"ctrl scale={scale:+.1f}"):
            emb = _run_cell_emb(adata_ctrl[[name]].copy(), alpha=scale,
                                probe_layers=PROBE_LAYERS, probe_dirs=probe_dirs_list,
                                injection_pos="last")
            if emb is not None:
                vecs.append(emb)
        if vecs:
            intv_acts_ctrl[scale] = torch.stack(vecs, 0)
        print(f"    scale={scale:+.1f}  →  {intv_acts_ctrl.get(scale, torch.empty(0)).shape}")
    torch.save(intv_acts_ctrl, _intv3_cache)
    print(f"    Saved: {_intv3_cache}")

# %% ── 7c. Option 4: steer all ctrl cells (inject at ALL tokens) ──────────────
# Broadest intervention; mirrors what generation steering does (every step).

_intv4_cache = os.path.join(_acts_dir, f"intv4_ctrl_all_{GENE_SELECT}_{_layer_tag}.pt")

if os.path.exists(_intv4_cache):
    print(f"\n[7c] Loading cached option-4 steered embeddings...")
    intv_acts_all = torch.load(_intv4_cache, map_location="cpu")
else:
    print(f"\n[7c] Option 4 — steering all ctrl cells (all tokens) at scales {SCALE_LIST_OPT3} (layers {PROBE_LAYERS})...")
    intv_acts_all: dict[float, torch.Tensor] = {}
    for scale in SCALE_LIST_OPT3:
        vecs = []
        for name in tqdm(ctrl_names, desc=f"ctrl scale={scale:+.1f}"):
            emb = _run_cell_emb(adata_ctrl[[name]].copy(), alpha=scale,
                                probe_layers=PROBE_LAYERS, probe_dirs=probe_dirs_list,
                                injection_pos="all")
            if emb is not None:
                vecs.append(emb)
        if vecs:
            intv_acts_all[scale] = torch.stack(vecs, 0)
        print(f"    scale={scale:+.1f}  →  {intv_acts_all.get(scale, torch.empty(0)).shape}")
    torch.save(intv_acts_all, _intv4_cache)
    print(f"    Saved: {_intv4_cache}")

# %% ── 7d. Option 5: steer all ctrl cells (ALL tokens, amplify mode) ──────────
# Same spatial scope as option 4 but uses amplify: scales the existing projection
# of each hidden state onto the concept directions rather than adding unconditionally.

_intv5_cache = os.path.join(_acts_dir, f"intv5_ctrl_all_{GENE_SELECT}_{_layer_tag}_amplify.pt")

if os.path.exists(_intv5_cache):
    print(f"\n[7d] Loading cached option-5 steered embeddings...")
    intv_acts_amplify = torch.load(_intv5_cache, map_location="cpu")
else:
    print(f"\n[7d] Option 5 — ctrl cells (all tokens, amplify) at scales {SCALE_LIST_OPT3} (layers {PROBE_LAYERS})...")
    intv_acts_amplify: dict[float, torch.Tensor] = {}
    for scale in SCALE_LIST_OPT3:
        vecs = []
        for name in tqdm(ctrl_names, desc=f"ctrl scale={scale:+.1f}"):
            emb = _run_cell_emb(adata_ctrl[[name]].copy(), alpha=scale,
                                probe_layers=PROBE_LAYERS, probe_dirs=probe_dirs_list,
                                injection_pos="all", mode="amplify")
            if emb is not None:
                vecs.append(emb)
        if vecs:
            intv_acts_amplify[scale] = torch.stack(vecs, 0)
        print(f"    scale={scale:+.1f}  →  {intv_acts_amplify.get(scale, torch.empty(0)).shape}")
    torch.save(intv_acts_amplify, _intv5_cache)
    print(f"    Saved: {_intv5_cache}")

# %% ── 8. Score steered activations ──────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

print("\n[8] Scoring (SGD regression: 0=ctrl-like, 1=pert-like)...")
print("  Option 1 — steered pert cells (inject at GENE_SELECT position):")
scores1 = score_steering_regression(ctrl_act, pert_act, intv_acts_pert)
for scale in sorted(scores1):
    print(f"    scale={scale:+.1f}  score={scores1[scale]:.4f}")

print("  Option 3 — steered ctrl cells (inject at last token):")
scores3 = score_steering_regression(ctrl_act, pert_act, intv_acts_ctrl)
for scale in sorted(scores3):
    print(f"    scale={scale:+.1f}  score={scores3[scale]:.4f}")

print("  Option 4 — steered ctrl cells (all tokens, add):")
scores4 = score_steering_regression(ctrl_act, pert_act, intv_acts_all)
for scale in sorted(scores4):
    print(f"    scale={scale:+.1f}  score={scores4[scale]:.4f}")

print("  Option 5 — steered ctrl cells (all tokens, amplify):")
scores5 = score_steering_regression(ctrl_act, pert_act, intv_acts_amplify)
for scale in sorted(scores5):
    print(f"    scale={scale:+.1f}  score={scores5[scale]:.4f}")

# %% ── 9. UMAP ────────────────────────────────────────────────────────────────

print("\n[9] Plotting UMAPs...")

umap1_path = os.path.join(OUT_DIR, f"umap_opt1_pert_{GENE_SELECT}.png")
plot_steering_umap(
    ctrl_act=ctrl_act,
    pert_act=pert_act,
    intv_acts=intv_acts_pert,
    gene_select=_cond_pert,
    save_path=umap1_path,
)
print(f"    Option 1 saved: {umap1_path}")

umap3_path = os.path.join(OUT_DIR, f"umap_opt3_ctrl_{GENE_SELECT}.png")
plot_steering_umap(
    ctrl_act=ctrl_act,
    pert_act=pert_act,
    intv_acts=intv_acts_ctrl,
    gene_select=_cond_pert,
    save_path=umap3_path,
)
print(f"    Option 3 saved: {umap3_path}")

umap4_path = os.path.join(OUT_DIR, f"umap_opt4_ctrl_all_{GENE_SELECT}.png")
plot_steering_umap(
    ctrl_act=ctrl_act,
    pert_act=pert_act,
    intv_acts=intv_acts_all,
    gene_select=_cond_pert,
    save_path=umap4_path,
)
print(f"    Option 4 saved: {umap4_path}")

umap5_path = os.path.join(OUT_DIR, f"umap_opt5_ctrl_all_{GENE_SELECT}_amplify.png")
plot_steering_umap(
    ctrl_act=ctrl_act,
    pert_act=pert_act,
    intv_acts=intv_acts_amplify,
    gene_select=_cond_pert,
    save_path=umap5_path,
)
print(f"    Option 5 saved: {umap5_path}")

print("\nDone.")

# %%