import os
import time
import numpy as np
import scanpy as sc
from tqdm import tqdm

from interp_pipeline.get_annotation.panel import panel_from_cosmx_adata
from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec
from interp_pipeline.get_annotation.enrichment_terms import gprofiler_go_term_qvals

from interp_pipeline.get_annotation.gconvert_client import GConvertClient, GConvertSpec
from interp_pipeline.get_annotation.quickgo_client import QuickGOClient, QuickGOSpec
from interp_pipeline.get_annotation.membership_gt import MembershipGTSpec, build_go_membership_gt


def _now() -> float:
    return time.time()


def _fmt_dt(t0: float) -> str:
    return f"{(time.time() - t0):.1f}s"


ADATA_PATH = "/maiziezhou_lab2/yunfei/Projects/FM_temp/InterPLM/interplm/ge_shards/cosmx_human_lung_sec8.h5ad"
OUT_DIR = "debug_acts/go_gt"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[1/5] loading AnnData: {ADATA_PATH}")
t0 = _now()
adata = sc.read_h5ad(ADATA_PATH)
panel = panel_from_cosmx_adata(adata, symbol_col="index")
genes_ens = panel.ensembl_ids
genes_sym = panel.symbols
print(f"  n_genes={len(genes_ens)} loaded in {_fmt_dt(t0)}")

print("[2/5] Ensembl -> UniProt via g:Profiler g:Convert (batched, cached)")
t0 = _now()
gp = GProfilerClient(cache_dir=os.path.join(OUT_DIR, "gprof_cache"))

gconv = GConvertClient(gp=gp, cache_dir=os.path.join(OUT_DIR, "gconvert_cache"))

# Start with SwissProt; if coverage is low, switch to UNIPROT
gconv_spec = GConvertSpec(organism="hsapiens", target="UNIPROTSWISSPROT")
ens_to_up = gconv.ensg_to_uniprot(genes_ens, spec=gconv_spec, force=False)

uniprots = sorted({u for accs in ens_to_up.values() for u in accs})
coverage = sum(1 for g in genes_ens if len(ens_to_up.get(g, [])) > 0) / max(1, len(genes_ens))
print(f"  mapped UniProt accessions={len(uniprots)} coverage={coverage:.3f} in {_fmt_dt(t0)}")
if coverage < 0.7:
    print("  WARNING: Swiss-Prot coverage looks low. Consider target='UNIPROT' for more coverage.")

print("first 10 uniprots:", uniprots[:10])
print("[3/5] fetching QuickGO annotations (cached)")
t0 = _now()
qg = QuickGOClient(cache_dir=os.path.join(OUT_DIR, "quickgo_cache"))
qg_spec = QuickGOSpec(taxon_id="9606")

print(f"  querying QuickGO for {len(uniprots)} UniProt accessions ...")
anns = qg.fetch_annotations_for_uniprot(uniprots, qg_spec, force=False)
print(f"  fetched annotations={len(anns)} in {_fmt_dt(t0)}")

print("  building GO membership GT (gene x GO term)")
t0 = _now()
gt_member = build_go_membership_gt(
    genes_ens=genes_ens,
    ensg_to_uniprot=ens_to_up,
    quickgo_annotations=anns,
    spec=MembershipGTSpec(high_conf_only=False, keep_aspects=None),
)
member_path = os.path.join(OUT_DIR, "GT_member_GO.csv")
gt_member.to_csv(member_path)
print(f"  GT_member saved: {member_path}")
print(f"  shape={gt_member.shape} density={float(gt_member.values.mean()):.6f} built in {_fmt_dt(t0)}")

print("[4/5] running g:Profiler enrichment on CosMx panel (GO + Reactome + KEGG) (cached)")
t0 = _now()
gp_spec = GProfilerSpec(
    organism="hsapiens",
    sources=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"],  # include Reactome
    user_threshold=0.05,
    significance_threshold_method="fdr",
    return_dataframe=True,
)

enr = gp.profile(genes_sym, spec=gp_spec, query_name="cosmx_panel")
enr_path = os.path.join(OUT_DIR, "enrichment_cosmx_panel_GO_REAC.csv")
if hasattr(enr, "to_csv"):
    enr.to_csv(enr_path, index=False)
else:
    import pandas as pd
    pd.DataFrame(enr).to_csv(enr_path, index=False)
print(f"  saved enrichment table: {enr_path} in {_fmt_dt(t0)}")

# Masking: only GO terms apply to GT_member columns (GO:...)
term_q = gprofiler_go_term_qvals(gp, genes_sym, gp_spec, query_name="cosmx_panel")
alpha = 0.05
keep_terms = [t for t in gt_member.columns if (t in term_q and term_q[t] < alpha)]
gt_masked = gt_member[keep_terms].copy()

masked_path = os.path.join(OUT_DIR, "GT_member_GO_masked_by_enrichment.csv")
gt_masked.to_csv(masked_path)
print(f"  masked GT saved: {masked_path}")
print(f"  masked shape={gt_masked.shape}")

print("[5/5] summary stats")
print("  GO terms in membership:", gt_member.shape[1])
print("  GO terms enriched+present:", gt_masked.shape[1])
print("  fraction kept:", gt_masked.shape[1] / max(1, gt_member.shape[1]))

per_gene_terms_before = gt_member.sum(axis=1).values
per_gene_terms_after = gt_masked.sum(axis=1).values
print("  avg GO terms/gene before:", float(np.mean(per_gene_terms_before)))
print("  avg GO terms/gene after :", float(np.mean(per_gene_terms_after)))

print("\nDONE.")