from interp_pipeline.get_annotation.gprofiler_client import GProfilerClient, GProfilerSpec

gp = GProfilerClient(cache_dir="debug_acts/go_cache")

spec = GProfilerSpec(
    organism="hsapiens",
    sources=["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG"],
    user_threshold=0.05,
    significance_threshold_method="fdr",
)

genes = ["NR1H4","TRIP12","UBC","FCRL3","PLXNA3","GDNF","VPS11"]
res = gp.profile(genes, spec=spec, query_name="demo")

print(res.head() if hasattr(res, "head") else res[:2])