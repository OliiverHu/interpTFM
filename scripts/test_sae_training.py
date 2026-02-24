from interp_pipeline.io.activation_store import ActivationStore, ActivationStoreSpec
from interp_pipeline.sae.sae_base import SAESpec
from interp_pipeline.sae.trainers import fit_sae_for_layer

OUT_DIR = "debug_acts"
LAYER = "layer_0.norm2"

store = ActivationStore(ActivationStoreSpec(root=OUT_DIR))

spec = SAESpec(
    n_latents=4096,   # small for test
    l1=1e-3,
    lr=1e-4,
    steps=200,       # short run
    seed=0,
)

res = fit_sae_for_layer(
    store=store,
    layer=LAYER,
    spec=spec,
    output_dir=OUT_DIR,
    device="cuda",
    batch_size=1024,
)

print("SAE saved:", res.model_path)
print("Summary:", res.summary)
