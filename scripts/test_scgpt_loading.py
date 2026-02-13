from interp_pipeline.adapters.models.scgpt import ScGPTAdapter
from interp_pipeline.adapters.model_base import ModelSpec

spec = ModelSpec(
    name="scgpt",
    checkpoint="/maiziezhou_lab/zihang/SpatialFoundationModel/eval/scGPT/whole-human-pretrain",
    device="cuda",
    options={}
)

adapter = ScGPTAdapter()
handle = adapter.load(spec)

layers = adapter.list_layers(handle)
print("loaded scGPT")
print("#layers:", len(layers))
print("first 5:", layers[:5])