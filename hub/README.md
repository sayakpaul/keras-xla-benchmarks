FLOPs count for the ViT and MLP-Mixer models were derived with `timm` and `fvcore` using the following code:

```python
import timm
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

random_input = torch.randn(1, 3, 224, 224)


def print_flops(model_names: list):
    for name in model_names:
        print("*" * 80)
        print(name)
        if name != "mixer_b32_224":
            model = timm.create_model(name, pretrained=True)
        else:
            model = timm.create_model(name)
        flop = FlopCountAnalysis(model, random_input)
        print(flop_count_table(flop, max_depth=1))
        print("*" * 80)
        print("\n")


vit_models_timm = [
    "vit_small_patch16_224",
    "vit_base_patch8_224",
    "vit_base_patch16_224",
    "vit_base_patch32_224",
    "vit_large_patch16_224",
    "vit_small_r26_s32_224",
    "vit_large_r50_s32_224",
]
print_flops(vit_models_timm)

mixer_models_timm = ["mixer_b16_224", "mixer_b32_224", "mixer_l16_224"]
print_flops(mixer_models_timm)
```


