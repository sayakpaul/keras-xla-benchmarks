MODEL_NAME_MAPPING = {
    "ViT": {
        "vit_s16": ("https://tfhub.dev/sayakpaul/vit_s16_classification/1", 4.251),
        "vit_b8": ("https://tfhub.dev/sayakpaul/vit_b8_classification/1", 66.865),
        "vit_b16": ("https://tfhub.dev/sayakpaul/vit_b16_classification/1", 16.867),
        "vit_b32": ("https://tfhub.dev/sayakpaul/vit_b32_classification/1", 4.368),
        "vit_l16": ("https://tfhub.dev/sayakpaul/vit_l16_classification/1", 59.697),
        "vit_r26_s32": (
            "https://tfhub.dev/sayakpaul/vit_r26_s32_lightaug_classification/1",
            "3.536",
        ),
        "vit_r50_l32": (
            "https://tfhub.dev/sayakpaul/vit_r50_l32_classification/1",
            "19.452",
        ),
    },
    "DeiT": {
        "deit_tiny_patch16_224": "https://tfhub.dev/sayakpaul/deit_tiny_patch16_224/1",
        "deit_tiny_distilled_patch16_224": "https://tfhub.dev/sayakpaul/deit_tiny_distilled_patch16_224/1",
        "deit_small_patch16_224": "https://tfhub.dev/sayakpaul/deit_small_patch16_224/1",
        "deit_small_distilled_patch16_224": "https://tfhub.dev/sayakpaul/deit_small_distilled_patch16_224/1",
        "deit_base_patch16_224": "https://tfhub.dev/sayakpaul/deit_base_patch16_224/1",
        "deit_base_distilled_patch16_224": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_224/1",
        "deit_base_patch16_384": "https://tfhub.dev/sayakpaul/deit_base_patch16_384/1",
        "deit_base_distilled_patch16_384": "https://tfhub.dev/sayakpaul/deit_base_distilled_patch16_384/1",
    },
    "Swin": {
        "swin_tiny_patch4_window7_224": "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1",
        "swin_small_patch4_window7_224": "https://tfhub.dev/sayakpaul/swin_small_patch4_window7_224/1",
        "swin_base_patch4_window7_224": "https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224/1",
        "swin_base_patch4_window12_384": "https://tfhub.dev/sayakpaul/swin_base_patch4_window12_384/1",
        "swin_large_patch4_window7_224": "https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224/1",
        "swin_large_patch4_window7_384": "https://tfhub.dev/sayakpaul/swin_large_patch4_window7_384/1",
        "swin_s3_tiny_224": "https://tfhub.dev/sayakpaul/swin_s3_tiny_224/1",
        "swin_s3_small_224": "https://tfhub.dev/sayakpaul/swin_s3_small_224/1",
        "swin_s3_base_224": "https://tfhub.dev/sayakpaul/swin_s3_base_224/1",
    },
    "MLP-Mixer": {
        "mixer_b16": (
            "https://tfhub.dev/sayakpaul/mixer_b16_sam_classification/1",
            12.621,
        ),
        "mixer_b32": (
            "https://tfhub.dev/sayakpaul/mixer_b32_sam_classification/1",
            3.242,
        ),
        "mixer_l16": (
            "https://tfhub.dev/sayakpaul/mixer_l16_i1k_classification/1",
            44.597,
        ),
    },
}
