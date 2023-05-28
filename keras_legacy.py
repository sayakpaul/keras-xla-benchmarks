from tensorflow import keras

MODEL_NAME_MAPPING = {
    "ResNet_V1": {
        "resnet50_v1": keras.applications.ResNet50(),
        "resnet101_v1": keras.applications.ResNet101(),
        "resnet152_v1": keras.applications.ResNet152(),
    },
    "ResNet_V2": {
        "resnet50_v2": keras.applications.ResNet50V2(),
        "resnet101_v2": keras.applications.ResNet101V2(),
        "resnet152_v2": keras.applications.ResNet152V2(),
    },
    "Inception": {
        "inception_v3": keras.applications.InceptionV3(),
        "inception_resnetv2": keras.applications.InceptionResNetV2(),
    },
    "ResNetRS": {
        "resnetrs_50": keras.applications.ResNetRS50(),
        "resnetrs_101": keras.applications.ResNetRS101(),
        "resnetrs_152": keras.applications.ResNetRS152(),
        "resnetrs_200": keras.applications.ResNetRS200(),
        "resnetrs_270": keras.applications.ResNetRS270(),
        "resnetrs_350": keras.applications.ResNetRS350(),
        "resnetrs_420": keras.applications.ResNetRS420(),
    },
    "ConvNeXt": {
        "convnext_tiny": keras.applications.ConvNeXtTiny(),
        "convnext_small": keras.applications.ConvNeXtSmall(),
        "convnext_base": keras.applications.ConvNeXtBase(),
        "convnext_large": keras.applications.ConvNeXtLarge(),
        "convnext_xlarge": keras.applications.ConvNeXtXLarge(),
    },
    "EfficientNet_V1": {
        "efficient_b0": keras.applications.EfficientNetB0(),
        "efficient_b1": keras.applications.EfficientNetB1(),
        "efficient_b2": keras.applications.EfficientNetB2(),
        "efficient_b3": keras.applications.EfficientNetB3(),
        "efficient_b4": keras.applications.EfficientNetB4(),
        "efficient_b5": keras.applications.EfficientNetB5(),
        "efficient_b6": keras.applications.EfficientNetB6(),
        "efficient_b7": keras.applications.EfficientNetB7(),
    },
    "EfficientNet_V2": {
        "efficient_b0_v2": keras.applications.EfficientNetV2B0(),
        "efficient_b1_v2": keras.applications.EfficientNetV2B1(),
        "efficient_b2_v2": keras.applications.EfficientNetV2B2(),
        "efficient_b3_v2": keras.applications.EfficientNetV2B3(),
        "efficient_l_v2": keras.applications.EfficientNetV2L(),
        "efficient_m_v2": keras.applications.EfficientNetV2M(),
        "efficient_s_v2": keras.applications.EfficientNetV2S(),
    },
    "Xception": {"xception": keras.applications.Xception()},
    "MobileNet_V1": {"mobilenet_v1": keras.applications.MobileNet()},
    "MobileNet_V2": {"mobilenet_v2": keras.applications.MobileNetV2()},
    "MobileNet_V3": {
        "mobilenet_v3_small": keras.applications.MobileNetV3Small(),
        "mobilenet_v3_large": keras.applications.MobileNetV3Large(),
    },
}
