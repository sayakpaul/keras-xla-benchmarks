import keras_cv

MODEL_NAME_MAPPING = {
    "YOLOV8": {"yolo_v8_m_pascalvoc": keras_cv.models.YOLOV8Detector},
    "RetinaNet": {
        "retinanet_resnet50_pascalvoc": keras_cv.models.RetinaNet,
    },
}
