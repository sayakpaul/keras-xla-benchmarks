# keras-xla-benchmarks 🌪
Presents comprehensive benchmarks of XLA-compatible pre-trained models in Keras. We use 
pre-trained computer vision models shipped by `keras.applications`, `keras_cv.models`, 
and TensorFlow Hub. Benchmarks were conducted across different image resolutions and
different GPU devices to provide a holistic overview of the possible gains from XLA.

Learn more about XLA from [here](https://www.tensorflow.org/xla).

## Model pool 🏊‍♂️

Following model families were benchmarked:

* From `keras.applications`
    * [ResNet_V1](https://arxiv.org/abs/1512.03385) 
    * [ResNet_V2](https://arxiv.org/abs/1603.05027) 
    * Inception ([one](http://arxiv.org/abs/1512.00567), [two](https://arxiv.org/abs/1602.07261))
    * [ResNetRS](https://arxiv.org/abs/2103.07579) 
    * [ConvNeXt](https://arxiv.org/abs/2201.03545) 
    * [EfficientNet_V1](https://arxiv.org/abs/1905.11946) 
    * [EfficientNet_V2](https://arxiv.org/abs/2104.00298) 
    * [Xception](https://arxiv.org/abs/1610.02357) 
    * [MobileNet_V1](https://arxiv.org/abs/1704.04861) 
    * [MobileNet_V2](https://arxiv.org/abs/1801.04381) 
    * [MobileNet_V3](https://arxiv.org/abs/1905.02244) 
    * [VGG](https://arxiv.org/abs/1409.1556) 
    * [RegNet_X](https://arxiv.org/abs/2003.13678) 
    * [RegNet_Y](https://arxiv.org/abs/2003.13678) 
    * [DenseNet](https://arxiv.org/abs/1608.06993) 
    * [NASNet](https://arxiv.org/abs/1707.07012)
* From `keras_cv.models`
    * [YOLOV8](https://arxiv.org/abs/2305.09972)
    * [RetinaNet](https://arxiv.org/abs/1708.02002)
* From TensorFlow Hub
    * [ViT](https://arxiv.org/abs/2010.11929)
    * [DeiT](https://arxiv.org/abs/2012.12877)
    * [Swin](https://arxiv.org/abs/2103.14030)
    * [MLP-Mixer](https://arxiv.org/abs/2105.01601)

## Dev environment 👨‍💻

Benchmark results can vary a lot from platform. So, it's important ensure a consistent development platform. For the dev environment, we use the following Docker container:  `spsayakpaul/keras-xla-benchmarks`, built on top of  `tensorflow/tensorflow:latest-gpu` ([reference](https://www.tensorflow.org/install/docker)). 

To run the Docker container:

```bash
nvidia-docker run -it --rm --shm-size=16g --ulimit memlock=-1 spsayakpaul/keras-xla-benchmarks
```

Once you're in the Docker image, navigate to any of the model folders (`hub`, `keras_legacy`, or `keras_cv`) and follow the instructions there. 

If you want to log the results to Weights and Biases, install the Python library by running `pip install wandb`. Then while launching a benchmark pass the `--log_wandb`
flag.

The Docker container was built like so:

```bash
docker build -t spsayakpaul/keras-xla-benchmarks .
docker push spsayakpaul/keras-xla-benchmarks
```

## Findings from the benchmark 🕵️‍♂️

Each folder (`keras_legacy`, `keras_cv`, or `hub`) contains a Jupyter Notebook called `analysis.ipynb` that provides some exploratory analysis on the results.

## Keep in mind 💡

When you compile a model into XLA, always ensure the outputs of the compiled
model match with the non-compiled model. Here is an example:

```py
import tensorflow as tf 
import numpy as np

model = tf.keras.applications.MobileNetV3Large()
random_inputs = tf.random.normal((4, 224, 224, 3))

model_call_fn = tf.function(model, jit_compile=True)

non_xla_outs = model.predict(random_inputs)
xla_outs = model_call_fn(random_inputs, training=False)

np.testing.assert_allclose(
    non_xla_outs,
    xla_outs.numpy(),
    atol=1e-5,
    rtol=1e-5
)
```

