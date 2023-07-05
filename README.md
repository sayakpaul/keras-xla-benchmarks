# keras-xla-benchmarks 🌪
Presents comprehensive benchmarks of XLA-compatible pre-trained vision models in Keras. We use pre-trained computer vision models shipped by `keras.applications`, `keras_cv.models`, and TensorFlow Hub. Benchmarks were conducted across different image resolutions and different GPU devices (A100, V100, and T4) to provide a holistic overview of the possible gains from XLA.

Jump straight to the interesting findings [here](https://github.com/sayakpaul/keras-xla-benchmarks#findings-from-the-benchmark-%EF%B8%8F%EF%B8%8F). 

## Useful links 🌐

* Learn more about XLA from [here](https://www.tensorflow.org/xla).
* You can explore the benchmark results here and interact with them:
[wandb.ai/sayakpaul/keras-xla-benchmarks](https://wandb.ai/sayakpaul/keras-xla-benchmarks). 
* The main CSV file collected from the benchmark is available here:
[https://hf.co/datasets/sayakpaul/sample-datasets/blob/main/keras_xla_benchmarks.csv](https://hf.co/datasets/sayakpaul/sample-datasets/blob/main/keras_xla_benchmarks.csv).
* A [presentation](https://docs.google.com/presentation/d/1HbzkdLnT36H3zFTlActwSj6LnH6yegAS0xywStuhqnI/edit?usp=sharing) on XLA.
* A [blog post](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html) on how Hugging Face used XLA to speed up the inference latency of its text generation models in 🤗 Transformers.

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

For the above command to work, you need to have CUDA and the latest version of Docker installed. You would also need to ensure that you're using a CUDA-compatible GPU.

Once you're in the Docker image, navigate to any of the model folders (`hub`, `keras_legacy`, or `keras_cv`) and follow the instructions there. 

If you want to log the results to Weights and Biases, install the Python library by running `pip install wandb`. Then while launching a benchmark pass the `--log_wandb`
flag.

The Docker container was built like so:

```bash
docker build -t spsayakpaul/keras-xla-benchmarks .
docker push spsayakpaul/keras-xla-benchmarks
```

## Findings from the benchmark 🕵️‍♂️

Each folder (`keras_legacy`, `keras_cv`, or `hub`) contains a Jupyter Notebook called `analysis.ipynb` that provides some exploratory analysis on the results. The `compare.ipynb` notebook presents some basic analysis as well. 

> 💡 **Note**: that for this project, we solely focus on benchmarking the throughput of the models and NOT on their predictive quality. 

Below are some findings I found interesting. 

### Across different GPUs, how fast are the models with XLA from `keras.applications`?

<div align="center">
<table>
  <tr>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/a100_224_True.png" width=500/></td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/v100_224_True.png" width=500/></td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/t4_224_True.png" width=500/></td>
  </tr>
</table>
<sub><b>Caption</b>: Throughput (samples/sec) of the top-10 models with XLA across different GPUs. Different GPUs seem to have different top performing models (throughput-wise). The volume of the dots in the plots was determined by the number of parameters each model contains.</sub> 
</div><br>

💡 One particularly interesting finding here is that models having more FLOPs or more number of parameters aren't always slower than the ones having less FLOPs or less number of parameters. Take the plot corresponding to A100, for example. We notice that VGG16, despite having more FLOPs and more number of parameters, is faster than say, ConvNeXt Tiny. This finding is in line with [The Efficiency Misnomer](https://arxiv.org/abs/2110.12894).

### Resolution-wise distribution of the throughputs obtained by different models in `keras.applications` with XLA

|    | **model_family** | **model_variant** | **resolution** | **accelerator** | **flop (giga)** | **params (million)** | **throughput (samples/sec)** |
|---:|:-----------------|:-----------------|---------------:|:----------------|----------------:|---------------------:|----------------------------:|
|  0 | MobileNet_V1     | mobilenet_v1     |           224 | v100            |            0.57 |                 4.25 |                     2842.09 |
|  1 | EfficientNet_V2  | efficient_b1_v2  |           240 | v100            |            1.21 |                 8.21 |                      866.32 |
|  2 | EfficientNet_V2  | efficient_b2_v2  |           260 | v100            |            1.71 |                10.18 |                      738.15 |
|  3 | Xception         | xception         |           299 | a100            |            8.36 |                22.91 |                      793.82 |
|  4 | EfficientNet_V1  | efficient_b3     |           300 | a100            |            1.86 |                12.32 |                      578.09 |
|  5 | NASNet           | nasnet_large     |           331 | a100            |           23.84 |                88.95 |                      149.77 |
|  6 | EfficientNet_V1  | efficient_b4     |           380 | a100            |            4.46 |                19.47 |                      463.45 |
|  7 | EfficientNet_V2  | efficient_s_v2   |           384 | a100            |            8.41 |                21.61 |                      474.41 |
|  8 | EfficientNet_V1  | efficient_b5     |           456 | a100            |           10.4  |                30.56 |                      268.44 |
|  9 | EfficientNet_V2  | efficient_m_v2   |           480 | a100            |           24.69 |                54.43 |                      238.62 |
| 10 | EfficientNet_V1  | efficient_b6     |           528 | a100            |           19.29 |                43.27 |                      162.92 |
| 11 | EfficientNet_V1  | efficient_b7     |           600 | a100            |           38.13 |                66.66 |                      107.52 |

💡 It seems like as we increase the resolution beyond 260, A100 tops the charts. But for resolutions lower than that, V100 tends to yield the highest amount of throughputs with XLA.

### What about the same but also grouped w.r.t the GPU being used? 

|    | **model_family**    | **model_variant**   |   **resolution** | **accelerator**   |   **flop (giga)** |   **params (million)** |   **throughput (samples/sec)** |
|---:|:-------------------|:-------------------|-----------------:|:------------------|-------------------:|-----------------------:|-------------------------------:|
|  0 | MobileNet_V1       | mobilenet_v1       |             224 | a100              |               0.57 |                   4.25 |                        2608.05 |
|  1 | RegNet_X           | regnetx_016        |             224 | t4                |               0.1  |                   2.71 |                        1921.77 |
|  2 | MobileNet_V1       | mobilenet_v1       |             224 | v100              |               0.57 |                   4.25 |                        2842.09 |
|  3 | EfficientNet_V1    | efficient_b1       |             240 | a100              |               0.7  |                   7.86 |                         710.85 |
|  4 | EfficientNet_V2    | efficient_b1_v2    |             240 | t4                |               1.21 |                   8.21 |                         477.9  |
|  5 | EfficientNet_V2    | efficient_b1_v2    |             240 | v100              |               1.21 |                   8.21 |                         866.32 |
|  6 | EfficientNet_V1    | efficient_b2       |             260 | a100              |               1.01 |                   9.18 |                         662.06 |
|  7 | EfficientNet_V2    | efficient_b2_v2    |             260 | t4                |               1.71 |                  10.18 |                         438.91 |
|  8 | EfficientNet_V2    | efficient_b2_v2    |             260 | v100              |               1.71 |                  10.18 |                         738.15 |
|  9 | Xception           | xception           |             299 | a100              |               8.36 |                  22.91 |                         793.82 |
| 10 | Inception          | inception_v3       |             299 | t4                |               5.73 |                  23.85 |                         224.77 |
| 11 | Xception           | xception           |             299 | v100              |               8.36 |                  22.91 |                         467.52 |
| 12 | EfficientNet_V1    | efficient_b3       |             300 | a100              |               1.86 |                  12.32 |                         578.09 |
| 13 | EfficientNet_V2    | efficient_b3_v2    |             300 | t4                |               3.03 |                  14.47 |                         283.02 |
| 14 | EfficientNet_V2    | efficient_b3_v2    |             300 | v100              |               3.03 |                  14.47 |                         515.21 |
| 15 | NASNet             | nasnet_large       |             331 | a100              |              23.84 |                  88.95 |                         149.77 |
| 16 | NASNet             | nasnet_large       |             331 | t4                |              23.84 |                  88.95 |                          42.37 |
| 17 | NASNet             | nasnet_large       |             331 | v100              |              23.84 |                  88.95 |                         104.47 |
| 18 | EfficientNet_V1    | efficient_b4       |             380 | a100              |               4.46 |                  19.47 |                         463.45 |
| 19 | EfficientNet_V1    | efficient_b4       |             380 | t4                |               4.46 |                  19.47 |                         131.74 |
| 20 | EfficientNet_V1    | efficient_b4       |             380 | v100              |               4.46 |                  19.47 |                         310.74 |
| 21 | EfficientNet_V2    | efficient_s_v2     |             384 | a100              |               8.41 |                  21.61 |                         474.41 |
| 22 | EfficientNet_V2    | efficient_s_v2     |             384 | t4                |               8.41 |                  21.61 |                         141.84 |
| 23 | EfficientNet_V2    | efficient_s_v2     |             384 | v100              |               8.41 |                  21.61 |                         323.35 |
| 24 | EfficientNet_V1    | efficient_b5       |             456 | a100              |              10.4  |                  30.56 |                         268.44 |
| 25 | EfficientNet_V1    | efficient_b5       |             456 | t4                |              10.4  |                  30.56 |                          47.08 |
| 26 | EfficientNet_V1    | efficient_b5       |             456 | v100              |              10.4  |                  30.56 |                         173.51 |
| 27 | EfficientNet_V2    | efficient_m_v2     |             480 | a100              |              24.69 |                  54.43 |                         238.62 |
| 28 | EfficientNet_V2    | efficient_m_v2     |             480 | t4                |              24.69 |                  54.43 |                          49.26 |
| 29 | EfficientNet_V2    | efficient_m_v2     |             480 | v100              |              24.69 |                  54.43 |                         133.36 |
| 30 | EfficientNet_V1    | efficient_b6       |             528 | a100              |              19.29 |                  43.27 |                         162.92 |
| 31 | EfficientNet_V1    | efficient_b6       |             528 | t4                |              19.29 |                  43.27 |                          36.88 |
| 32 | EfficientNet_V1    | efficient_b6       |             528 | v100              |              19.29 |                  43.27 |                         104.09 |
| 33 | EfficientNet_V1    | efficient_b7       |             600 | a100              |              38.13 |                  66.66 |                         107.52 |
| 34 | EfficientNet_V1    | efficient_b7       |             600 | t4                |              38.13 |                  66.66 |                          20.85 |
| 35 | EfficientNet_V1    | efficient_b7       |             600 | v100              |              38.13 |                  66.66 |                          63.23 |

💡 So, the fastest model changes for a fixed resolution when the GPU (being used for benchmarking) changes. This phenomena becomes less evident when the resolution increases. 

### Which model family (from `keras.applications`) has the highest amount of absolute speedup from XLA for a particular resolution (say 224) and accelerator (say A100)?

|    | model_family    | model_variant   |   speedup |
|---:|:----------------|:----------------|----------:|
|  0 | MobileNet_V1    | mobilenet_v1    |   2543.92 |
|  1 | RegNet_X        | regnetx_016     |   1933.78 |
|  2 | MobileNet_V2    | mobilenet_v2    |   1668.39 |
|  3 | RegNet_Y        | regnety_002     |   1216.29 |
|  4 | VGG             | vgg16           |   1209.08 |
|  5 | ConvNeXt        | convnext_tiny   |   1134.46 |
|  6 | EfficientNet_V1 | efficient_b0    |    893.08 |
|  7 | ResNetRS        | resnetrs_50     |    787.59 |
|  8 | EfficientNet_V2 | efficient_b0_v2 |    780    |
|  9 | DenseNet        | densenet_121    |    700.73 |
| 10 | ResNet_V1       | resnet50_v1     |    671.24 |
| 11 | ResNet_V2       | resnet101_v2    |    569.12 |
| 12 | NASNet          | nasnet_mobile   |    423.78 |

💡 Absolute speedup here means `throughput_with_xla` - `throughput_without_xla`. Interestingly, for each model family, the smallest model doesn't necessarily always lead to the highest amount of absolute speedup. For example, for RegNetX, RegNetX_16 isn't the smallest variant. Same holds for ResNet101_V2. 

### What about the relative speedup in percentages?

|    | model_family    | model_variant   |   speedup_percentage |
|---:|:----------------|:----------------|---------------------:|
|  0 | RegNet_X        | regnetx_016     |             4452.64  |
|  1 | NASNet          | nasnet_mobile   |             4368.87  |
|  2 | ConvNeXt        | convnext_small  |             4188.45  |
|  3 | MobileNet_V1    | mobilenet_v1    |             3966.82  |
|  4 | DenseNet        | densenet_121    |             3686.11  |
|  5 | RegNet_Y        | regnety_004     |             3427.97  |
|  6 | ResNetRS        | resnetrs_350    |             3300.45  |
|  7 | MobileNet_V2    | mobilenet_v2    |             2964.45  |
|  8 | ResNet_V2       | resnet101_v2    |             2844.18  |
|  9 | EfficientNet_V1 | efficient_b0    |             2841.49  |
| 10 | EfficientNet_V2 | efficient_b0_v2 |             2761.06  |
| 11 | ResNet_V1       | resnet152_v1    |             1639.69  |
| 12 | VGG             | vgg16           |              396.472 |

💡 Some whopping speedup (**4452.64%**) right there 🤯 Again, smallest variant from a model family doesn't always lead to the highest amount of relative speedup here. 

### How do these models fair to non-CNN models such as Swin, ViT, DeiT, and MLP-Mixer?

|    | model_family    | model_variant                    |   speedup_percentage |
|---:|:----------------|:---------------------------------|---------------------:|
|  0 | RegNet_X        | regnetx_016                      |           4452.64    |
|  1 | NASNet          | nasnet_mobile                    |           4368.87    |
|  2 | ConvNeXt        | convnext_small                   |           4188.45    |
|  3 | MobileNet_V1    | mobilenet_v1                     |           3966.82    |
|  4 | DenseNet        | densenet_121                     |           3686.11    |
|  5 | RegNet_Y        | regnety_004                      |           3427.97    |
|  6 | ResNetRS        | resnetrs_350                     |           3300.45    |
|  7 | MobileNet_V2    | mobilenet_v2                     |           2964.45    |
|  8 | ResNet_V2       | resnet101_v2                     |           2844.18    |
|  9 | EfficientNet_V1 | efficient_b0                     |           2841.49    |
| 10 | EfficientNet_V2 | efficient_b0_v2                  |           2761.06    |
| 11 | ResNet_V1       | resnet152_v1                     |           1639.69    |
| 12 | Swin            | swin_s3_small_224                |           1382.65    |
| 13 | DeiT            | deit_small_distilled_patch16_224 |            525.086   |
| 14 | VGG             | vgg16                            |            396.472   |
| 15 | MLP-Mixer       | mixer_b32                        |             75.1291  |
| 16 | ViT             | vit_b16                          |              5.69305 |

💡 Seems like the non-CNN models don't benefit as much in comparison to the CNN ones from XLA.

### Within non-CNN models, what's the trend?

<div align="center">
<table>
  <tr>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/non_cnn/a100_224_True.png" width=500/></td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/non_cnn/v100_224_True.png" width=500/></td>
    <td><img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/non_cnn/t4_224_True.png" width=500/></td>
  </tr>
</table>
<sub><b>Caption</b>: Throughput (samples/sec) of the top-10 non-CNN models with XLA across different GPUs. Different GPUs seem to have different top performing models (throughput-wise). The volume of the dots in the plots was determined by the number of parameters each model contains.</sub> 
</div><br>

💡 Here also the similar finding holds as the one presented after Table 1. Mixer-B32, despite being much larger than many models, is faster.

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

