# keras-xla-benchmarks
Presents comprehensive benchmarks of XLA-compatible pre-trained models in Keras.

## Dev environment

Benchmark results can vary a lot from platform. So, it's important ensure a consistent development platform. For the dev environment, we use the following Docker container:  `spsayakpaul/keras-xla-benchmarks`, built on top of  `tensorflow/tensorflow:latest-gpu` ([reference](https://www.tensorflow.org/install/docker)). 

To run the Docker container:

```bash
nvidia-docker run -it --rm --shm-size=16g --ulimit memlock=-1 spsayakpaul/keras-xla-benchmarks
```

The Docker container was built like so:

```bash
docker build -t spsayakpaul/keras-xla-benchmarks .
docker push spsayakpaul/keras-xla-benchmarks
```

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

