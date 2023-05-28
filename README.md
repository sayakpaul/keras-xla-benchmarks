# keras-xla-benchmarks
Presents comprehensive benchmarks of XLA-compatible pre-trained models in Keras.

## Dev environment

Benchmark results can vary a lot from platform. So, it's important ensure a consistent development platform. For running the benchmarks from this repository, we use an A100 (40 GB) as the GPU. For the dev environment, we use the following Docker container: `nvcr.io/nvidia/tensorflow:23.04-tf2-py3`. 

To run the Docker container:

```bash
nvidia-docker run -it --rm --shm-size=16g --ulimit memlock=-1 nvcr.io/nvidia/tensorflow:23.04-tf2-py3
```

We use a container from the [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow), because the containers provided by NGC are  optimized for the NVIDIA hardware platform. 
