## Running the benchmarks

We leverage KerasCV to benchmark the following models:

* [YOLOV8](https://arxiv.org/abs/2305.09972)
* [RetinaNet](https://arxiv.org/abs/1708.02002)

You can launch benchmarks in bulk by running `python run_all_benchmarks.py`. To run
a benchmark individually, run `python run_benchmark.py`. If you do `python run_benchmark.py -h`, you will be able to see the CLI arguments supported by the script.