## Running the benchmarks

We leverage `keras.applications` to benchmark the following models: 

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

You can launch benchmarks in bulk by running `python run_all_benchmarks.py`. To run
a benchmark individually, run `python run_benchmark.py`. If you do `python run_benchmark.py -h`, you will be able to see the CLI arguments supported by the script.