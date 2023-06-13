# MLPerf™ Inference Benchmarks for Image Classification

This is the reference implementation for MLPerf Inference Classification benchmark

### Setting up and running the benchmark

(Tested on Ubuntu 22.04(LTS) with python 3.10)

1. Clone the repository(must include submodules while cloning, to avoid errors in future steps)

    ```bash
    git clone --recurse-submodules https://github.com/bipinKrishnan/inference.git
    ```

2. Change directory to the repo and switch to `ivy_resnet_clf` branch:

    ```bash
    cd inference && git checkout ivy_resnet_clf
    ```

3. Move to the classification directory(that is, the directory where this readme resides)

    ```bash
    cd ./vision/classification_and_detection
    ```

4. Run the setup script, the script will automatically download imagenet validation set, create fake imagenet dataset(for experimentation), download the models, and also install all the required modules including ivy.

    ```bash
    chmod +x setup_clf_benchmark.sh
    ./setup_clf_benchmark.sh
    ```

5. The above script will create separate directories for original imagenet and fake imagenet. You can set the data directory for the benchmark as per you needs(i.e, either imagenet or fake imagenet):

    Run this command if you wish to use fake imagenet dataset for the benchmarking:
    ```bash
    export DATA_DIR=fake_imagenet
    ```
   Or use this command if you wish to use the imagenet dataset:

    ```bash
    export DATA_DIR=ILSVRC2012_img_val
    ```

6. Set model directory variable(all the downloaded models are stored in `models` folder):

    ```bash
    export MODEL_DIR=models
    ```

7. Get the api key for ivy compiler and place it in `.ivy/key.pem` file in `inference/vision/classification_and_detection` directory.

8. Run the benchmark with pytorch backend using the below command:

    ```bash
    ./run_local.sh pytorch-native resnet50 cpu --dataset imagenet_pytorch_native --profile resnet50-pytorch-native --data-format NCHW
    ```

If you wish to run the benchmark with Ivy compiled pytorch model, add `--compile_with_ivy` flag to above command.

Similary the tensorflow2 benchmark could be run using(add `--compile_with_ivy` flag if required):

    ./run_local.sh tf2 resnet50 cpu --dataset imagenet_tf2

Note: If you face any issue with installing loadgen package, run the following commands and try re-installing the package again(for python 3.10):

    sudo apt install python3.10-dev -y
    python3.10-config --includes

Run the following lines, if you face any issue with cmake:

    sudo apt-get install software-properties-common
    sudo add-apt-repository ppa:george-edison55/cmake-3.x
    sudo apt-get update
    sudo apt-get install cmake
    sudo apt-get upgrade

### Usage
```
usage: main.py [-h]
    [--mlperf_conf ../../mlperf.conf]
    [--user_conf user.conf]
    [--dataset {imagenet,openimages-300-retinanet,openimages-800-retinanet,openimages-1200-retinanet,openimages-800-retinanet-onnx,imagenet_mobilenet,coco,coco-300,coco-1200,coco-1200-onnx,coco-1200-pt,coco-1200-tf}]
    --dataset-path DATASET_PATH [--dataset-list DATASET_LIST]
    [--data-format {NCHW,NHWC}]
    [--profile {defaults,resnet50-tf,resnet50-onnxruntime,retinanet-pytorch,retinanet-onnxruntime,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-pytorch,ssd-resnet34-onnxruntime,resnet50-tvm-onnx,resnet50-tvm-pytorch}]
    [--scenario list of SingleStream,MultiStream,Server,Offline]
    [--max-batchsize MAX_BATCHSIZE]
    --model MODEL [--output OUTPUT] [--inputs INPUTS]
    [--outputs OUTPUTS] [--backend BACKEND] [--threads THREADS]
    [--time TIME] [--count COUNT] [--qps QPS]
    [--max-latency MAX_LATENCY] [--cache CACHE] [--accuracy]
```

```--mlperf_conf```
the mlperf config file to use for rules compliant parameters, defaults to ../../mlperf.conf

```--user_conf```
the user config file to use for user LoadGen settings such as target QPS, defaults to user.conf

```--dataset```
use the specified dataset. Currently we only support ImageNet.

```--dataset-path```
path to the dataset.

```--data-format {NCHW,NHWC}```
data-format of the model (default: the backends prefered format).

```--scenario {SingleStream,MultiStream,Server,Offline}```
comma separated list of benchmark modes.

```--profile {resnet50-tf,resnet50-onnxruntime,retinanet-onnxruntime,retinanet-pytorch,mobilenet-tf,mobilenet-onnxruntime,ssd-mobilenet-tf,ssd-mobilenet-onnxruntime,ssd-resnet34-tf,ssd-resnet34-onnxruntime,resnet50-tvm-onnx,resnet50-tvm-pytorch}```
this fills in default command line options with the once specified in the profile. Command line options that follow may override the those.

```--model MODEL```
the model file.

```--inputs INPUTS```
comma separated input name list in case the model format does not provide the input names. This is needed for tensorflow since the graph does not specify the inputs.

```--outputs OUTPUTS```
comma separated output name list in case the model format does not provide the output names. This is needed for tensorflow since the graph does not specify the outputs.

```--output OUTPUT]```
location of the JSON output.

```--backend BACKEND```
which backend to use. Currently supported is tensorflow, onnxruntime, pytorch and tflite.

```--threads THREADS```
number of worker threads to use (default: the number of processors in the system).

```--count COUNT```
Number of images the dataset we use (default: use all images in the dataset).

```--qps QPS```
Expected QPS.

```--max-latency MAX_LATENCY```
comma separated list of which latencies (in seconds) we try to reach in the 99 percentile (deault: 0.01,0.05,0.100).

```--max-batchsize MAX_BATCHSIZE```
maximum batchsize we generate to backend (default: 128).

```--compile_with_ivy```
whether to compile the model using ivy before starting the benchmark.


## License

[Apache License 2.0](LICENSE)
