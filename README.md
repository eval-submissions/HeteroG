# HeteroG
## Abstract 
HeteroG is a software module that automatically produces deployment strategies for different DNNs in heterogeneous environments. To successfully launch HeteroG, each server used in experiments should be equipped with at least one GPU. 
The GNN in HeteroG is implemented using Tensorflow1.14 which is modified to support customized execution orders. 
The detailed software and hardware requirements are introduced in following sections. We will also introduce the dependencies needed and the procedures to install HeteroG, and the detailed steps to conduct corresponding experiments.

## Hardware dependency
For offline training, we use one server equipped with two NVIDIA 16GB Tesla V100 GPUs, two 10-core Intel Xeon processor E5-2630v4 CPUs.

For distributed experiments, we deploy HeteroG-boosted Tensor-Flow framework in 4 physical machines and a 100Gbps switch:
- One server equipped with two NVIDIA 16GB Tesla V100GPUs, two 10-core Intel Xeon processor E5-2630 v4 CPUsand one 100GbE Mellanox RDMA card.
- Two servers equipped with two 11GB NVIDIA GTX 1080 TiGPUs, one 8-core Intel Xeon E5-1660 v4 CPU and one 50GbEMellanox RDMA card.
- One server equipped with two 12GB NVIDIA Tesla P100GPUs, one 8-core Intel Xeon E5-1660 v4 CPU and one 50GbEMellanox RDMA card.
- One Dell Networking Z9100-ON switch, 32x QSFP28 100GbE.

## Software dependency
Dependency | Version 
--- | --- 
OS  | Ubuntu-16.04   
Linux Kernel | Linux 4.4.0-131-generic x86_64 
GCC | gcc 5.4.0
CUDA-Toolkit |  cuda-10.0
CUDNN | cudnn-7.6.0
NCCL |  nccl-2.6.4 
Python |  python3.7
TensorFlow |  Modified version based on 1.14

The software dependency is listed in the table above. 
CUDA-Toolkit can be downloaded for https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604. 
CUDNN can be downloaded from https://developer.nvidia.com/cudnn-download-survey. NCCL can be downloaded from https://developer.nvidia.com/nccl/nccl2-download-survey. 
The modified Tensorflow should be downloaded from our this repository.

## Dataset
We conduct experiments based on both synthetic data and real training data sets including cifar10 and SQuAD. These data sets are already included in the repository.

## Models
We conduct experiments using 8 models including VGG-19,ResNet200, InceptionV3, MobileNetV2, NasNet, Transformer,Bert-large and XLNet-large. The implementation of all these models is already included in the github repository.

## Installation
We will detailed introduced the installation steps of heteroG in this part.
### Install python environment. 
We recommand to use anaconda, please download the installation script to install anaconda through the link: https://repo.anaconda.com/archive/Anaconda2-2019.10-Linux-x86_64.sh
After the installation, create an environment named heterog with python3.7

`conda create -n heterog python=3.7`

Then activate the environment:

`conda activate heterog`

### Install CUDA CUDNN and NCCL. 
CUDA-Toolkit can be downloaded for https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604. 

CUDNN can be downloaded from https://developer.nvidia.com/cudnn-download-survey. 

NCCL can be downloaded from https://developer.nvidia.com/nccl/nccl2-download-survey. 

### build modified tensorflow from source
First, clone the HeteroG project and the submodule of it from repo:

`git clone https://github.com/eval-submissions/HeteroG.git --recursive`

Then step into tensorflow folder:

`cd tensorflow`

Please check that the branch of tensorflow is r1.14, if not, please run

`git checkout r1.14`

Then install bazel to build tensorflow:

`conda install bazel=0.24`

Then configure the build:

`./configure`

The following shows a sample run of ./configure script (your session may differ):
```
  You have bazel 0.24 installed.

  Please specify the location of python. [Default is /usr/bin/python3]: 


Found possible Python library paths:

/usr/lib/python3/dist-packages

/usr/local/lib/python3.7/dist-packages

Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 

No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: 

No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: Y

CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: 

No TensorRT support will be enabled for TensorFlow.

Found CUDA 10.1 in:

/usr/local/cuda-10.1/targets/x86_64-linux/lib

/usr/local/cuda-10.1/targets/x86_64-linux/include

Found cuDNN 7 in:

/usr/lib/x86_64-linux-gnu

/usr/include



Please specify a list of comma-separated CUDA compute capabilities you want to build with.


You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus Each capability can be specified as "x.y" or "compute_xy" to include both virtual 
and binary GPU code, or as "sm_xy" to only include the binary code.

Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 6.1


Do you want to use clang as CUDA compiler? [y/N]: 

nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 

Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.

  --config=mkl            # Build with MKL support.

  --config=monolithic     # Config for mostly static monolithic build.

  --config=ngraph         # Build with Intel nGraph support.

  --config=numa           # Build with NUMA support.

  --config=dynamic_kernels    # (Experimental) Build kernels into separate shared objects.

  --config=v2             # Build TensorFlow 2.x instead of 1.x.

Preconfigured Bazel build configs to DISABLE default on features:

  --config=noaws          # Disable AWS S3 filesystem support.

  --config=nogcp          # Disable GCP support.

  --config=nohdfs         # Disable HDFS support.

  --config=nonccl         # Disable NVIDIA NCCL support.

Configuration finished
```
After Configuration, run the build script in the folder. Before running it, please modify the file accordingly to specify a path to store the whl file. After that you can run:

`sh build.sh`

After the execution of the script. The modified tensorflow should be successfully installed.

## Offline training of HeteroG
After sucessfully building the customized tensorflow. We can start training HeteroG. First, please change the folder to GAT:

`cd ../GAT`

In this folder, the file to train HeteroG is named as *main.py*. Before executing the file, we need to write a config file named *config.txt* in the folder first. To be convenient, we already provided the file with a sample configuration:
 ```
 {"inputs": ["data/graph1"], "devices":[
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:0/device:GPU:2",
    "/job:worker/replica:0/task:0/device:GPU:3"

],
"sinks":[["GradientDescent"]],
"CUDA_VISIBLE_DEVICES": "0,1,2,3", "max_replica_num": 4, "learning_rate": 5e-5, 
"bandwidth": ["10000", "747"], "device_mems": [13000000000.0, 13000000000.0, 10000000000.0, 10000000000.0]}
```
The file is in json format. "data/graph1" is the path of model for HeteroG to find best strategy. We provided a sample model(VGG19) in the folder "data/graph1", as well as its profiling data obtained from 4 GPUs. In this sample configuration, HeteroG will find best stratgey to deploy the VGG model in a 4-GPU environment.

To launch the training, we only need to run:
`python main.py`

Then we can see the training process from the screen.

In order to conveniently observe the training process, the program produces two files during the training process: *time.log* and *best_time.log* under the folder "HeteroG/GAT/". *time.log* records the simulated per-iteration time of found strategies every 10 steps. *best_time.log* records the simulated per-iteration time of the best strategies that HeteroG ever found.
## Experiment workflow

The experiment workflow is as follows: (1) profile 8 models toobtain the computation cost model; 
(2) transfer data throughthe link of each pair of devices to estimate the link band-width. 
(3) generate input features and train the GNN usingreinforcement learning. 
(4) record the deployment strategyafter convergence and analyze the strategy.

## Evaluation and expected result
**Per-iteration training speedup**. After convergence, we use script activater.py to deploy different strategies specified in configuration file activate_config.txt. 
It launches the distributed training process over the heterogeneous environment and records the per-iteration time of each deployment strategy. 
We activate the strategy found by HeteroG and the 4 baseline data parallelism strategies in this experiment and calculate the speedup shown in Table. 1 in the paper.

**Deployment of large models**. In this experiment, we test ResNet200, Bert-large and Xlnet-large with larger batch sizes and Transformer, Bert-large and Xlnet-large with more layers. 
Different from the previous experiment, most of data parallelism strategies are expected to be out of memory this time, while HeteroG can still find feasible solutions to deploy these large models.
