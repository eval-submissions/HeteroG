# HeteroG
## Abstract 
HeteroG is a software module that automatically produces deployment strategies for different DNNs in heterogeneous environments. To successfully launch HeteroG, each server used in experiments should be equipped with at least one GPU. 
The GNN in HeteroG is implemented using Tensorflow1.14 which is modified to support customized execution orders. 
The detailed software and hardware requirements are introduced in following sections. We will also introduce the dependencies needed and the procedures to install HeteroG, and the detailed steps to conduct corresponding experiments.

## Hardware dependency
We deploy HeteroG-boosted TensorFlow framework in 4 physical machines: one equipped with two NVIDIA 16GBTesla V100 GPUs, two 10-core Intel Xeon processor E5-2630v4 CPUs and one 100GbE Mellanox RDMA card; two equipped with two 11GB NVIDIA GTX 1080 Ti GPUs, one 8-core IntelXeon E5-1660 v4 CPU and one 50GbE Mellanox RDMA card;and the other equipped with two 12GB NVIDIA Tesla P100 GPUs, one 8-core Intel Xeon E5-1660 v4 CPU and one 50GbE Mellanox RDMA card. The machines are connected through a 100Gbps switch.

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
TBD

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
