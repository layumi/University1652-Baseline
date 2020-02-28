<h1 align="center"> University1652-Baseline </h1>

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/layumi/Person_reID_baseline_pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/Person_reID_baseline_pytorch/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/layumi/Person_reID_baseline_pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/Person_reID_baseline_pytorch/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[[Paper]](https://arxiv.org/abs/1711.05535) 

![](https://github.com/layumi/University1652-Baseline/blob/master/doc/index_files/Data.jpg)

This repository contains the code for our paper [University-1652: A Synthetic Benchmark for Drone-based Geo-localization](https://arxiv.org/abs/1711.05535). Thank you for your kindly attention.

## Table of contents
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Train Evaluation ](#train--evaluation)
* [Citation](#citation)

## Features
Now we have supported:
- Float16 to save GPU memory based on [apex](https://github.com/NVIDIA/apex)
- Multiple Query Evaluation
- Re-Ranking
- Random Erasing
- ResNet/VGG-16
- Visualize Training Curves
- Visualize Ranking Result
- Linear Warm-up 

## Prerequisites

- Python 3.6
- GPU Memory >= 8G
- Numpy > 1.12.1
- Pytorch 0.3+
- [Optional] apex (for float16) 

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optinal] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```

## Dataset & Preparation
Download [University-1652] or [CVUSA] upon request.

## Train & Evaluation 
### Train & Evaluation University-1652
```
python train.py --name three_view_long_share_d0.75_256_s1_google  --extra --views 3  --droprate 0.75  --share  --stride 1 --h 256  --w 256 --fp16; 
python test.py --name three_view_long_share_d0.75_256_s1_google
```

### Train & Evaluation CVUSA
```
python prepare_cvusa.py
python train_cvusa.py --name usa_vgg_noshare_warm5_lr2 --warm 5 --lr 0.02 --use_vgg16 --h 256 --w 256  --fp16 --batchsize 16;
python test_cvusa.py  --name usa_vgg_noshare_warm5_lr2 
```



## Citation
The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```
@article{zheng2020joint,
  title={},
  author={Zheng, Zhedong },
  journal={arXiv },
  year={2019}
}
```
