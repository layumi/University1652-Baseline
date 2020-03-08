<h1 align="center"> University1652-Baseline </h1>

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/layumi/Person_reID_baseline_pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/Person_reID_baseline_pytorch/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/layumi/Person_reID_baseline_pytorch.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/Person_reID_baseline_pytorch/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![VideoDemo](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/youtube1.png)](https://www.youtube.com/embed/dzxXPp8tVn4?vq=hd1080)

[[Paper]](https://arxiv.org/abs/2002.12186) 
[[Explore Drone-view Data]](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/sample_drone.jpg?raw=true)
[[Explore Satellite-view Data]](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/sample_satellite.jpg?raw=true)
[[Explore Street-view Data]](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/sample_street.jpg?raw=true)
[[Video Sample]](https://www.youtube.com/embed/dzxXPp8tVn4?vq=hd1080)

![](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/Data.jpg)

This repository contains the dataset link and the code for our paper [University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization](https://arxiv.org/abs/2002.12186). We collect 1652 buildings of 72 universities around the world. Thank you for your kindly attention.

**Task 1: Drone-view target localization.** (Drone -> Satellite)} Given one drone-view image or video, the task aims to find the most similar satellite-view image to localize the target building in the satellite view. 

**Task 2: Drone navigation.** (Satellite -> Drone)}  Given one satellite-view image, the drone intends to find the most relevant place (drone-view images) that it has passed by. According to its flight history, the drone could be navigated back to the target place.

## Table of contents
* [About Dataset](#about-dataset)
* [Code Features](#code-features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Dataset Preparation](#dataset--preparation)
    * [Train Evaluation ](#train--evaluation)
    * [Trained Model](#trained--model)
* [Citation](#citation)

## About Dataset
The dataset split is as follows: 
| Split | #imgs | #classes | #universities|
| --------   | -----  | ----| ----|
|Training | 50,218 | 701 | 33 |
| Query_drone | 37,855 | 701 |  39 |
| Query_satellite | 701 | 701 | 39|
| Query_ground | 2,579 | 701 | 39|
| Gallery_drone | 51,355 | 951 | 39|
| Gallery_satellite |  951 | 951 | 39|
| Gallery_ground | 2,921 | 793  | 39|

We note that there are no ovelaps between 33 univeristies of training set and 39 univeristies of test set.

## Code Features
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
Download [University-1652] upon request. You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

Or download [CVUSA](http://cs.uky.edu/~jacobs/datasets/cvusa/) / [CVACT](https://github.com/Liumouliu/OriCNN).

## Train & Evaluation 
### Train & Evaluation University-1652
```
python train.py --name three_view_long_share_d0.75_256_s1_google  --extra --views 3  --droprate 0.75  --share  --stride 1 --h 256  --w 256 --fp16; 
python test.py --name three_view_long_share_d0.75_256_s1_google
```

Default setting: Drone -> Satellite
If you want to try other evaluation setting, you may change these lines at: https://github.com/layumi/University1652-Baseline/blob/master/test.py#L217-L225

### Train & Evaluation CVUSA
```
python prepare_cvusa.py
python train_cvusa.py --name usa_vgg_noshare_warm5_lr2 --warm 5 --lr 0.02 --use_vgg16 --h 256 --w 256  --fp16 --batchsize 16;
python test_cvusa.py  --name usa_vgg_noshare_warm5_lr2 
```

## Trained Model

You could download the trained model at [GoogleDrive](https://drive.google.com/open?id=1iES210erZWXptIttY5EBouqgcF5JOBYO). After download, please put model folders under `./model/`.

## Citation
The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```
@article{zheng2020joint,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zhedong Zheng, Yunchao Wei, Yi Yang},
  journal={arXiv 2002.12186},
  year={2020}
}
```

## Related Work
- Instance Loss [Code](https://github.com/layumi/Image-Text-Embedding)
- Lending Orientation to Neural Networks for Cross-view Geo-localization [Code](https://github.com/Liumouliu/OriCNN)
- Predicting Ground-Level Scene Layout from Aerial Imagery [Code](https://github.com/viibridges/crossnet)
