<h1 align="center"> University1652-Baseline </h1>

![Python 3.6+](https://img.shields.io/badge/python-3.6+-green.svg)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/layumi/University1652-Baseline.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/University1652-Baseline/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/layumi/University1652-Baseline.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/layumi/University1652-Baseline/alerts/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![VideoDemo](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/youtube1.png)](https://www.youtube.com/embed/dzxXPp8tVn4?vq=hd1080)

[[Paper]](https://arxiv.org/abs/2002.12186) 
[[Slide]](http://zdzheng.xyz/files/ACM-MM-Talk.pdf)
[[Explore Drone-view Data]](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/sample_drone.jpg?raw=true)
[[Explore Satellite-view Data]](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/sample_satellite.jpg?raw=true)
[[Explore Street-view Data]](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/sample_street.jpg?raw=true)
[[Video Sample]](https://www.youtube.com/embed/dzxXPp8tVn4?vq=hd1080)
[[中文介绍]](https://zhuanlan.zhihu.com/p/110987552)

![](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/Data.jpg)

![](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/Motivation.png)


### Download [University-1652] upon request (Usually I will reply you in 5 minutes). You may use the request [template](https://github.com/layumi/University1652-Baseline/blob/master/Request.md).

This repository contains the dataset link and the code for our paper [University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization](https://arxiv.org/abs/2002.12186), ACM Multimedia 2020. The offical paper link is at https://dl.acm.org/doi/10.1145/3394171.3413896. We collect 1652 buildings of 72 universities around the world. Thank you for your kindly attention.

**Task 1: Drone-view target localization.** (Drone -> Satellite) Given one drone-view image or video, the task aims to find the most similar satellite-view image to localize the target building in the satellite view. 

**Task 2: Drone navigation.** (Satellite -> Drone) Given one satellite-view image, the drone intends to find the most relevant place (drone-view images) that it has passed by. According to its flight history, the drone could be navigated back to the target place.


### 1. ACM ICMR Workshop

**12 Jan 2024** We are holding a workshop at ACM ICMR 2024 on  Multimedia Object Re-ID. You are welcome to show your insights. See you at Phuket, Thailand!😃 The workshop link is https://www.zdzheng.xyz/MORE2024/ . Submission DDL is **15 April 2024**.

<details>
 <summary><b>
  2023 Workshop and Sepcial Session
</b></summary>

### 1.  IEEE ITSC Special Session
We host a special session on IEEE Intelligent Transportation Systems Conference (ITSC), covering the object re-identification & point cloud topic. The paper ddl is by **May 15, 2023** and the paper notification is at June 30, 2023. Please select the session code ``w7r4a'' during submission. More details can be found at [Special Session Website](https://2023.ieee-itsc.org/wp-content/uploads/2023/03/IEEE-ITSC-2023-Special-Session-Proposal-Safe-Critical-Scenario-Understanding-in-Intelligent-Transportation-Systems-SCSU-ITS.pdf).  

### 2. Remote Sensing Special Issue
We raise a special issue on Remote Sensing (IF=5.3) from now to ~~**16 June 2023**~~ **16 Dec 2023**. You are welcomed to submit your manuscript at (https://www.mdpi.com/journal/remotesensing/special_issues/EMPK490239), but you need to keep open-source fee in mind.

### 3. ACM Multimedia Workshop
We are holding the workshop at ACM Multimedia 2023 on Aerial-view Imaging. [Call for papers](https://www.zdzheng.xyz/ACMMM2023Workshop/) [中文介绍](https://zhuanlan.zhihu.com/p/620180604)

### 4. Coda Lab Challenge
We also provide a challenging cross-view geo-localization dataset, called University160k, and the workshop audience may consider to participate the competition. The motivation is to simulate the real- world geo-localization scenario that we usually face an extremely large satellite-view pool. In particular, University160k extends the current University-1652 dataset with extra 167,486 satellite- view gallery distractors. We have release University160k on the challenge page, and made a public leader board.
(More details are at https://codalab.lisn.upsaclay.fr/competitions/12672)

</details>

## Table of contents
* [About Dataset](#about-dataset)
* [News](#news)
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
| Split | #imgs | #buildings | #universities|
| --------   | -----  | ----| ----|
|Training | 50,218 | 701 | 33 |
| Query_drone | 37,855 | 701 |  39 |
| Query_satellite | 701 | 701 | 39|
| Query_ground | 2,579 | 701 | 39|
| Gallery_drone | 51,355 | 951 | 39|
| Gallery_satellite |  951 | 951 | 39|
| Gallery_ground | 2,921 | 793  | 39|

More detailed file structure:
```
├── University-1652/
│   ├── readme.txt
│   ├── train/
│       ├── drone/                   /* drone-view training images 
│           ├── 0001
|           ├── 0002
|           ...
│       ├── street/                  /* street-view training images 
│       ├── satellite/               /* satellite-view training images       
│       ├── google/                  /* noisy street-view training images (collected from Google Image)
│   ├── test/
│       ├── query_drone/  
│       ├── gallery_drone/  
│       ├── query_street/  
│       ├── gallery_street/ 
│       ├── query_satellite/  
│       ├── gallery_satellite/ 
│       ├── 4K_drone/
```

We note that there are no overlaps between 33 univeristies of training set and 39 univeristies of test set.

## News

**26 Jan 2023** 1652 Building Name List is at [Here](https://github.com/layumi/University1652-Baseline/blob/master/new_name_list.txt).

**10 Jul 2022** Rainy？Night？Foggy？ Snow？ You may check our new paper "Multiple-environment Self-adaptive Network for Aerial-view Geo-localization" at https://arxiv.org/pdf/2204.08381 

**1 Dec 2021** Fix the issue due to the latest torchvision, which do not allow the empty subfolder. Note that some buildings do not have google images.  

**3 March 2021** [GeM Pooling](https://cmp.felk.cvut.cz/~radenfil/publications/Radenovic-arXiv17a.pdf) is added. You may use it by `--pool gem`.

**21 January 2021** The GPU-Re-Ranking,  a GNN-based real-time post-processing code, is at [Here](GPU-Re-Ranking/).

**21 August 2020** The transfer learning code for Oxford and Paris is at [Here](https://github.com/layumi/cnnimageretrieval-pytorch/blob/master/cirtorch/examples/test_My1652model.py).

**27 July 2020** The meta data of 1652 buildings, such as latitude and longitude, are now available at [Google Driver](https://drive.google.com/file/d/1PL8fVky9KZg7XESsuS5NCsYRyYAwui3S/view?usp=sharing). (You could use Google Earth Pro to open the kml file or use vim to check the value).  
We also provide the spiral flight tour file at [Google Driver](https://drive.google.com/file/d/1EW5Esi72tPcfL3zmoHYpufKj_SXrY-xE/view?usp=sharing). (You could open the kml file via Google Earth Pro to enable the flight camera).  

**26 July 2020** The paper is accepted by ACM Multimedia 2020.

**12 July 2020** I made the baseline of triplet loss (with soft margin) on University-1652 public available at [Here](https://github.com/layumi/University1652-triplet-loss).

**12 March 2020** I add the [state-of-the-art](https://github.com/layumi/University1652-Baseline/tree/master/State-of-the-art) page for geo-localization and [tutorial](https://github.com/layumi/University1652-Baseline/tree/master/tutorial), which will be updated soon.

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

- Python 3.6+
- GPU Memory >= 8G
- Numpy > 1.12.1
- Pytorch 0.3+ 
- [Optional] apex (for float16) 

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source (Please check the README. Or directly install by anaconda. It will be Okay.)
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

For CVUSA, I follow the training/test split in (https://github.com/Liumouliu/OriCNN). 

## Train & Evaluation 
### Train & Evaluation University-1652
```
python train.py --name three_view_long_share_d0.75_256_s1_google  --extra --views 3  --droprate 0.75  --share  --stride 1 --h 256  --w 256 --fp16; 
python test.py --name three_view_long_share_d0.75_256_s1_google
```

Default setting: Drone -> Satellite
If you want to try other evaluation setting, you may change these lines at: https://github.com/layumi/University1652-Baseline/blob/master/test.py#L217-L225 

### Ablation Study only Satellite & Drone
```
python train_no_street.py --name two_view_long_no_street_share_d0.75_256_s1  --share --views 3  --droprate 0.75  --stride 1 --h 256  --w 256  --fp16; 
python test.py --name two_view_long_no_street_share_d0.75_256_s1
```
Set three views but set the weight of loss on street images to zero.

### Train & Evaluation CVUSA
```
python prepare_cvusa.py
python train_cvusa.py --name usa_vgg_noshare_warm5_lr2 --warm 5 --lr 0.02 --use_vgg16 --h 256 --w 256  --fp16 --batchsize 16;
python test_cvusa.py  --name usa_vgg_noshare_warm5_lr2 
```

### Show the retrieved Top-10 result 
```
python test.py --name three_view_long_share_d0.75_256_s1_google # after test
python demo.py --query_index 0 # which image you want to query in the query set 
```
It will save an image named `show.png' containig top-10 retrieval results in the folder. 

## Trained Model

You could download the trained model at [GoogleDrive](https://drive.google.com/open?id=1iES210erZWXptIttY5EBouqgcF5JOBYO) or [OneDrive](https://studentutsedu-my.sharepoint.com/:u:/g/personal/12639605_student_uts_edu_au/EW19pLps66RCuJcMAOtWg5kB6Ux_O-9YKjyg5hP24-yWVQ?e=BZXcdM). After download, please put model folders under `./model/`.

## Citation
The following paper uses and reports the result of the baseline model. You may cite it in your paper.
```bibtex
@article{zheng2020university,
  title={University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization},
  author={Zheng, Zhedong and Wei, Yunchao and Yang, Yi},
  journal={ACM Multimedia},
  year={2020}
}
@inproceedings{zheng2023uavm,
  title={UAVM'23: 2023 Workshop on UAVs in Multimedia: Capturing the World from a New Perspective},
  author={Zheng, Zhedong and Shi, Yujiao and Wang, Tingyu and Liu, Jun and Fang, Jianwu and Wei, Yunchao and Chua, Tat-seng},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={9715--9717},
  year={2023}
}
```
Instance loss is defined in 
```bibtex
@article{zheng2017dual,
  title={Dual-Path Convolutional Image-Text Embeddings with Instance Loss},
  author={Zheng, Zhedong and Zheng, Liang and Garrett, Michael and Yang, Yi and Xu, Mingliang and Shen, Yi-Dong},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  doi={10.1145/3383184},
  volume={16},
  number={2},
  pages={1--23},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```
## Related Work
- Instance Loss [Code](https://github.com/layumi/Image-Text-Embedding)
- Person re-ID from Different Viewpoints [Code](https://github.com/layumi/Person_reID_baseline_pytorch)
- Lending Orientation to Neural Networks for Cross-view Geo-localization [Code](https://github.com/Liumouliu/OriCNN)
- Predicting Ground-Level Scene Layout from Aerial Imagery [Code](https://github.com/viibridges/crossnet)
