# Geo-Localization-Practical

[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial) (8 min read)

By [Zhedong Zheng](http://zdzheng.xyz/)


[[30-minute Talk in English (The talk wasn't fully recorded)]](https://www.youtube.com/watch?v=eG_UgzWRFqM&ab_channel=XiaohanZhang)

[[30-minute Talk in Chinese]](https://www.bilibili.com/video/BV138TszsEW9)

This is a [University of Macau](https://www.cis.um.edu.mo/) computer vision practical, authored by Zhedong Zheng.
The practical explores the basis of learning shared features for different platforms. In this practical, we will learn to build a simple geo-localization system step by step.  :+1: **Any suggestion is welcomed.**

![](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/top3.png)

We hope this tutorial could help the drone-related tasks, such as drone delivery (e.g., sending mask), event detection and agriculture. Next, we mainly focus on the two basic tasks: 

**Task 1: Drone-view target localization.** (Drone -> Satellite) Given one drone-view image or video, the task aims to find the most similar satellite-view image to localize the target building in the satellite view. 

**Task 2: Drone navigation.** (Satellite -> Drone) Given one satellite-view image, the drone intends to find the most relevant place (drone-view images) that it has passed by. According to its flight history, the drone could be navigated back to the target place.


## Keywords
Geo-Localization, University-1652, CVUSA, CVACT, Vigor

## Prerequisites
- Python 3.6
- GPU Memory >= 4G
- Numpy
- Pytorch 0.3+ (http://pytorch.org/)
- (Optional) Torchvision, which is usually already installed with pytorch.
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

## Troubleshooting
We do not suggest using Windows considering lower GPU usage and unexpected errors.

If you still want to use Windows, you should keep two points in mind. 

- Path: Ubuntu path is `\home\zzd\` but Windows path is `D://Downloads/` using `/` instead of `\` 
- Multi-thread: Pytorch does not support multiple thread on Windows to read the data. Please set `num_workers=0` during trainning and test.
- No Triton or other error: Please remove the `torch.compile` in training and test code.

Please also refer to https://github.com/layumi/Person_reID_baseline_pytorch/issues/34 

## Getting started
Check the Prerequisites. The download links for this practice are:

- Code: [Practical-Baseline](https://github.com/layumi/University1652-Baseline)
- Data: [University-1652](https://github.com/layumi/University1652-Baseline/blob/master/Request.md)

## Part 1: Training
### Part 1.1: Prepare Data Folder 
You may notice that the downloaded folder is organized as: 
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

In every subdir, such as `train/drone/0002`, images with the same ID are arranged in the folder.
Now we have successfully prepared the data for `torchvision` to read the data. 

### Part 1.2: Build Neural Network (`model.py`)
We can use the pretrained networks, such as `AlexNet`, `VGG16`, `ResNet` and `DenseNet`. Generally, the pretrained networks help to achieve a better performance, since it preserves some good visual patterns from ImageNet [1].

In pytorch, we can easily import them by two lines. For example,
```python
from torchvision import models
model = models.resnet50(pretrained=True)
```
You can simply check the structure of the model by:
```python
print(model)
```

But we need to modify the networks a little bit. There are 1652 classes (different buildings) in University-1652, which is different with 1,000 classes in ImageNet. So here we change the model to use our classifier.
```python
import torch
import torch.nn as nn
from torchvision import models

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool2(x)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'max':
            x = self.model.maxpool2(x)
            x = x.view(x.size(0), x.size(1))
        #x = self.classifier(x)
        return x
```

We have the data from three different platiforms, which may not share the low-level patterns. One straight-forward idea is to use backbone without sharing  weights. Here we re-use the class `ft_net` that we just defined to build `model_1` and `model_2`.

```python
class two_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False):
        super(two_view_net, self).__init__()
        if VGG16:
            self.model_1 =  ft_net_VGG16(class_num, stride=stride, pool = pool)
        else:
            self.model_1 =  ft_net(class_num, stride=stride, pool = pool)
        if share_weight:
            self.model_2 = self.model_1
        else:
            if VGG16:
                self.model_2 =  ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_2 =  ft_net(class_num, stride = stride, pool = pool)

        self.classifier = ClassBlock(2048, class_num, droprate)
        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        if VGG16:
            self.classifier = ClassBlock(512, class_num, droprate)
            if pool =='avg+max':
                self.classifier = ClassBlock(1024, class_num, droprate)

    def forward(self, x1, x2):
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            y2 = self.classifier(x2)
        return y1, y2
```
Note that the `classifier` does share weight, which is the key of [instance loss](https://arxiv.org/abs/1711.05535).

### Part 1.3: Training (`python train.py`)
OK. Now we have prepared the training data and defined model structure.

We can train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of the model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Let's look at what we do in the `train.py`.
The first thing is how to read data and their labels from the prepared folder.
Using `torch.utils.data.DataLoader`, we can obtain two iterators `dataloaders['train']` and `dataloaders['val']` to read data and label.
```python
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```

Here is the main code to train the model.
Yes. It's only about 20 lines. Make sure you can understand every line of the code.
```python
            # Iterate over data.
            for data in dataloaders[phase]:
                # get a batch of inputs
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable, if gpu is used, we transform the data to cuda.
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                #-------- forward --------
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                #-------- backward + optimize -------- 
                # only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
```
Every 10 training epoch, we save a snapshot and update the loss curve.
```python
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
```
## Part 2: Test
### Part 2.1: Extracting feature (`python test.py`)
In this part, we load the network weight (we just trained) to extract the visual feature of every image.
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--name` the dir name of the trained model.


`--batchsize` batch size.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.

Let's look at what we do in the `test.py`.
First, we need to import the model structure and then load the weight to the model.
```python
model_structure = ft_net(751)
model = load_network(model_structure)
```
For every query and gallery image, we extract the feature by simply forward the data.
```python
outputs = model(input_img) 
# ---- L2-norm Feature ------
ff = outputs.data.cpu()
fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
ff = ff.div(fnorm.expand_as(ff))
```
### Part 2.2: Evaluation
Yes. Now we have the feature of every image. The only thing we need to do is matching the images by the feature.
```bash
python evaluate_gpu.py
```

Let's look what we do in `evaluate_gpu.py`. We sort the predicted similarity score.
```python
query = qf.view(-1,1)
# print(query.shape)
score = torch.mm(gf,query) # Cosine Distance
score = score.squeeze(1).cpu()
score = score.numpy()
# predict index
index = np.argsort(score)  #from small to large
index = index[::-1]
```

Note that there are two kinds of images we do not consider as right-matching images.
* Junk_index1 is the index of mis-detected images, which contain the body parts.

* Junk_index2 is the index of the images, which are of the same identity in the same cameras.

```python
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    # The images of the same identity in different cameras
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # Only part of body is detected. 
    junk_index1 = np.argwhere(gl==-1)
    # The images of the same identity in same cameras
    junk_index2 = np.intersect1d(query_index, camera_index)
```

We can use the function `compute_mAP` to obtain the final result.
In this function, we will ignore the junk_index.
```python
CMC_tmp = compute_mAP(index, good_index, junk_index)
```

## Part 3: A simple visualization (`python demo.py`)
To visualize the result, 
```
python demo.py --query_index 233
```
`--query_index ` which query you want to test. You may select a number in the range of `0 ~ 700`. (query number)

It is similar to the `evaluate.py`. We add the visualization part.
```python
try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(10): #Show top-10 images
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green') # true matching
        else:
            ax.set_title('%d'%(i+1), color='red') # false matching
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
```

## Part 4: Your Turn. 

- University-1652 is a dataset collected in normal weather.

Let's try another dataset called [University1652-WX](https://github.com/wtyhub/MuseNet), which simulates different weathers.

![](https://github.com/wtyhub/MuseNet/raw/master/docs/visual.png)

## Part5: Other Related Works
- The building has some specific attributes, e.g., keypoints. They can help the feature learning. You could check [this code](https://github.com/AggMan96/RK-Net).
![](https://github.com/layumi/University1652-Baseline/blob/master/tutorial/RKNet.png?raw=true)

- Could we use natural language as query? Check [this paper](https://multimodalgeo.github.io/GeoText/).
![](https://multimodalgeo.github.io/GeoText/static/images/images/Fig2_1.jpg)


## Star History

If you like this repo, please star it. Thanks a lot!

[![Star History Chart](https://api.star-history.com/svg?repos=layumi/University1652-Baseline&type=Date)](https://star-history.com/#layumi/University1652-Baseline&Date)

## Reference
[1]University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization.
ACM Multimedia (ACM MM), 2020.
Zhedong Zheng, Yunchao Wei, Yi Yang

[2]Multiple-environment Self-adaptive Network for Aerial-view Geo-localization
Pattern Recognition (PR) 2024.
Tingyu Wang, Zhedong Zheng, Yaoqi Sun, Chenggang Yan, Yi Yang, Tat-Seng Chua

[3]Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization.
IEEE Transactions on Image Processing (TIP), 2022.
Jinliang Lin, Zhedong Zheng, Zhun Zhong, Zhiming Luo, Shaozi Li, Yi Yang, Nicu Sebe

[4]Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching.
European Conference on Computer Vision (ECCV), 2024.
Meng Chu, Zhedong Zheng, Wei Ji, Tingyu Wang, Tat-Seng Chua
