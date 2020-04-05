# Geo-Localization-Practical

[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial) 

By [Zhedong Zheng](http://zdzheng.xyz/)

This is a [University of Technology Sydney](https://www.uts.edu.au) computer vision practical, authored by Zhedong Zheng.
The practical explores the basis of learning shared features for different platforms. In this practical, we will learn to build a simple geo-localization system step by step. (8 min read) :+1: **Any suggestion is welcomed.**

![](https://github.com/layumi/University1652-Baseline/blob/master/docs/index_files/top3.jpg)

## Keywords
Geo-Localization, University-1652, CVUSA

## Prerequisites
- Python 3.6
- GPU Memory >= 4G
- Numpy
- Pytorch 0.3+ (http://pytorch.org/)
- Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

## Getting started
Check the Prerequisites. The download links for this practice are:

- Code: [Practical-Baseline](https://github.com/layumi/University1652-Baseline)
- Data: [University-1652](https://github.com/layumi/University1652-Baseline/blob/master/Request.md)

## Part 1: Training
### Part 1.1: Prepare Data Folder 
You may notice that the downloaded folder is organized as: 
```
├── Market/
│   ├── bounding_box_test/          /* Files for testing (candidate images pool)
│   ├── bounding_box_train/         /* Files for training 
│   ├── gt_bbox/                    /* Files for multiple query testing 
│   ├── gt_query/                   /* We do not use it
│   ├── query/                      /* Files for testing (query images)
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* train 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* val
│       ├── train_all/               /* train+val      
│       ├── query/                   /* query files  
│       ├── gallery/                 /* gallery files  
```

In every subdir, such as `pytorch/train/0002`, images with the same ID are arranged in the folder.
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
    def __init__(self, class_num = 751):
        super(ft_net, self).__init__()
        #load the model
        model_ft = models.resnet50(pretrained=True) 
        # change avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num) #define our classifier.

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #use our classifier.
        return x
```
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
