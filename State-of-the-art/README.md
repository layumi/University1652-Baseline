## State-of-the-art
### University-1652

|Methods | R@1 | AP | R@1 | AP | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|Contrastive Loss | 52.39 | 57.44 | 63.91 | 52.24|
|Triplet Loss (margin=0.3)  | 55.18 | 59.97 | 63.62 | 53.85 |
|Triplet Loss (margin=0.5)  | 53.58 | 58.60 | 64.48 | 53.15 |
|Weighted Soft Margin Triplet Loss | 53.21 | 58.03 | 65.62 | 54.47|
|Instance Loss | 58.23 | 62.91 | 74.47 | 59.45 |


### cvusa
|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|Workman | - | - | - | 34.40 | Scott Workman, Richard Souvenir, and Nathan Jacobs. ICCV 2015. Wide-area image geolocalization with aerial reference imagery |
|Zhai  | - | - | - | 43.20 | Menghua Zhai, Zachary Bessinger, Scott Workman, and Nathan Jacobs. CVPR 2017. Predicting ground-level scene layout from aerial imagery. |
|Vo | - | - | - | 63.70 | Nam N Vo and James Hays. ECCV 2016. Localizing and orienting street views using overhead imagery| 
|CVM-Net | 18.80 | 44.42 | 57.47 | 91.54 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization.| 
|Orientation* | 27.15 | 54.66 | 67.54 | 93.91 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization|
|Ours  | 43.91 | 66.38 | 74.58 | 91.78 |
|CVFT | 61.43 | 84.69 | 90.49 | 99.02 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2019.|

*: The method utilizes extra orientation information as input.
