## Awesome Geo-localization
### University-1652

|Methods | R@1 | AP | R@1 | AP | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|Contrastive Loss | 52.39 | 57.44 | 63.91 | 52.24|
|Triplet Loss (margin=0.3)  | 55.18 | 59.97 | 63.62 | 53.85 |
|Triplet Loss (margin=0.5)  | 53.58 | 58.60 | 64.48 | 53.15 |
|Weighted Soft Margin Triplet Loss | 53.21 | 58.03 | 65.62 | 54.47| Liu L, Li H. Lending orientation to neural networks for cross-view geo-localization[C]. CVPR, 2019: 5624-5633. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Lending_Orientation_to_Neural_Networks_for_Cross-View_Geo-Localization_CVPR_2019_paper.pdf) |
|Instance Loss | 58.23 | 62.91 | 74.47 | 59.45 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535) |
|Instance Loss + Verification Loss | 61.30 | 65.68 | 75.04 | 62.87| Zheng Z, Zheng L, Yang Y. A discriminatively learned cnn embedding for person reidentification[J]. TOMM, 2017, 14(1): 1-20. [[Paper]](https://arxiv.org/pdf/1611.05666.pdf) |
|Instance Loss + Weighted Soft Margin Triplet Loss | 65.93 | 70.18 | 76.03 | 66.36|
|LCM (ResNet-50) | 66.65 | 70.82 | 79.89 |65.38 | Ding L, Zhou J, Meng L, et al. A Practical Cross-View Image Matching Method between UAV and Satellite for UAV-Based Geo-Localization[J]. Remote Sensing, 2021, 13(1): 47. [[Paper]](https://www.mdpi.com/2072-4292/13/1/47/pdf)|   
|LPN | 75.93 | 79.14 | 86.45 | 74.79 | Tingyu W, Zhedong Z, Chenggang Y, and Yi, Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. arXiv 2020. [[Paper]](https://arxiv.org/abs/2008.11646) |
|Instance Loss + Verification Loss + LPN | 77.08 | 80.18 | 85.02 | 73.80 |
|Instance Loss + Weighted Soft Margin Triplet Loss + LPN | 76.29 | 79.46 | 81.74 | 73.58 |

### cvusa
|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|Workman | - | - | - | 34.40 | Scott Workman, Richard Souvenir, and Nathan Jacobs. ICCV 2015. Wide-area image geolocalization with aerial reference imagery [[Paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Workman_Wide-Area_Image_Geolocalization_ICCV_2015_paper.pdf) |
|Zhai  | - | - | - | 43.20 | Menghua Zhai, Zachary Bessinger, Scott Workman, and Nathan Jacobs. CVPR 2017. Predicting ground-level scene layout from aerial imagery.[[Paper]](https://arxiv.org/abs/1612.02709) |
|Vo | - | - | - | 63.70 | Nam N Vo and James Hays. ECCV 2016. Localizing and orienting street views using overhead imagery| 
|CVM-Net | 18.80 | 44.42 | 57.47 | 91.54 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.html)| 
|Orientation* | 27.15 | 54.66 | 67.54 | 93.91 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/1903.12351) |
|Siam-FCANet | - | - | - | 98.3 | Sudong C, Yulan G, Salman K, et al. Ground-to-Aerial Image Geo-Localization With a Hard Exemplar Reweighting Triplet Loss. ICCV 2019. [[Paper]](https://salman-h-khan.github.io/papers/ICCV19-3.pdf) |
|Feature Fusion | 48.75 | - | 81.27 | 95.98 | Krishna Regmi, Mubarak Shah, et al. Bridging the Domain Gap for Ground-to-Aerial Image Matching. ICCV 2019. [[Paper]](https://arxiv.org/abs/1904.11045) |
|Instance Loss  | 43.91 | 66.38 | 74.58 | 91.78 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535)|
|CVFT | 61.43 | 84.69 | 90.49 | 99.02 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2020. [[Paper]](https://arxiv.org/abs/1907.05021) |
|SAFA | 89.84 | 96.93 | 98.14 | 99.64 | Yujiao Shi, Liu Liu, Xin Yu, et al. Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization. NIPS 2019. [[Paper]](http://papers.neurips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization) |
|SAFA + LPN | 92.83 | 98.00 | 98.85 | 99.78 | Tingyu W, Zhedong Z, Chenggang Y, and Yi, Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. arXiv 2020. [[Paper]](https://arxiv.org/abs/2008.11646) 
*: The method utilizes extra orientation information as input.

### cvact
|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|CVM-Net | 20.15 | 45.00 | 56.87 | 87.57 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.html)| 
|Instance Loss  | 31.20 | 53.64 | 63.00 | 85.27 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535)|
|Orientation* | 46.96 | 68.28 | 75.48 | 92.04 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/1903.12351) |
|CVFT | 61.05 | 81.33 | 86.52 | 95.93 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2020. [[Paper]](https://arxiv.org/abs/1907.05021) |
|SAFA | 81.03 | 92.80 | 94.84 | 98.17 | Yujiao Shi, Liu Liu, Xin Yu, et al. Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization. NIPS 2019. [[Paper]](http://papers.neurips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization) |
|SAFA + LPN | 83.66 | 94.14 | 95.92 | 98.41 | Tingyu W, Zhedong Z, Chenggang Y, and Yi, Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. arXiv 2020. [[Paper]](https://arxiv.org/abs/2008.11646) 
*: The method utilizes extra orientation information as input.
