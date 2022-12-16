## Awesome Geo-localization

 * [University-1652 Dataset](#university-1652-dataset)
 * [cvusa Dataset](#cvusa-dataset)
 * [cvact Dataset](#cvact-val-dataset)

### News 

Recently, we raise a special issue on Remote Sensing (IF=5.349) from now to 16 June 2023. You are welcomed to submit your manuscript at (https://www.mdpi.com/journal/remotesensing/special_issues/EMPK490239), but you need to keep open-source fee in mind.

### University-1652 Dataset

Drone <-> Satellite 
|Methods | R@1 | AP | R@1 | AP | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|| Drone -> Satellite | | Satellite -> Drone |  |
|Contrastive Loss | 52.39 | 57.44 | 63.91 | 52.24|
|Triplet Loss (margin=0.3)  | 55.18 | 59.97 | 63.62 | 53.85 |
|Triplet Loss (margin=0.5)  | 53.58 | 58.60 | 64.48 | 53.15 |
|Weighted Soft Margin Triplet Loss | 53.21 | 58.03 | 65.62 | 54.47| Liu L, Li H. Lending orientation to neural networks for cross-view geo-localization[C]. CVPR, 2019: 5624-5633. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Lending_Orientation_to_Neural_Networks_for_Cross-View_Geo-Localization_CVPR_2019_paper.pdf) |
|Instance Loss | 58.23 | 62.91 | 74.47 | 59.45 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535) |
|Instance Loss + Verification Loss | 61.30 | 65.68 | 75.04 | 62.87| Zheng Z, Zheng L, Yang Y. A discriminatively learned cnn embedding for person reidentification[J]. TOMM, 2017, 14(1): 1-20. [[Paper]](https://arxiv.org/pdf/1611.05666.pdf) [[Code]](https://github.com/layumi/University1652-Baseline) |
|Instance Loss + GeM Pooling | 65.32	| 69.61	| 79.03	| 65.35| Radenović, Filip, Giorgos Tolias, and Ondřej Chum. "Fine-tuning CNN image retrieval with no human annotation." TPAMI (2018): 1655-1668. | 
|Instance Loss + Weighted Soft Margin Triplet Loss | 65.93 | 70.18 | 76.03 | 66.36|
|RK-Net (USAM) | 66.13 | 70.23 | 80.17 | 65.76 |  Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|LCM (ResNet-50) | 66.65 | 70.82 | 79.89 |65.38 | Ding L, Zhou J, Meng L, et al. A Practical Cross-View Image Matching Method between UAV and Satellite for UAV-Based Geo-Localization[J]. Remote Sensing, 2021, 13(1): 47. [[Paper]](https://www.mdpi.com/2072-4292/13/1/47/pdf)|   
|Instance Loss + GNN ReRanking |70.30| 74.11 | - | - | Zhang, Xuanmeng, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, and Yi Yang. "Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective." arXiv 2020. [[Paper]](https://arxiv.org/abs/2012.07620)[[Code]](https://github.com/layumi/University1652-Baseline/tree/master/GPU-Re-Ranking)|
|Instance Loss + USAM + SAFA | 72.19 | 75.79 | 83.23 | 71.77 |
|LPN | 75.93 | 79.14 | 86.45 | 74.79 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|LPN + CA-HRS | 76.67 | 79.77 | 86.88 | 74.84 | Zeng Lu, Tao Pu, Tianshui Chen, and Liang Lin. Content-Aware Hierarchical Representation Selection for Cross-View Geo-Localization ACCV2022. [[Paper]](https://openaccess.thecvf.com/content/ACCV2022/papers/Lu_Content-Aware_Hierarchical_Representation_Selection_for_Cross-View_Geo-Localization_ACCV_2022_paper.pdf)  [[Code]](https://github.com/Allen-lz/CA-HRS) |
|Instance Loss + Weighted Soft Margin Triplet Loss + LPN | 76.29 | 79.46 | 81.74 | 73.58 |
|Instance Loss + Verification Loss + LPN | 77.08 | 80.18 | 85.02 | 73.80 |
|Instance Loss + USAM + LPN | 77.60 | 80.55 | 86.59 | 75.96 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|LDRVSD| 78.66 | 81.55 | 89.30 | 79.17 | Qian Hu, Wansi Li, Xing Xu, Ning Liu, Lei Wang. Learning discriminative representations via variational self-distillation for cross-view geo-localization. Computers and Electrical Engineering 2022 [[Paper]](https://www.sciencedirect.com/science/article/pii/S0045790622005559) |
|PCL | 79.47 | 83.63 | 87.69 | 78.51 | Xiaoyang Tian, Jie Shao, Deqiang Ouyang, and Heng Tao Shen. UAV-Satellite View Synthesis for Cross-view Geo-Localization. TCSVT 2021. [[Paper]](https://ieeexplore.ieee.org/document/9583266) |
|FSRA (k=1)| 82.25 | 84.82 | 87.87 | 81.53 | Ming Dai, Jianhong Hu, Jiedong Zhuang, Enhui Zheng. A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization. TCSVT 2022. [[Paper]](https://arxiv.org/pdf/2201.09206.pdf)  [[Code]](https://github.com/dmmm1997/fsra) |
|FSRA (k=3)| 84.51 | 86.71 | 88.45 | 83.37 |
|PAAN | 84.51 | 86.78 | 91.01 | 82.28 | Duc Viet Bui, Masao Kubo, Hiroshi Sato. A Part-aware Attention Neural Network for Cross-view Geo-localization between UAV and Satellite.  Journal of Robotics Networking and Artificial Life 2022 [[Paper]](https://www.researchgate.net/profile/Viet-Bui-9/publication/366091845_A_Part-aware_Attention_Neural_Network_for_Cross-view_Geo-localization_between_UAV_and_Satellite/links/63914181e42faa7e75a6122e/A-Part-aware-Attention-Neural-Network-for-Cross-view-Geo-localization-between-UAV-and-Satellite.pdf) |


Ground <-> Satellite

| Methods | Training Set | R@1 | AP | R@1 | AP | Reference |
| -------- | -------- | ----- | ---- | ---- |  ---- |  ---- |
||| Ground -> Satellite | | Satellite -> Ground |  |
|Instance Loss | Satellite + Ground | 0.62 | 1.60 | 0.86 | 1.00| Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535)
|Instance Loss | Satellite + Drone + Ground| 1.28 | 2.29 | 1.57 | 1.52| 
|Instance Loss | Satellite + Drone + Ground + Google Image | 1.20 | 2.52 | 1.14 | 1.41| 
|LPN | Satellite + Ground | 0.74 | 1.83 | 1.43 | 1.31 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|LPN | Satellite + Drone + Ground | 0.81 | 2.21 | 1.85 | 1.66 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|PCLD| Satellite + Drone + Ground | 9.15 | 14.16 | - | - | Zeng, Z., Wang, Z., Yang, F., & Satoh, S. I. (2022). Geo-Localization via Ground-to-Satellite Cross-View Image Retrieval. IEEE Transactions on Multimedia. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9684950/)

### cvusa Dataset
|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|Workman | - | - | - | 34.40 | Scott Workman, Richard Souvenir, and Nathan Jacobs. ICCV 2015. Wide-area image geolocalization with aerial reference imagery [[Paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Workman_Wide-Area_Image_Geolocalization_ICCV_2015_paper.pdf) |
|Zhai  | - | - | - | 43.20 | Menghua Zhai, Zachary Bessinger, Scott Workman, and Nathan Jacobs. CVPR 2017. Predicting ground-level scene layout from aerial imagery.[[Paper]](https://arxiv.org/abs/1612.02709) |
|Vo | - | - | - | 63.70 | Nam N Vo and James Hays. ECCV 2016. Localizing and orienting street views using overhead imagery| 
|CVM-Net | 18.80 | 44.42 | 57.47 | 91.54 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.html)| 
|Orientation* | 27.15 | 54.66 | 67.54 | 93.91 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/1903.12351) |
|Siam-FCANet | - | - | - | 98.3 | Sudong C, Yulan G, Salman K, et al. Ground-to-Aerial Image Geo-Localization With a Hard Exemplar Reweighting Triplet Loss. ICCV 2019. [[Paper]](https://salman-h-khan.github.io/papers/ICCV19-3.pdf) |
|Feature Fusion | 48.75 | - | 81.27 | 95.98 | Krishna Regmi, Mubarak Shah, et al. Bridging the Domain Gap for Ground-to-Aerial Image Matching. ICCV 2019. [[Paper]](https://arxiv.org/abs/1904.11045) |
|Instance Loss  | 43.91 | 66.38 | 74.58 | 91.78 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535) [[Code]](https://github.com/layumi/University1652-Baseline)|
|RK-Net (USAM) | 52.50 | - | - | 96.52 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|CVFT | 61.43 | 84.69 | 90.49 | 99.02 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2020. [[Paper]](https://arxiv.org/abs/1907.05021) |
|MS Attention w DataAug| 75.95 | 91.90 | 95.00 | 99.42 |Rodrigues, Royston, and Masahiro Tani. "Are These From the Same Place? Seeing the Unseen in Cross-View Image Geo-Localization." WACV 2021. [[Paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Rodrigues_Are_These_From_the_Same_Place_Seeing_the_Unseen_in_WACV_2021_paper.pdf)|
|LPN| 85.79 | 95.38 | 96.98 | 99.41 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|LPN + CA-HRS | 87.16 | 95.98 | 97.55 | 99.49 | Zeng Lu, Tao Pu, Tianshui Chen, and Liang Lin. Content-Aware Hierarchical Representation Selection for Cross-View Geo-Localization ACCV2022. [[Paper]](https://openaccess.thecvf.com/content/ACCV2022/papers/Lu_Content-Aware_Hierarchical_Representation_Selection_for_Cross-View_Geo-Localization_ACCV_2022_paper.pdf)  [[Code]](https://github.com/Allen-lz/CA-HRS) |
|SAFA | 89.84 | 96.93 | 98.14 | 99.64 | Yujiao Shi, Liu Liu, Xin Yu, et al. Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization. NIPS 2019. [[Paper]](http://papers.neurips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization) |
|SAFA + USAM | 90.16 | - | - | 99.67 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|LPN + USAM | 91.22 | - | - | 99.67 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|DSM| 91.96 | 97.50 | 98.54 | 99.67 | Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li. "Where am i looking at? joint location and orientation estimation by cross-view matching." CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_CVPR_2020_paper.pdf) [[Code]](https://github.com/shiyujiao/cross_view_localization_DSM)| 
|Toker etal. | 92.56 | 97.55 | 98.33 | 99.67 | Aysim Toker, Qunjie Zhou, Maxim Maximov, Laura Leal-Taixé. Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization. CVPR 2021 [[Paper]](https://arxiv.org/pdf/2103.06818.pdf) | 
|SAFA + LPN | 92.83 | 98.00 | 98.85 | 99.78 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|Polar-EgoTR | 94.05 | 98.27 | 98.99 | 99.67 | Hongji Yang, Xiufan Lu, Yingying Zhu. Cross-view Geo-localization with Layer-to-Layer Transformer. NeurIPS 2021 [[Paper]](https://papers.nips.cc/paper/2021/file/f31b20466ae89669f9741e047487eb37-Paper.pdf) [[Code]](https://github.com/yanghongji2007/cross_view_localization_L2LTR)|
|TransGeo | 94.08 | 98.36 | 99.04 | 99.77 | Sijie Zhu, Mubarak Shah, Chen Chen. TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization. CVPR 2022 [[Paper]](https://arxiv.org/pdf/2204.00097.pdf) [[Code]](https://github.com/jeff-zilence/transgeo2022)|
| MGTL | 94.11 | 98.30 | 99.03 | 99.74 | Jianwei Zhao, Qiang Zhai, Rui Huang, Hong Cheng. Mutual Generative Transformer Learning for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/2203.09135)|
| GeoDTR | 93.76 | 98.47 | 99.22 | 99.85 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| GeoDTR* | 95.43 | 98.86 | 99.34 | 99.86 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| FI* | 95.50 | - | - | - | Wenmiao Hu, Yichen Zhang, Yuxuan Liang, Yifang Yin, Anderi Georgecu, An Tran, Hannes Kruppa, See-Kiong Ng, Roger Zimmermann. Beyond Geo-localization: Fine-grained Orientation of Street-view Images by Cross-view Matching with Satellite Imagery. ACM MM 2022 [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548102) |
*: The method utilizes extra orientation information as input.

### cvact val Dataset
|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|CVM-Net | 20.15 | 45.00 | 56.87 | 87.57 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.html)| 
|Instance Loss  | 31.20 | 53.64 | 63.00 | 85.27 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535) [[Code]](https://github.com/layumi/University1652-Baseline) |
|RK-Net (USAM) | 40.53 | - | - | 89.12 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|Orientation* | 46.96 | 68.28 | 75.48 | 92.04 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/1903.12351) |
|CVFT | 61.05 | 81.33 | 86.52 | 95.93 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2020. [[Paper]](https://arxiv.org/abs/1907.05021) |
|MS Attention w DataAug| 73.19 | 90.39 | 93.38 | 97.45 |Rodrigues, Royston, and Masahiro Tani. "Are These From the Same Place? Seeing the Unseen in Cross-View Image Geo-Localization." WACV 2021. [[Paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Rodrigues_Are_These_From_the_Same_Place_Seeing_the_Unseen_in_WACV_2021_paper.pdf)|
|LPN| 79.99 | 90.63 | 92.56 | 97.03 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|LPN + CA-HRS | 80.91 | 90.95 | 92.93 | 97.07 | Zeng Lu, Tao Pu, Tianshui Chen, and Liang Lin. Content-Aware Hierarchical Representation Selection for Cross-View Geo-Localization ACCV2022. [[Paper]]([https://arxiv.org/abs/2008.11646](https://openaccess.thecvf.com/content/ACCV2022/papers/Lu_Content-Aware_Hierarchical_Representation_Selection_for_Cross-View_Geo-Localization_ACCV_2022_paper.pdf))  [[Code]](https://github.com/Allen-lz/CA-HRS) |
|LDRVSD| 80.98 | 91.48 | 93.33 | - | Qian Hu, Wansi Li, Xing Xu, Ning Liu, Lei Wang. Learning discriminative representations via variational self-distillation for cross-view geo-localization. Computers and Electrical Engineering 2022|
|SAFA | 81.03 | 92.80 | 94.84 | 98.17 | Yujiao Shi, Liu Liu, Xin Yu, et al. Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization. NIPS 2019. [[Paper]](http://papers.neurips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization) |
|LPN + USAM | 82.02 | - | - | 98.18 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|SAFA + USAM | 82.40 | - | - | 98.00 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|DSM | 82.49 | 92.44 | 93.99 | 97.32 | Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li. "Where am i looking at? joint location and orientation estimation by cross-view matching." CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_CVPR_2020_paper.pdf) [[Code]](https://github.com/shiyujiao/cross_view_localization_DSM) | 
|Toker etal. | 83.28 | 93.57 | 95.42 | 98.22 | Aysim Toker, Qunjie Zhou, Maxim Maximov, Laura Leal-Taixé. Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization. CVPR 2021 [[Paper]](https://arxiv.org/pdf/2103.06818.pdf) |
|SAFA + LPN | 83.66 | 94.14 | 95.92 | 98.41 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|Polar-EgoTR | 84.89 | 94.59 | 95.96 | 98.37 | Hongji Yang, Xiufan Lu, Yingying Zhu. Cross-view Geo-localization with Layer-to-Layer Transformer. NeurIPS 2021 [[Paper]](https://papers.nips.cc/paper/2021/file/f31b20466ae89669f9741e047487eb37-Paper.pdf) [[Code]](https://github.com/yanghongji2007/cross_view_localization_L2LTR)|
|TransGeo | 84.95 | 94.14 | 95.78 | 98.37 | Sijie Zhu, Mubarak Shah, Chen Chen. TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization. CVPR 2022 [[Paper]](https://arxiv.org/pdf/2204.00097.pdf) [[Code]](https://github.com/jeff-zilence/transgeo2022)|
| MGTL | 85.35 | 94.45 | 96.06 | 98.48 | Jianwei Zhao, Qiang Zhai, Rui Huang, Hong Cheng. Mutual Generative Transformer Learning for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/2203.09135)|
| GeoDTR | 85.43 | 94.81 | 96.11 | 98.26 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| GeoDTR* | 86.21 | 95.44 | 96.72 | 98.77 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| FI* | 86.79 | - | - | - | Wenmiao Hu, Yichen Zhang, Yuxuan Liang, Yifang Yin, Anderi Georgecu, An Tran, Hannes Kruppa, See-Kiong Ng, Roger Zimmermann. Beyond Geo-localization: Fine-grained Orientation of Street-view Images by Cross-view Matching with Satellite Imagery. ACM MM 2022 [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548102) |
*: The method utilizes extra orientation information as input.
