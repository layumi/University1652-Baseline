# Awesome Geo-localization

 - [University-1652 Dataset](#university-1652-dataset) : a dataset containing 1652 locations of global universities, with images captured from ground, drone, and satellite perspectives. `ACM MM 2020`
     - [University-WX](https://github.com/wtyhub/MuseNet): ***(Weather)*** Extension of University-1652 with multiple weathers on the fly. `Pattern Recognition 2024`
     - [University160k](https://codalab.lisn.upsaclay.fr/competitions/12672): ***(Larger Gallery)*** Large-scale extension of University-1652 with +167,486 distractors for realistic geo-localization testing. Always-open eval server & leaderboard: https://codalab.lisn.upsaclay.fr/competitions/12672
     - [GeoText-1652](https://github.com/MultimodalGeo/GeoText-1652): ***(Text)*** Dense text extension of University-1652. `ECCV2024`
     - [UniV](https://github.com/HaoDot/Video2BEV-Open): ***(Video)*** Video extension with 30° and 45° viewing angles. `ICCV2025`
     - [RoadMap](https://www.zdzheng.xyz/publication/Road-Map2026): ***(RoadMap)*** Driving Roadmap extension of University-1652 (data is released).
     - [PairUAV](https://github.com/YaxuanLi-cn/PairUAV): ***(Pose Estimation)*** Visual servoing extension of University-1652. 
 - [CVUSA Dataset](#cvusa) : a dataset in America, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.
The dataset can be accessed from https://github.com/viibridges/crossnet
 - [CVACT Dataset](#cvact-val) : a dataset in Australia, with pairs of ground-level images and satellite images. All ground-level images are panoramic images.
The dataset can be accessed from https://github.com/Liumouliu/OriCNN
- [UAVReason Dataset](https://github.com/JT-Sun/UAVReason) : a unified, large-scale benchmark for multimodal aerial scene reasoning and generation on UAV-view images. Built on UAVScenes RGB images, it provides VQA / caption annotations, image-to-image generation JSONL files, and additional depth data, supporting UAV visual question answering, scene captioning, spatial / temporal / heading reasoning, depth-aware perception, and cross-modal generation. The repository provides the data usage guide and BAGEL data adaptation scripts.
The dataset can be accessed from https://github.com/JT-Sun/UAVReason

 - [DenseUAV Dataset](#denseuav-dataset) : a large-scale benchmark for UAV self-positioning in low-altitude urban environments, densely sampled over 14 university campuses. ***(Retrieval)*** `TIP 2023`
 - [SUES-200 Dataset](#sues-200-dataset-150-m) : a multi-height (150 / 200 / 250 / 300 m) multi-scene benchmark for cross-view matching between UAV and satellite imagery. ***(Retrieval)*** `TCSVT 2023`
 

Keywords: Cross-view Geo-localization, Spatial Intelligence, Aerial Agents.

## News 

- Multi-weather University-WX leaderboard is available at https://github.com/wtyhub/MuseNet/blob/master/State-of-the-art.md .

## University-1652 Dataset

### Drone <-> Satellite 

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
|DWDR | 69.77 | 73.73 | 81.46 | 70.45 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" IEEE Transactions on Geoscience and Remote Sensing, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10744586) [[Code]](https://github.com/wtyhub/DWDR) |
|Instance Loss + GNN ReRanking |70.30| 74.11 | - | - | Zhang, Xuanmeng, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, and Yi Yang. "Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective." arXiv 2020. [[Paper]](https://arxiv.org/abs/2012.07620)[[Code]](https://github.com/layumi/University1652-Baseline/tree/master/GPU-Re-Ranking)|
|Instance Loss + USAM + SAFA | 72.19 | 75.79 | 83.23 | 71.77 |
|MuSe-Net (Normal Weather) | 74.48 | 77.83 |88.02 | 75.10 | Wang T, Zheng Z, Sun Y, et al. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization[J]. Pattern Recognition, 2024. [[Code]](https://github.com/wtyhub/MuseNet) |
|LPN | 75.93 | 79.14 | 86.45 | 74.79 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|LPN + CA-HRS | 76.67 | 79.77 | 86.88 | 74.84 | Zeng Lu, Tao Pu, Tianshui Chen, and Liang Lin. Content-Aware Hierarchical Representation Selection for Cross-View Geo-Localization ACCV2022. [[Paper]](https://openaccess.thecvf.com/content/ACCV2022/papers/Lu_Content-Aware_Hierarchical_Representation_Selection_for_Cross-View_Geo-Localization_ACCV_2022_paper.pdf)  [[Code]](https://github.com/Allen-lz/CA-HRS) |
|Instance Loss + Weighted Soft Margin Triplet Loss + LPN | 76.29 | 79.46 | 81.74 | 73.58 |
|Instance Loss + Verification Loss + LPN | 77.08 | 80.18 | 85.02 | 73.80 |
|Instance Loss + USAM + LPN | 77.60 | 80.55 | 86.59 | 75.96 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|F3Net| 78.64 | 81.60 | - | - | Bo Sun, Ganchao Liu and Yuan Yuan. F3-Net: Multiview Scene Matching for Drone-Based Geo-Localization. IEEE TGRS, 61: 1-11, 2023. [[Paper]](https://doi.org/10.1109/TGRS.2023.3278257) |
|SAIG-D| 78.85 | 81.62 | 86.45 | 78.48 | Yingying Zhu, Hongji Yang, Yuxin Lu and Qiang Huang. Simple, Effective and General: A New Backbone for Cross-view Image Geo-localization. ArXiv 2023 [[Code]](https://github.com/yanghongji2007/SAIG) |
|LDRVSD| 78.66 | 81.55 | 89.30 | 79.17 | Qian Hu, Wansi Li, Xing Xu, Ning Liu, Lei Wang. Learning discriminative representations via variational self-distillation for cross-view geo-localization. Computers and Electrical Engineering 2022 [[Paper]](https://www.sciencedirect.com/science/article/pii/S0045790622005559) |
|EAGLe (FSRA) (Normal Weather) | 79.24 | 82.87 |88.30 | 79.55 | Cuiwei Liu, Shiting Peng, Shishen Li, Huaijun Qiu, Yuhao Xia and Zhaokui Li. A Novel EAGLe Framework for Robust UAV-View Geo-Localization. IEEE Transactions on Geoscience and Remote Sensing, 2025.  [[Paper]](https://ieeexplore.ieee.org/abstract/document/11048970)|
|PCL | 79.47 | 83.63 | 87.69 | 78.51 | Xiaoyang Tian, Jie Shao, Deqiang Ouyang, and Heng Tao Shen. UAV-Satellite View Synthesis for Cross-view Geo-Localization. TCSVT 2021. [[Paper]](https://ieeexplore.ieee.org/document/9583266) |
|SCPNet | 79.96 |83.04 | 87.33 | 79.87| Yuan Gao, Haibo Liu, and Xiaohui Wei. Semantic Concept Perception Network With Interactive Prompting for Cross-View Image Geo-Localization. TCSVT 2025. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10852334#full-text-header) |
|LPN + DWDR | 81.51 | 84.11 | 88.30 | 79.38 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" IEEE Transactions on Geoscience and Remote Sensing, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10744586) [[Code]](https://github.com/wtyhub/DWDR) |
|FSRA (k=1)| 82.25 | 84.82 | 87.87 | 81.53 | Ming Dai, Jianhong Hu, Jiedong Zhuang, Enhui Zheng. A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization. TCSVT 2022. [[Paper]](https://arxiv.org/pdf/2201.09206.pdf)  [[Code]](https://github.com/dmmm1997/fsra) |
|WeatherPrompt (Normal Weather) | 82.78 | 85.18 |89.16 | 81.80 | Jiahao Wen, Hang Yu, and Zhedong Zheng. WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization. NeurIPS 2025.  [[Paper]](https://arxiv.org/abs/2508.09560) [[Code]](https://github.com/Jahawn-Wen/WeatherPrompt) |
|FSRA (k=3)| 84.51 | 86.71 | 88.45 | 83.37 | Ming Dai, Jianhong Hu, Jiedong Zhuang, Enhui Zheng. A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization. TCSVT 2022. [[Paper]](https://arxiv.org/pdf/2201.09206.pdf)  [[Code]](https://github.com/dmmm1997/fsra) |
| + UniGeoRS | 83.82 | 86.21 | 88.59 | 82.78 | Liang, X., Tang, H., Zhang, F., Yuan, S., Hu, C., Zheng, D., & Ma, K. (2026). UniGeoRS: A Unified Benchmark for Tri-view Geo-Localization. CVPR 2026. [[Code]](https://github.com/BIT-MSense/UniGeoRS) |
|TransFG| 84.01 | 86.31 | 90.16 | 84.61 |  Zhao, H., Ren, K., Yue, T., Zhang, C., & Yuan, S. (2024). TransFG: A Cross-View Geo-Localization of Satellite and UAVs Imagery Pipeline Using Transformer-Based Feature Aggregation and Gradient Guidance. IEEE Transactions on Geoscience and Remote Sensing. |
|PAAN | 84.51 | 86.78 | 91.01 | 82.28 | Duc Viet Bui, Masao Kubo, Hiroshi Sato. A Part-aware Attention Neural Network for Cross-view Geo-localization between UAV and Satellite.  Journal of Robotics Networking and Artificial Life 2022 [[Paper]](https://www.researchgate.net/profile/Viet-Bui-9/publication/366091845_A_Part-aware_Attention_Neural_Network_for_Cross-view_Geo-localization_between_UAV_and_Satellite/links/63914181e42faa7e75a6122e/A-Part-aware-Attention-Neural-Network-for-Cross-view-Geo-localization-between-UAV-and-Satellite.pdf) |
|Swin-B + DWDR | 86.41 | 88.41 | 91.30 | 86.02 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" IEEE Transactions on Geoscience and Remote Sensing, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10744586) [[Code]](https://github.com/wtyhub/DWDR) |
|MBF | 89.05 | 90.61 | 93.15| 88.17 | Runzhe Zhu , Mingze Yang , Ling Yin * , Fei Wu and Yuncheng Yang. "UAV’s Status Is Worth Considering: A Fusion Representations Matching Method for Geo-Localization" Sensors [[Code]](https://github.com/Reza-Zhu/MBF) |
|MCCG| 89.64 | 91.32 | 94.30 | 89.39 | Tianrui Shen, Yingmei Wei, Lai Kang, Shanshan Wan and Yee-Hong Yang. MCCG: A ConvNeXt-based Multiple-Classifier Method for Cross-view Geo-localization. TCSVT 2023 [[Code]](https://github.com/mode-str/crossview) |
|SDPL | 90.16 |91.64|93.58 |89.45| Quan Chen, Tingyu Wang, Zihao Yang, Haoran Li, Rongfeng Lu and Yaoqi Sun. SDPL: Shifting-dense partition learning for UAV-view geo-localization. TCSVT 2024. [[Paper]](https://ieeexplore.ieee.org/document/10587023) [[Code]](https://github.com/C-water/SDPL_release) |
| MFJR | 91.87 | 93.15 | 95.29 | 91.51 | Ge, F., Zhang, Y., Wang, L., Liu, W., Liu, Y., Coleman, S., & Kerr, D. (2024). Multi-level Feedback Joint Representation Learning Network Based on Adaptive Area Elimination for Cross-view Geo-localization. IEEE Transactions on Geoscience and Remote Sensing. |
|CCR | 92.54 | 93.78 | 95.15 | 91.80 | Du, H., He, J., & Zhao, Y. (2024). CCR: A Counterfactual Causal Reasoning-Based Method for Cross-View Geo-Localization. IEEE TCSVT, 34(11): 11630-11643. |
|Sample4Geo| 92.65 | 93.81 | 95.14 | 91.39 | Fabian Deuser, Konrad Habel, Norbert Oswald. Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation. ICCV 2023 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.html) [[Code]](https://github.com/Skyy93/Sample4Geo) |
|CLNet | 93.02 | 94.18 | 98.31 | 91.63 | Cao, X., Quan, D., Wang, S., Huyan, N., Wang, W., Li, Y., & Jiao, L. (2025). CLNet: Cross-View Correspondence Makes a Stronger Geo-Localizationer. arXiv 2025. [[Paper]](https://arxiv.org/abs/2512.14560) |
| MEAN | 93.55 | 94.53 | 96.01 | 92.08 | Chen, Zhongwei, Zhao-Xu Yang, and Hai-Jun Rong. "Multi-level embedding and alignment network with consistency and invariance learning for cross-view geo-localization." TGRS 2025. |
| APA-BI | 93.57 | 94.55 | 95.86 | 92.88 | Xichen Zhang, Shuying Zhao, Yunzhou Zhang, Fawei Ge, Bin Zhao and Yizhong Zhang. APA-BI: Adaptive Partition Aggregation and Bidirectional Integration for UAV-View Geo-Localization. ICRA 2025 [[Paper]](https://ieeexplore.ieee.org/abstract/document/11128402)| 
|UniABG (Unsupervised) | 93.62 | 94.61 | 95.43 | 93.29 | Chen, C., Chen, Q., Yang, B., & Zhang, X. (2026). UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization. AAAI 2026 (Oral). [[Paper]](https://arxiv.org/abs/2511.12054) [[Code]](https://github.com/chenqi142/UniABG) |
|SHAA | 93.69 |94.68| 96.15 |93.49| Nanhua Chen, Dongshuo Zhang, Kai Jiang, Meng Yu, Yeqing Zhu, and Tai-shan Lou. SHAA: Spatial Hybrid Attention Network With Adaptive Cross-Entropy Loss Function for UAV-View Geo-Localization. TCSVT 2025. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10965775) |
| MFRGN | 94.33 | 95.24	| 96.15 | 93.94 | Wang, Y., Zhang, J., Wei, R., Gao, W., & Wang, Y. MFRGN: Multi-scale Feature Representation Generalization Network for Ground-to-Aerial Geo-localization. ACM MM2024. [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3664647.3681431) [[Code]](https://github.com/ytao-wang/MFRGN) |
| CAMP | 94.46 | 95.38 | 96.15 | 92.72 | Wu, Q., Wan, Y., Zheng, Z., Zhang, Y., Wang, G., & Zhao, Z. (2024). Camp: A cross-view geo-localization method using contrastive attributes mining and position-aware partitioning. TGRS 2024. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10644040) [[Code]](https://github.com/Mabel0403/CAMP) |
| DAC | 94.67 | 95.50 | 96.43 | 93.79 | Xia, P., Wan, Y., Zheng, Z., Zhang, Y., & Deng, J. (2024). Enhancing cross-view geo-localization with domain alignment and scene consistency. TCSVT 2024. [[Paper]](https://ieeexplore.ieee.org/document/10636268) [[Code]](https://github.com/SummerpanKing/DAC)|
| QDFL | 95.00 | 95.83 | 97.15 | 94.57 | Hu, S., Shi, Z., Jin, T., & Liu, Y. Query-Driven Feature Learning for Cross-View Geo-Localization. TGRS 2025. | 
| CDM-Net | 95.13 | 96.04 | 96.68 | 94.05 | Zhou, Xin, Xuerong Yang, and Yanchun Zhang. "CDM-Net: A Framework for Cross-View Geo-Localization With Multimodal Data." TGRS 2025. |
| JRN-Geo | 95.13 | 95.85 | 96.72 | 94.93 | Zhou, H., Zhang, Y., Huang, T., Ge, F., Qi, M., Zhang, X., & Zhang, Y. (2025, May). JRN-Geo: A Joint Perception Network based on RGB and Normal images for Cross-view Geo-localization. ICRA 2025 | 
| CGSI (DinoV2+BERT) | 95.45 | 96.10 | 96.58 | 95.38 | Sun, J., Huang, J., Jiang, X., Zhou, Y., & VONG, C. M. CGSI: Context-Guided and UAV’s Status Informed Multimodal Framework for Generalizable Cross-View Geo-Localization. TCSVT 2025 |
| ONLoc* | 95.65 | 96.45 | - | - | Qiao, Q., Liu, W., Liu, T., Shu, J., & Wang, P. (2026). OffNadirLoc: Benchmark and Framework for Challenging UAV-to-Satellite Geo-Localization under Large Off-Nadir Views. CVPR Findings 2026. | 
|DINO-GFSA | 95.68 | 96.34 | 96.29 | 95.56 | Hu, B., Guo, Y., Cai, J., Li, C., Wang, Y., Wu, S., & Wu, Z. (2026). DINO-GFSA: Geo-Localization via Semantic Gated Fusion and Mamba-based Sequential Aggregation. arXiv 2026. [[Paper]](https://arxiv.org/abs/2606.00784) [[Code]](https://github.com/Bear611/DINO-GFSA) |
| GeoBridge* | 95.82 | 97.77 | 97.14 | 95.05 | Z Song, J Zhang, D Wang, Z Zhou, W Liu, H Guo, E Wang, B Du (2026). Geobridge: A semantic-anchored multi-view foundation model bridging images and text for geo-localization. CVPR 2026 [[Code]](https://github.com/MiliLab/GeoBridge) |
|BGG | 96.24 | 96.81 | 97.57 | 95.44 | Wang, W., Quan, D., Huyan, N., Wang, S., Li, Y., He, P., & Jiao, L. (2026). BGG: Bridging the Geometric Gap between Cross-View Images by Vision Foundation Model Adaptation for Geo-Localization. arXiv 2026. [[Paper]](https://arxiv.org/abs/2605.10345) |
|Warp-free (GeoCoM) | 97.42 | 97.84 | 97.72 | 96.40 | Song, Z., Xu, L., Jiang, R., Zhang, Y., Li, K., Zhang, Y., & Guo, Y. (2026). Warp-free Cross-view Geo-Localization via Feature-space Consensus Mining. ECCV 2026. [[Paper]](https://arxiv.org/abs/2608.09321) |
|(MGS)²-Net | 97.60 | 98.03 | 98.86 | 97.25 | Li, M., He, M., Li, C., Chen, C., Shao, X., & Meng, Z. (2026). (MGS)²-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization (336×336 test). arXiv 2026. [[Paper]](https://arxiv.org/abs/2602.10704) |

* Extra dataset is used. 

[Multi-weather leaderboard](https://github.com/wtyhub/MuseNet/blob/master/State-of-the-art.md)

### Ground <-> Satellite

| Methods | Training Set | R@1 | AP | R@1 | AP | Reference |
| -------- | -------- | ----- | ---- | ---- |  ---- |  ---- |
||| Ground -> Satellite | | Satellite -> Ground |  |
|Instance Loss | Satellite + Ground | 0.62 | 1.60 | 0.86 | 1.00| Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535)
|Instance Loss | Satellite + Drone + Ground| 1.28 | 2.29 | 1.57 | 1.52| 
|Instance Loss | Satellite + Drone + Ground + Google Image | 1.20 | 2.52 | 1.14 | 1.41| 
|Baseline | + UniGeoRS dataset | 1.01 | 3.25 | - | - | Liang, X., Tang, H., Zhang, F., Yuan, S., Hu, C., Zheng, D., & Ma, K. UniGeoRS: A Unified Benchmark for Tri-view Geo-Localization. CVPR 2026. [[Code]](https://github.com/BIT-MSense/UniGeoRS) |
|LPN | Satellite + Ground | 0.74 | 1.83 | 1.43 | 1.31 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|LPN | Satellite + Drone + Ground | 0.81 | 2.21 | 1.85 | 1.66 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|CVGS| Unpaired Satellite + Drone + Ground | 0.89 | 2.80| - | - | Xie, K., Zhou, W., Huang, X., Guan, H., & Yulong, F. (2025). Self-supervised Cross-view Graph Search Framework for Ground-to-Satellite Geo-localization. TGRS 2025. | 
|PCLD| Satellite + Drone + Ground | 9.15 | 14.16 | - | - | Zeng, Z., Wang, Z., Yang, F., & Satoh, S. I. (2022). Geo-Localization via Ground-to-Satellite Cross-View Image Retrieval. IEEE Transactions on Multimedia. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9684950/) |
| Street2orbit | Google Search| 25.57 | - | - | - |  Min, Jeongho, Dongyoung Kim, and Jaehyup Lee. "From Street to Orbit: Training-Free Cross-View Retrieval via Location Semantics and LLM Guidance." WACV. 2026. [[Code]](https://github.com/jeonghomin/street2orbit) |
|VICI| Satellite + Ground | 24.66 | -  | - | - | Zhang, X., Shore, T., Chen, C., Mendez, O., Hadfield, S., & Wshah, S. (2025). VICI: VLM-Instructed Cross-view Image-localisation. ACM MM UAVM Workshop 2025. [[Paper]](https://arxiv.org/pdf/2507.04107)[[Code]](https://github.com/tavisshore/VICI) |
|VICI| Satellite + Drone + Ground | 27.49 | -  | - | - | |
|VICI + VLM| Satellite + Drone + Ground | 30.21 | -  | - | - | |



## DenseUAV Dataset

[DenseUAV](https://github.com/Dmmm1997/DenseUAV) is a large-scale benchmark for UAV self-positioning in low-altitude urban environments, with dense sampling over 14 university campuses in Zhejiang, China. Only the **retrieval** results are listed below.

**Note:** The literature reports two evaluation protocols. Their numbers are **not comparable** and are therefore kept in separate tables.

### (a) Standard Protocol

Official split, clear weather. Sorted by Drone -> Satellite R@1.

|Methods | R@1 | R@5 | Reference |
| -------- | ----- | ---- |  ---- |
|MSBA | 46.13 | 64.22 | Zhuang, J., Dai, M., Chen, X., & Zheng, E. (2021). A Faster and More Effective Cross-View Matching Method of UAV and Satellite Images for UAV Geolocalization. Remote Sensing, 13(19): 3979. |
|FSRA (block=2)† | 82.58 | 94.94 | Ming Dai, Jianhong Hu, Jiedong Zhuang, Enhui Zheng. A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization. TCSVT 2022. [[Paper]](https://arxiv.org/pdf/2201.09206.pdf)  [[Code]](https://github.com/dmmm1997/fsra) |
|DenseUAV Baseline (ViT-S) | 83.01 | 95.58 | Dai, M., Zheng, E., Feng, Z., Qi, L., Zhuang, J., & Yang, W. (2023). Vision-Based UAV Self-Positioning in Low-Altitude Urban Environments. IEEE TIP, 33: 493-508. [[Paper]](https://arxiv.org/abs/2201.09201) [[Code]](https://github.com/Dmmm1997/DenseUAV) |
|LPN (block=2)† | 83.05 | 94.89 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646)  [[Code]](https://github.com/wtyhub/LPN) |
|DINOv2-based | 86.27 | 96.83 | Yang, J., Qin, D., Tang, H., Tao, S., Bie, H., & Ma, L. (2025). DINOv2-Based UAV Visual Self-Localization in Low-Altitude Urban Environments. IEEE RA-L. |
|MCCG | 89.19 | 96.87 | Tianrui Shen, Yingmei Wei, Lai Kang, Shanshan Wan and Yee-Hong Yang. MCCG: A ConvNeXt-based Multiple-Classifier Method for Cross-view Geo-localization. TCSVT 2023. [[Code]](https://github.com/mode-str/crossview) |
|SHAA | 93.69 | 98.76 | Nanhua Chen, Dongshuo Zhang, Kai Jiang, Meng Yu, Yeqing Zhu, and Tai-shan Lou. SHAA: Spatial Hybrid Attention Network With Adaptive Cross-Entropy Loss Function for UAV-View Geo-Localization. TCSVT 2025. |
|DINO-GFSA | 97.17 | 99.57 | Hu, B., Guo, Y., Cai, J., Li, C., Wang, Y., Wu, S., & Wu, Z. (2026). DINO-GFSA: Geo-Localization via Semantic Gated Fusion and Mamba-based Sequential Aggregation. [[Paper]](https://arxiv.org/abs/2606.00784) [[Code]](https://github.com/Bear611/DINO-GFSA) |

†: evaluated with a ViT-S backbone at 512-d in the head ablation of the DenseUAV paper.

### (b) Multi-weather Protocol

Mean R@1 / AP over ten weather conditions (normal, fog, rain, snow, fog+rain, fog+snow, rain+snow, dark, overexposure, wind), following the [multi-weather leaderboard](https://github.com/wtyhub/MuseNet/blob/master/State-of-the-art.md).

|Methods | R@1 | AP | R@1 | AP | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|| Drone -> Satellite | | Satellite -> Drone |  |
|Safe-Net | 13.03 | 16.84 | 14.74 | 19.03 | Jinliang Lin, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong. A Self-Adaptive Feature Extraction Method for Aerial-View Geo-Localization. TIP 2025. |
|WeatherPrompt | 29.25 | 35.19 | 27.78 | 33.91 | Jiahao Wen, Hang Yu, and Zhedong Zheng. WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization. NeurIPS 2025. [[Paper]](https://arxiv.org/abs/2508.09560) [[Code]](https://github.com/Jahawn-Wen/WeatherPrompt) |
|MuSe-Net | 37.28 | 43.17 | 33.55 | 39.79 | Wang T, Zheng Z, Sun Y, et al. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization[J]. Pattern Recognition, 2024. [[Code]](https://github.com/wtyhub/MuseNet) |
|LRFR | 41.53 | 48.86 | 41.89 | 49.81 | Gan, W., Zhou, Y., Hu, X., Zhao, L., Huang, G., & Hou, M. (2025). Learning Robust Feature Representation for Cross-View Image Geo-Localization. IEEE GRSL. |
|LPN | 44.67 | 45.77 | 43.39 | 49.31 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN) |
|GeoFuse | 52.43 | 58.39 | 49.03 | 54.96 | Yunsong Fang, Tingyu Wang, and Zhedong Zheng. Road Maps as Free Geometric Priors: Weather-Invariant Drone Geo-Localization with GeoFuse. arXiv 2026. [[Paper]](https://arxiv.org/abs/2605.14925) [[Code]](https://github.com/YsongF/GeoFuse) |


## SUES-200 Dataset (150 m)

[SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark) is a multi-height (150 / 200 / 250 / 300 m) multi-scene benchmark for cross-view matching between UAV and satellite imagery. Only the **retrieval** results at the 150 m height are listed below.

### (a) Standard Protocol (150 m)

Clear weather. Sorted by Drone -> Satellite R@1.

|Methods | R@1 | AP | R@1 | AP | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|| Drone -> Satellite | | Satellite -> Drone |  |
|LCM (ResNet-50) | 43.42 | 49.65 | 57.50 | 38.11 | Ding L, Zhou J, Meng L, Meng L, and Long Z. A Practical Cross-View Image Matching Method between UAV and Satellite for UAV-Based Geo-Localization. Remote Sensing, 13(1): 47, 2021. [[Paper]](https://www.mdpi.com/2072-4292/13/1/47/pdf) |
|SUES-200 Baseline | 59.32 | 64.93 | 82.50 | 58.95 | Zhu, R., Yin, L., Yang, M., Wu, F., Yang, Y., & Hu, W. (2023). SUES-200: A Multi-Height Multi-Scene Cross-View Image Benchmark Across Drone and Satellite. IEEE TCSVT. [[Paper]](https://arxiv.org/abs/2204.10704) [[Code]](https://github.com/Reza-Zhu/SUES-200-Benchmark) |
|LPN (block=4) | 61.58 | 67.23 | 83.75 | 66.78 | Tingyu W, Zhedong Z, Chenggang Y, and Yi Y. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN) |
|FSRA | 68.25 | 73.45 | 83.75 | 76.67 | Ming Dai, Jianhong Hu, Jiedong Zhuang, Enhui Zheng. A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization. TCSVT 2022. [[Paper]](https://arxiv.org/pdf/2201.09206.pdf) [[Code]](https://github.com/dmmm1997/fsra) |
|IFSs | 77.57 | 81.30 | 93.75 | 89.49 | Ge, F., Zhang, Y., Liu, Y., Wang, G., Coleman, S., Kerr, D., & Wang, L. (2024). Multibranch Joint Representation Learning Based on Information Fusion Strategy for Cross-View Geo-Localization. IEEE TGRS, 62: 1-16. |
|Safe-Net | 81.05 | 84.76 | 97.50 | 86.36 | Lin, J., Luo, Z., Lin, D., Li, S., & Zhong, Z. (2024). A Self-Adaptive Feature Extraction Method for Aerial-View Geo-Localization. IEEE TIP, 34: 126-139. |
|MCCG | 82.22 | 85.47 | 93.75 | 89.72 | Tianrui Shen, Yingmei Wei, Lai Kang, Shanshan Wan and Yee-Hong Yang. MCCG: A ConvNeXt-based Multiple-Classifier Method for Cross-view Geo-localization. TCSVT 2023. [[Code]](https://github.com/mode-str/crossview) |
|SDPL | 82.95 | 85.82 | 93.75 | 83.75 | Quan Chen, Tingyu Wang, Zihao Yang, Haoran Li, Rongfeng Lu and Yaoqi Sun. SDPL: Shifting-dense partition learning for UAV-view geo-localization. TCSVT 2024. [[Paper]](https://ieeexplore.ieee.org/document/10587023) [[Code]](https://github.com/C-water/SDPL_release) |
|CCR | 87.08 | 89.55 | 92.50 | 88.54 | Du, H., He, J., & Zhao, Y. (2024). CCR: A Counterfactual Causal Reasoning-Based Method for Cross-View Geo-Localization. IEEE TCSVT, 34(11): 11630-11643. |
|MFJR | 88.95 | 91.05 | 95.00 | 89.31 | Ge, F., Zhang, Y., Wang, L., Liu, W., Liu, Y., Coleman, S., & Kerr, D. (2024). Multi-level Feedback Joint Representation Learning Network Based on Adaptive Area Elimination for Cross-view Geo-localization. IEEE Transactions on Geoscience and Remote Sensing. |
|SRLN | 89.90 | 91.90 | 93.75 | 93.01 | Lv, H., Zhu, H., Zhu, R., Wu, F., Wang, C., Cai, M., & Zhang, K. (2024). Direction-Guided Multiscale Feature Fusion Network for Geo-Localization. IEEE TGRS, 62: 1-13. |
|SCOF | 90.75 | 92.32 | 95.00 | 89.72 | Fang, C., Gao, J., Han, P., Zhao, C., & Gao, B. (2025). SCOF: Supervised Contrastive Orthogonal Fusion for Robust Cross-View Geolocalization. IEEE TGRS, 63: 1-15. |
|UniABG (Unsupervised) | 92.40 | 93.95 | 98.75 | 91.54 | Chen, C., Chen, Q., Yang, B., & Zhang, X. (2026). UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization. AAAI 2026 (Oral). [[Paper]](https://arxiv.org/abs/2511.12054) [[Code]](https://github.com/chenqi142/UniABG) |
|Sample4Geo | 92.60 | 94.00 | 97.50 | 93.63 | Fabian Deuser, Konrad Habel, Norbert Oswald. Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation. ICCV 2023. [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.html) [[Code]](https://github.com/Skyy93/Sample4Geo) |
|CDM-Net | 93.78 | 95.16 | 95.25 | 92.24 | Zhou, Xin, Xuerong Yang, and Yanchun Zhang. "CDM-Net: A Framework for Cross-View Geo-Localization With Multimodal Data." TGRS 2025. |
|QDFL | 93.97 | 95.42 | 98.75 | 95.10 | Hu, S., Shi, Z., Jin, T., & Liu, Y. Query-Driven Feature Learning for Cross-View Geo-Localization. TGRS 2025. |
|Game4Loc | 94.62 | 95.59 | 93.75 | 93.06 | Ji, Y., He, B., Tan, Z., & Wu, L. (2025). Game4Loc: A UAV Geo-Localization Benchmark from Game Data. AAAI 2025: 3913-3921. |
|CAMP | 95.40 | 96.38 | 96.25 | 93.69 | Wu, Q., Wan, Y., Zheng, Z., Zhang, Y., Wang, G., & Zhao, Z. (2024). CAMP: A Cross-view Geo-localization Method Using Contrastive Attributes Mining and Position-aware Partitioning. TGRS 2024. [[Paper]](https://ieeexplore.ieee.org/abstract/document/10644040) [[Code]](https://github.com/Mabel0403/CAMP) |
|MEAN | 95.50 | 96.46 | 97.50 | 94.75 | Chen, Zhongwei, Zhao-Xu Yang, and Hai-Jun Rong. "Multi-level embedding and alignment network with consistency and invariance learning for cross-view geo-localization." TGRS 2025. |
|DAC | 96.80 | 97.54 | 97.50 | 94.06 | Xia, P., Wan, Y., Zheng, Z., Zhang, Y., & Deng, J. (2024). Enhancing cross-view geo-localization with domain alignment and scene consistency. TCSVT 2024. [[Paper]](https://ieeexplore.ieee.org/document/10636268) [[Code]](https://github.com/SummerpanKing/DAC) |
|(MGS)²-Net | 98.45 | 98.78 | 98.75 | 96.50 | Li, M., He, M., Li, C., Chen, C., Shao, X., & Meng, Z. (2026). (MGS)²-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization. [[Paper]](https://arxiv.org/abs/2602.10704) |
|BGG | 99.30 | 99.46 | 98.75 | 98.22 | Wang, W., Quan, D., Huyan, N., Wang, S., Li, Y., He, P., & Jiao, L. (2026). BGG: Bridging the Geometric Gap between Cross-View Images by Vision Foundation Model Adaptation for Geo-Localization. [[Paper]](https://arxiv.org/abs/2605.10345) |


### (b) Multi-weather Protocol (150 m)

Mean R@1 / AP over ten weather conditions, following the [multi-weather leaderboard](https://github.com/wtyhub/MuseNet/blob/master/State-of-the-art.md).

|Methods | R@1 | AP | R@1 | AP | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|| Drone -> Satellite | | Satellite -> Drone |  |
|Zheng et al. | 33.40 | 39.63 | 45.75 | 30.58 | Zhedong Zheng, Yunchao Wei, Yi Yang. University-1652: A Multi-view Multi-source Benchmark for Drone-based Geo-localization. ACM MM 2020. [[Paper]](https://dl.acm.org/doi/abs/10.1145/3394171.3413896) [[Code]](https://github.com/layumi/University1652-Baseline) |
|IBN-Net | 39.58 | 46.23 | 52.25 | 37.78 | Xingang Pan, Ping Luo, Jianping Shi, Xiaoou Tang. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV 2018. [[Paper]](https://arxiv.org/abs/1807.09441) |
|MuSe-Net | 41.59 | 48.53 | 53.38 | 39.20 | Wang T, Zheng Z, Sun Y, et al. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization[J]. Pattern Recognition, 2024. [[Code]](https://github.com/wtyhub/MuseNet) |
|WeatherPrompt | 62.52 | 63.26 | 80.73 | 66.12 | Jiahao Wen, Hang Yu, and Zhedong Zheng. WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization. NeurIPS 2025. [[Paper]](https://arxiv.org/abs/2508.09560) [[Code]](https://github.com/Jahawn-Wen/WeatherPrompt) |
|P2FCN | 78.64 | 82.44 | 91.50 | 82.49 | Zhao, Q., Zhou, J., Wang, T., Chen, Q., Lu, R., & Yan, C. (2025). P2FCN: Environment-Independent UAV-View Geo-Localization via Pixel-to-Feature Co-Enhancement. IEEE TGRS. |

## CVUSA

|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|Workman | - | - | - | 34.40 | Scott Workman, Richard Souvenir, and Nathan Jacobs. ICCV 2015. Wide-area image geolocalization with aerial reference imagery [[Paper]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Workman_Wide-Area_Image_Geolocalization_ICCV_2015_paper.pdf) |
|Zhai  | - | - | - | 43.20 | Menghua Zhai, Zachary Bessinger, Scott Workman, and Nathan Jacobs. CVPR 2017. Predicting ground-level scene layout from aerial imagery.[[Paper]](https://arxiv.org/abs/1612.02709) [[Code]](https://github.com/viibridges/crossnet) |
|Vo | - | - | - | 63.70 | Nam N Vo and James Hays. ECCV 2016. Localizing and orienting street views using overhead imagery| 
|CVM-Net | 18.80 | 44.42 | 57.47 | 91.54 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.html)| 
|Orientation** | 27.15 | 54.66 | 67.54 | 93.91 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/1903.12351) [[Code]](https://github.com/Liumouliu/OriCNN) |
|Siam-FCANet | - | - | - | 98.3 | Sudong C, Yulan G, Salman K, et al. Ground-to-Aerial Image Geo-Localization With a Hard Exemplar Reweighting Triplet Loss. ICCV 2019. [[Paper]](https://salman-h-khan.github.io/papers/ICCV19-3.pdf) |
|Feature Fusion | 48.75 | - | 81.27 | 95.98 | Krishna Regmi, Mubarak Shah, et al. Bridging the Domain Gap for Ground-to-Aerial Image Matching. ICCV 2019. [[Paper]](https://arxiv.org/abs/1904.11045) [[Code]](https://github.com/kregmi/cross-view-image-matching) |
|Instance Loss  | 43.91 | 66.38 | 74.58 | 91.78 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535) [[Code]](https://github.com/layumi/University1652-Baseline)|
|RK-Net (USAM) | 52.50 | - | - | 96.52 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zdzheng.xyz/files/TIP_RKNet.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|CVFT | 61.43 | 84.69 | 90.49 | 99.02 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2020. [[Paper]](https://arxiv.org/abs/1907.05021) [[Code]](https://github.com/shiyujiao/cross_view_localization_CVFT) |
|DWDR | 75.62 | 90.45 | 93.60 | 98.60 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" IEEE Transactions on Geoscience and Remote Sensing, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10744586) [[Code]](https://github.com/wtyhub/DWDR) |
|MS Attention w DataAug| 75.95 | 91.90 | 95.00 | 99.42 |Rodrigues, Royston, and Masahiro Tani. "Are These From the Same Place? Seeing the Unseen in Cross-View Image Geo-Localization." WACV 2021. [[Paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Rodrigues_Are_These_From_the_Same_Place_Seeing_the_Unseen_in_WACV_2021_paper.pdf)|
|MuSe-Net (Normal Weather) | 78.04 | - | - | - | Wang T, Zheng Z, Sun Y, et al. Multiple-environment Self-adaptive Network for Aerial-view Geo-localization[J]. Pattern Recognition, 2024. |
|LPN| 85.79 | 95.38 | 96.98 | 99.41 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|LPN + CA-HRS | 87.16 | 95.98 | 97.55 | 99.49 | Zeng Lu, Tao Pu, Tianshui Chen, and Liang Lin. Content-Aware Hierarchical Representation Selection for Cross-View Geo-Localization ACCV2022. [[Paper]](https://openaccess.thecvf.com/content/ACCV2022/papers/Lu_Content-Aware_Hierarchical_Representation_Selection_for_Cross-View_Geo-Localization_ACCV_2022_paper.pdf)  [[Code]](https://github.com/Allen-lz/CA-HRS) |
|SAFA* | 89.84 | 96.93 | 98.14 | 99.64 | Yujiao Shi, Liu Liu, Xin Yu, et al. Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization. NeurIPS 2019. [[Paper]](http://papers.neurips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization) [[Code]](https://github.com/shiyujiao/SAFA) |
|SAFA* + USAM | 90.16 | - | - | 99.67 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zdzheng.xyz/files/TIP_RKNet.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|LPN + USAM | 91.22 | - | - | 99.67 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zdzheng.xyz/files/TIP_RKNet.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|DSM*| 91.96 | 97.50 | 98.54 | 99.67 | Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li. "Where am i looking at? joint location and orientation estimation by cross-view matching." CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_CVPR_2020_paper.pdf) [[Code]](https://github.com/shiyujiao/cross_view_localization_DSM)| 
|Toker etal.* | 92.56 | 97.55 | 98.33 | 99.67 | Aysim Toker, Qunjie Zhou, Maxim Maximov, Laura Leal-Taixé. Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization. CVPR 2021 [[Paper]](https://arxiv.org/pdf/2103.06818.pdf) | 
|Shi etal.* | 92.69 | 97.78 | 98.60 | 99.61 | Yujiao Shi, Xin Yu,  Liu Liu, Dylan Campbell, Piotr Koniusz, and Hongdong Li. Accurate 3-DoF Camera Geo-Localization via Ground-to-Satellite Image Matching. TPAMI 2022. [[Paper]](https://arxiv.org/pdf/2203.14148.pdf) [[Code]](https://github.com/shiyujiao/ibl)|
|SAFA* + LPN | 92.83 | 98.00 | 98.85 | 99.78 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
| W2WBEV | 93.22 | 97.99 | 98.72 | 99.35 | Cheng L, Wang T, Meng L, et al. Window-to-Window BEV Representation Learning for Limited FoV Cross-View Geo-localization[J]. arXiv preprint arXiv:2407.06861, 2024 |
| S6G | 93.64	| 98.20	| 98.82	| 99.76 | Han, P., & Chen, C. (2025). An efficient cross-view image fusion method based on selected state space and hashing for promoting urban perception. Information Fusion, 115, 102737. |
| SIRNet* | 93.74 | 98.02 | 98.85 | 99.76 | Xiufan Lu, Siqi Luo, Yingying Zhu. "It’s Okay to Be Wrong: Cross-View Geo-Localization With Step-Adaptive Iterative Refinement" IEEE Transactions on Geoscience and Remote Sensing 2022 [[Paper]](https://ieeexplore.ieee.org/document/9913952/) |
|Polar-L2LTR* | 94.05 | 98.27 | 98.99 | 99.67 | Hongji Yang, Xiufan Lu, Yingying Zhu. Cross-view Geo-localization with Layer-to-Layer Transformer. NeurIPS 2021 [[Paper]](https://papers.nips.cc/paper/2021/file/f31b20466ae89669f9741e047487eb37-Paper.pdf) [[Code]](https://github.com/yanghongji2007/cross_view_localization_L2LTR)|
|TransGeo | 94.08 | 98.36 | 99.04 | 99.77 | Sijie Zhu, Mubarak Shah, Chen Chen. TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization. CVPR 2022 [[Paper]](https://arxiv.org/pdf/2204.00097.pdf) [[Code]](https://github.com/jeff-zilence/transgeo2022)|
| MGTL* | 94.11 | 98.30 | 99.03 | 99.74 | Jianwei Zhao, Qiang Zhai, Rui Huang, Hong Cheng. Mutual Generative Transformer Learning for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/2203.09135)|
| GeoDTR | 93.76 | 98.47 | 99.22 | 99.85 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| TransGeo + 4SCIG | 94.10 | 98.74 | 99.22| 99.81 | Li, J., Yang, C., Qi, B., Zhu, M., & Wu, N. (2024). 4SCIG: A four-branch framework to reduce the interference of sky area in cross-view image geo-localization. IEEE Transactions on Geoscience and Remote Sensing. |
| Dual-Transformer | 94.11 | 98.74 | - | 99.83 | Guan, F., Zhao, N., Wang, H., Fang, Z., Zhang, J., Yu, Y., ... & Huang, H. (2025). Dual-branch transformer framework with gradient-aware weighting feature alignment for robust cross-view geo-localization. Information Fusion, 103808. |
| LPN* + DWDR | 94.33 | 98.54 | 99.09 | 99.80 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" arXiv 2022. [[Paper]](https://arxiv.org/pdf/2211.05296.pdf) |
| STG | 95.31 | 98.75 | 99.25 | 99.81 | Liang, J., Bao, M., Dong, H., Xie, L., Liu, R. W., & Chen, N. (2025). DSTG: Distillation Swin Transformer for Cross-view Geo-localization. IEEE Transactions on Geoscience and Remote Sensing. |
| SVT* | 95.35 | 98.82 | 99.29 | 99.80 | Ahn, W. J., Park, S. Y., Pae, D. S., Choi, H. D., & Lim, M. T. (2024). Bridging viewpoints in cross-view geo-localization with Siamese vision transformer. IEEE Transactions on Geoscience and Remote Sensing. | 
| GeoDTR* | 95.43 | 98.86 | 99.34 | 99.86 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| FI* | 95.50 | - | - | - | Wenmiao Hu, Yichen Zhang, Yuxuan Liang, Yifang Yin, Anderi Georgecu, An Tran, Hannes Kruppa, See-Kiong Ng, Roger Zimmermann. Beyond Geo-localization: Fine-grained Orientation of Street-view Images by Cross-view Matching with Satellite Imagery. ACM MM 2022 [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548102) |
| VimGeo | 96.19 | 98.62 | 99.00 | 99.52 | Huang, J., Wu, M., Li, P., Wu, W., & Yu, R. (2025, August). VimGeo: Efficient Cross-View Geo-Localization with Vision Mamba Architecture. IJCAI 2025. [[Code]](https://github.com/VimGeoTeam/VimGeo) |
| SAIG-D*| 96.34 | 99.10 | 99.50 | 99.86 | Yingying Zhu, Hongji Yang, Yuxin Lu and Qiang Huang. Simple, Effective and General: A New Backbone for Cross-view Image Geo-localization. ArXiv 2023 [[Code]](https://github.com/yanghongji2007/SAIG) |
|GANet | 96.38 | 99.12 | 99.41 | 99.85 | Su, F., Zhou, Z., Zhang, H., & Zhang, H. (2026). Hierarchical feature alignment for cross-view geo-localization. Scientific Reports. |
| ConGeo | 96.6 | 98.9 | 99.2 | 99.7 | Mi, L., Xu, C., Castillo-Navarro, J., Montariol, S., Yang, W., Bosselut, A., & Tuia, D. (2025). Congeo: Robust cross-view geo-localization across ground view variations. In ECCV 2024 |
| FRGeo | 97.06 | 99.25 | 99.47 | 99.85 | Zhang, Qingwang, and Yingying Zhu. "Aligning Geometric Spatial Layout in Cross-View Geo-Localization via Feature Recombination." In AAAI 2024 |
| ArcGeo* | 98.33 | 97.47 | 99.48 | 99.67| Shugaev, M., Semenov, I., Ashley, K., Klaczynski, M., Cuntoor, N., Lee, M. W., & Jacobs, N. (2024). ArcGeo: Localizing Limited Field-of-View Images using Cross-view Matching. WACV 2024 |
| MFRGN | 98.67 | 99.57 | 99.71 | 99.85 | Wang, Y., Zhang, J., Wei, R., Gao, W., & Wang, Y. MFRGN: Multi-scale Feature Representation Generalization Network for Ground-to-Aerial Geo-localization. ACM MM2024. [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3664647.3681431) [[Code]](https://github.com/ytao-wang/MFRGN) | 
| Sample4Geo| 98.68 | 99.68 | 99.78 | 99.87 | Fabian Deuser, Konrad Habel, Norbert Oswald. Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation. ICCV 2023 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.html) [[Code]](https://github.com/Skyy93/Sample4Geo) |
| BEV | 98.71 | 99.70 | 99.78 | 99.86 | Ye, J., Lv, Z., Li, W., Yu, J., Yang, H., Zhong, H., & He, C. (2024). Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network. ECCV2024. [[Code]](https://github.com/yejy53/EP-BEV) |
|*: The method utilizes the polar transformation (assuming that all satellite images face north) as input. | |
 |** : The method utilizes the polar prior hint. |

## CVACT val 

|Methods | R@1 | R@5 | R@10 | R@Top1 | Reference |
| -------- | ----- | ---- | ---- |  ---- |  ---- |
|CVM-Net | 20.15 | 45.00 | 56.87 | 87.57 | Sixing Hu, Mengdan Feng, Rang MH Nguyen, and Gim Hee Lee. CVPR 2018. CVM-net:Cross-view matching network for image-based ground-to-aerial geo-localization. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/html/Hu_CVM-Net_Cross-View_Matching_CVPR_2018_paper.html)| 
|Instance Loss  | 31.20 | 53.64 | 63.00 | 85.27 | Zheng Z, Zheng L, Garrett M, et al. Dual-Path Convolutional Image-Text Embedding with Instance Loss. TOMM 2020. [[Paper]](https://arxiv.org/abs/1711.05535) [[Code]](https://github.com/layumi/University1652-Baseline) |
|RK-Net (USAM) | 40.53 | - | - | 89.12 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|Orientation** | 46.96 | 68.28 | 75.48 | 92.04 | Liu Liu and Hongdong Li. CVPR 2019. Lending Orientation to Neural Networks for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/1903.12351) [[Code]](https://github.com/Liumouliu/OriCNN) |
|CVFT | 61.05 | 81.33 | 86.52 | 95.93 | Shi Y, Yu X, Liu L, et al. Optimal Feature Transport for Cross-View Image Geo-Localization. AAAI 2020. [[Paper]](https://arxiv.org/abs/1907.05021) [[Code]](https://github.com/shiyujiao/cross_view_localization_CVFT) |
|DWDR | 66.76 | 83.34 | 87.11 | 95.10 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" IEEE Transactions on Geoscience and Remote Sensing, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10744586) [[Code]](https://github.com/wtyhub/DWDR) |
|MS Attention w DataAug| 73.19 | 90.39 | 93.38 | 97.45 |Rodrigues, Royston, and Masahiro Tani. "Are These From the Same Place? Seeing the Unseen in Cross-View Image Geo-Localization." WACV 2021. [[Paper]](https://openaccess.thecvf.com/content/WACV2021/papers/Rodrigues_Are_These_From_the_Same_Place_Seeing_the_Unseen_in_WACV_2021_paper.pdf)|
|LPN| 79.99 | 90.63 | 92.56 | 97.03 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|LPN + CA-HRS | 80.91 | 90.95 | 92.93 | 97.07 | Zeng Lu, Tao Pu, Tianshui Chen, and Liang Lin. Content-Aware Hierarchical Representation Selection for Cross-View Geo-Localization ACCV2022. [[Paper]](https://openaccess.thecvf.com/content/ACCV2022/papers/Lu_Content-Aware_Hierarchical_Representation_Selection_for_Cross-View_Geo-Localization_ACCV_2022_paper.pdf)  [[Code]](https://github.com/Allen-lz/CA-HRS) |
|LDRVSD| 80.98 | 91.48 | 93.33 | - | Qian Hu, Wansi Li, Xing Xu, Ning Liu, Lei Wang. Learning discriminative representations via variational self-distillation for cross-view geo-localization. Computers and Electrical Engineering 2022|
|SAFA* | 81.03 | 92.80 | 94.84 | 98.17 | Yujiao Shi, Liu Liu, Xin Yu, et al. Spatial-Aware Feature Aggregation for Cross-View Image based Geo-Localization. NeurIPS 2019. [[Paper]](http://papers.neurips.cc/paper/9199-spatial-aware-feature-aggregation-for-image-based-cross-view-geo-localization) [[Code]](https://github.com/shiyujiao/SAFA) |
|LPN + USAM | 82.02 | - | - | 98.18 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|SAFA* + USAM | 82.40 | - | - | 98.00 | Lin J, Zheng Z, Zhong Z, Luo Z, Li S, Yang Y, Sebe N. Joint Representation Learning and Keypoint Detection for Cross-view Geo-localization. TIP 2022. [[Paper]](https://zhunzhong.site/paper/RK_Net.pdf)  [[Code]](https://github.com/AggMan96/RK-Net) |
|DSM* | 82.49 | 92.44 | 93.99 | 97.32 | Yujiao Shi, Xin Yu, Dylan Campbell, and Hongdong Li. "Where am i looking at? joint location and orientation estimation by cross-view matching." CVPR 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Where_Am_I_Looking_At_Joint_Location_and_Orientation_Estimation_CVPR_2020_paper.pdf) [[Code]](https://github.com/shiyujiao/cross_view_localization_DSM) | 
|Shi etal.* | 82.70 | 92.50 | 94.24 | 97.65 | Yujiao Shi, Xin Yu,  Liu Liu, Dylan Campbell, Piotr Koniusz, and Hongdong Li. Accurate 3-DoF Camera Geo-Localization via Ground-to-Satellite Image Matching. TPAMI 2022. [[Paper]](https://arxiv.org/pdf/2203.14148.pdf) [[Code]](https://github.com/shiyujiao/ibl)|
| ConGeo | 83.0 | 90.6 | 92.4 | 96.3 | Mi, L., Xu, C., Castillo-Navarro, J., Montariol, S., Yang, W., Bosselut, A., & Tuia, D. (2025). Congeo: Robust cross-view geo-localization across ground view variations. In ECCV 2024 |
| W2WBEV | 83.09 | 91.20 | 92.98 | 96.45 | Cheng L, Wang T, Meng L, et al. Window-to-Window BEV Representation Learning for Limited FoV Cross-View Geo-localization[J]. arXiv preprint arXiv:2407.06861, 2024 | 
|Toker etal.* | 83.28 | 93.57 | 95.42 | 98.22 | Aysim Toker, Qunjie Zhou, Maxim Maximov, Laura Leal-Taixé. Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization. CVPR 2021 [[Paper]](https://arxiv.org/pdf/2103.06818.pdf) |
|SAFA* + LPN | 83.66 | 94.14 | 95.92 | 98.41 | Tingyu Wang, Zhedong Zheng, Chenggang Yan, and Yi, Yang. Each Part Matters: Local Patterns Facilitate Cross-view Geo-localization. TCSVT 2021. [[Paper]](https://arxiv.org/abs/2008.11646) [[Code]](https://github.com/wtyhub/LPN)|
|LPN* + DWDR | 83.73 | 92.78 | 94.53 | 97.78 | Tingyu W, Zhedong Z, Zunjie Z, Yuhan G, Yi Y, and Chenggang Y. "Learning Cross-view Geo-localization Embeddings via Dynamic Weighted Decorrelation Regularization" IEEE Transactions on Geoscience and Remote Sensing, 2024. [[Paper]](https://ieeexplore.ieee.org/document/10744586) [[Code]](https://github.com/wtyhub/DWDR) |
|Polar-L2LTR* | 84.89 | 94.59 | 95.96 | 98.37 | Hongji Yang, Xiufan Lu, Yingying Zhu. Cross-view Geo-localization with Layer-to-Layer Transformer. NeurIPS 2021 [[Paper]](https://papers.nips.cc/paper/2021/file/f31b20466ae89669f9741e047487eb37-Paper.pdf) [[Code]](https://github.com/yanghongji2007/cross_view_localization_L2LTR)|
| TransGeo | 84.95 | 94.14 | 95.78 | 98.37 | Sijie Zhu, Mubarak Shah, Chen Chen. TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization. CVPR 2022 [[Paper]](https://arxiv.org/pdf/2204.00097.pdf) [[Code]](https://github.com/jeff-zilence/transgeo2022)|
| TransGeo + 4SCIG | 83.73 | 94.41 | 95.79 | 98.41 | Li, J., Yang, C., Qi, B., Zhu, M., & Wu, N. (2024). 4SCIG: A four-branch framework to reduce the interference of sky area in cross-view image geo-localization. IEEE Transactions on Geoscience and Remote Sensing. |
| S6G | 	85.23	| 94.69 |	96.21 | 98.50 | Han, P., & Chen, C. (2025). An efficient cross-view image fusion method based on selected state space and hashing for promoting urban perception. Information Fusion, 115, 102737. | 
| MGTL* | 85.35 | 94.45 | 96.06 | 98.48 | Jianwei Zhao, Qiang Zhai, Rui Huang, Hong Cheng. Mutual Generative Transformer Learning for Cross-view Geo-localization [[Paper]](https://arxiv.org/abs/2203.09135)|
| SIRNet* | 86.02 | 94.45 | 96.02 | 98.33 | Xiufan Lu, Siqi Luo, Yingying Zhu. "It’s Okay to Be Wrong: Cross-View Geo-Localization With Step-Adaptive Iterative Refinement" IEEE Transactions on Geoscience and Remote Sensing 2022 [[Paper]](https://ieeexplore.ieee.org/document/9913952/)|
| GeoDTR | 85.43 | 94.81 | 96.11 | 98.26 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| GeoDTR* | 86.21 | 95.44 | 96.72 | 98.77 | Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah.  Cross-view Geo-localization via Learning Disentangled Geometric Layout Correspondence [[Paper]](https://arxiv.org/pdf/2212.04074.pdf) [[Code]](https://gitlab.com/vail-uvm/geodtr)|
| SVT* | 86.35 | 95.67 | 97.07 | 98.61  | Ahn, W. J., Park, S. Y., Pae, D. S., Choi, H. D., & Lim, M. T. (2024). Bridging viewpoints in cross-view geo-localization with Siamese vision transformer. IEEE Transactions on Geoscience and Remote Sensing. | 
| STG | 86.5 | 95.66 | 96.93 | 98.76 | Liang, J., Bao, M., Dong, H., Xie, L., Liu, R. W., & Chen, N. (2025). DSTG: Distillation Swin Transformer for Cross-view Geo-localization. IEEE Transactions on Geoscience and Remote Sensing. | 
| GANet | 87.52 | 96.00 | 97.05 | 98.80 | Su, F., Zhou, Z., Zhang, H., & Zhang, H. (2026). Hierarchical feature alignment for cross-view geo-localization. Scientific Reports. |
| FI* | 86.79 | - | - | - | Wenmiao Hu, Yichen Zhang, Yuxuan Liang, Yifang Yin, Anderi Georgecu, An Tran, Hannes Kruppa, See-Kiong Ng, Roger Zimmermann. Beyond Geo-localization: Fine-grained Orientation of Street-view Images by Cross-view Matching with Satellite Imagery. ACM MM 2022 [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3503161.3548102) |
| VimGeo | 87.62 | 94.88 | 96.06  | 98.06 | Huang, J., Wu, M., Li, P., Wu, W., & Yu, R. (2025, August). VimGeo: Efficient Cross-View Geo-Localization with Vision Mamba Architecture. IJCAI 2025. [[Code]](https://github.com/VimGeoTeam/VimGeo) |
|SAIG-D*| 89.06 | 96.11 | 97.08 | 98.89 | Yingying Zhu, Hongji Yang, Yuxin Lu and Qiang Huang. Simple, Effective and General: A New Backbone for Cross-view Image Geo-localization. ArXiv 2023 [[Code]](https://github.com/yanghongji2007/SAIG) |
| FRGeo | 90.35 | 96.45 | 97.25 | 98.74 | Zhang, Qingwang, and Yingying Zhu. "Aligning Geometric Spatial Layout in Cross-View Geo-Localization via Feature Recombination." In AAAI 2024 |
|Sample4Geo| 90.81 | 96.74 | 97.48 | 98.77 | Fabian Deuser, Konrad Habel, Norbert Oswald. Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation. ICCV 2023 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Deuser_Sample4Geo_Hard_Negative_Sampling_For_Cross-View_Geo-Localisation_ICCV_2023_paper.html) [[Code]](https://github.com/Skyy93/Sample4Geo) |
| ArcGeo* | 90.90 | 95.84 | 96.77 | - | Shugaev, M., Semenov, I., Ashley, K., Klaczynski, M., Cuntoor, N., Lee, M. W., & Jacobs, N. (2024). ArcGeo: Localizing Limited Field-of-View Images using Cross-view Matching. WACV 2024 | 
| MFRGN | 91.09 | 96.34 | 97.14 | 98.44 | Wang, Y., Zhang, J., Wei, R., Gao, W., & Wang, Y. MFRGN: Multi-scale Feature Representation Generalization Network for Ground-to-Aerial Geo-localization. ACM MM2024. [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3664647.3681431) [[Code]](https://github.com/ytao-wang/MFRGN) |
| BEV| 91.90 | 97.23 | 97.84 | 98.84 | Ye, J., Lv, Z., Li, W., Yu, J., Yang, H., Zhong, H., & He, C. (2024). Cross-view image geo-localization with Panorama-BEV Co-Retrieval Network. ECCV2024. [[Code]](https://github.com/yejy53/EP-BEV) |
|*: The method utilizes the polar transformation (assuming that all satellite images face north) as input. | |
 |** : The method utilizes the polar prior hint. |
