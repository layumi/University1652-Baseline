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
|Workman \cite{workman2015wide} | - | - | - | 34.40 |
|Zhai \cite{zhai2017predicting} | - | - | - | 43.20 |
|Vo \cite{vo2016localizing} | - | - | - | 63.70 |
|CVM-Net \cite{hu2018cvm} | 18.80 | 44.42 | 57.47 | 91.54 |
|Orientation \cite{liu2019lending}$^\dagger$ | 27.15 | 54.66 | 67.54 | 93.91 |
|Ours  | 43.91 | 66.38 | 74.58 | 91.78 |
