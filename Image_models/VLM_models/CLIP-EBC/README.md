# CLIP-EBC

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{ma2024clip,
title={CLIP-EBC: CLIP Can Count Accurately through Enhanced Blockwise Classification},
author={Ma, Yiming and Sanchez, Victor and Guha, Tanaya},
journal={arXiv preprint arXiv:2403.09281},
year={2024}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for ShanghaiTech, NWPU-Crowd, and UCF-QNRF datasets, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/train_nwpu.sh
bash scripts/train_qnrf.sh
bash scripts/test_nwpu.sh
```

## 4. Acknowledgement
* [Yiming-M/CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC)
