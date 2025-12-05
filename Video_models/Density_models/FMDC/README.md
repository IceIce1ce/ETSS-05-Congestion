# FMDC

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{wan2024density,
  title={Density-Based Flow Mask Integration via Deformable Convolution for Video People Flux Estimation},
  author={Wan, Chang-Lin and Huang, Feng-Kai and Shuai, Hong-Han},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6573--6582},
  year={2024}
}
```

## 2. To install the environment, run the following script:
```shell
bash scripts/install.sh
```

## 3. To download the pretrained weight, run the following script:
```shell
bash scripts/download_weight.sh
```

## 4. To train and test the model for the CroHD dataset, run the following script:
```shell
bash scripts/train_crohd.sh
bash scripts/test_crohd.sh
```

## 5. Acknowledgement
* [LeoHuang0511/FMDC](https://github.com/LeoHuang0511/FMDC)
