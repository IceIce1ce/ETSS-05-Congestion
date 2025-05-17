# APGCC

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{chen2024improving,
  title={Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance},
  author={Chen, I and Chen, Wei-Ting and Liu, Yu-Wei and Yang, Ming-Hsuan and Kuo, Sy-Yen},
  journal={arXiv preprint arXiv:2405.10589},
  year={2024}
}
```

## 2. To download pretrained weights, run the following script:
```shell
bash scripts/download_weights.sh
```

## 3. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 4. To train and test the model for ShanghaiTech and NWPU-Crowd datasets, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/train_nwpu.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
bash scripts/test_nwpu.sh
```

## 5. Acknowledgement
* [AaronCIH/APGCC](https://github.com/AaronCIH/APGCC)
