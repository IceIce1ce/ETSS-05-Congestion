# SASNet

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{sasnet,
  title={To Choose or to Fuse? Scale Selection for Crowd Counting},
  author={Qingyu Song and Changan Wang and Yabiao Wang and Ying Tai and Chengjie Wang and Jilin Li and Jian Wu and Jiayi Ma},
  journal={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
  year={2021}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/test_sha.sh
bash scripts/test_shb.sh
```

## 4. Acknowledgement
* [TencentYoutuResearch/CrowdCounting-SASNet](https://github.com/TencentYoutuResearch/CrowdCounting-SASNet)
