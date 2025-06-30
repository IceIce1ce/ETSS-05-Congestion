# DCCUS

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{du2022domain,
  title={Domain-general Crowd Counting in Unseen Scenarios},
  author={Du, Zhipeng and Deng, Jiankang and Shi, Miaojing},
  journal={arXiv preprint arXiv:2212.02573},
  year={2022}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To download weights, run the following script:
```shell
bash scripts/download_weights.sh
```

## 4. To train and test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/test_sta.sh
bash scripts/test_stb.sh
```

## 5. Acknowledgement
* [ZPDu/Domain-general-Crowd-Counting-in-Unseen-Scenarios](https://github.com/ZPDu/Domain-general-Crowd-Counting-in-Unseen-Scenarios)
