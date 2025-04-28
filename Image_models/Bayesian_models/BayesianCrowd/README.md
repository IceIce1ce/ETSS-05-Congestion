# BayesianCrowd

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{ma2019bayesian,
  title={Bayesian loss for crowd count estimation with point supervision},
  author={Ma, Zhiheng and Wei, Xing and Hong, Xiaopeng and Gong, Yihong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={6142--6151},
  year={2019}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for the UCF-QNRF dataset, run the following scripts:
```shell
bash scripts/train_ucf_qnrf.sh
bash scripts/test_ucf_qnrf.sh
```

## 4. To train and test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/train_shanghaitech.sh
bash scripts/test_shanghaitech.sh
```

## 5. Acknowledgement
* [zhiheng-ma/Bayesian-Crowd-Counting](https://github.com/zhiheng-ma/Bayesian-Crowd-Counting)
