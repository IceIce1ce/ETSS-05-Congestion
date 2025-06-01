# HMoDE

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{du2023redesigning,
  title={Redesigning multi-scale neural network for crowd counting},
  author={Du, Zhipeng and Shi, Miaojing and Deng, Jiankang and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for the ShanghaiTech dataset, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
```

## 4. Acknowledgement
* [ZPDu/Redesigning-Multi-Scale-Neural-Network-for-Crowd-Counting](https://github.com/ZPDu/Redesigning-Multi-Scale-Neural-Network-for-Crowd-Counting)
