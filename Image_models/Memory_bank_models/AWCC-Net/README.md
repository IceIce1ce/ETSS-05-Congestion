# AWCC-Net

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{huang2023counting,
  title={Counting Crowds in Bad Weather},
  author={Zhi-Kai Huang and Wei-Ting Chen and Yuan-Chun Chiang and Sy-Yen Kuo and Ming-Hsuan Yang},
  journal={ICCV},
  year={2023}
}
```

## 2. To download the weight, run the following script:
```shell
bash scripts/download_weight.sh
```

## 3. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 4. To test the model for the JHU-Crowd++ dataset, run the following script:
```shell
bash scripts/test_jhu.sh
```

## 5. Acknowledgement
* [awccnet/AWCC-Net](https://github.com/awccnet/AWCC-Net)
