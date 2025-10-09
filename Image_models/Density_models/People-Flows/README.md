# People-Flows

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@InProceedings{Liu_2020_ECCV,
author = {Liu, Weizhe and Salzmann, Mathieu and Fua, Pascal},
title = {Estimating People Flows to Better Count Them in Crowded Scenes},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {August},
year = {2020}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To download the weight, run the following script:
```shell
bash scripts/download_weight.sh
```

## 4. To train, test, and visualize the model for the FDST dataset, run the following scripts:
```shell
bash scripts/train_fdst.sh
bash scripts/test_fdst.sh
bash scripts/vis_fdst.sh
```

## 5. Acknowledgement
* [weizheliu/People-Flows](https://github.com/weizheliu/People-Flows)
