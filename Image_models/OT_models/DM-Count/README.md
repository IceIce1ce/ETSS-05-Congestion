# DM-Count

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{wang2020DMCount,
  title={Distribution Matching for Crowd Counting},
  author={Boyu Wang and Huidong Liu and Dimitris Samaras and Minh Hoai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020},
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for ShanghaiTech, UCF-QNRF, and NWPU-Crowd datasets, run the following scripts:
```shell
bash scripts/train_sha.sh
bash scripts/train_shb.sh
bash scripts/train_qnrf.sh
bash scripts/train_nwpu.sh
bash scripts/test_sha.sh
bash scripts/test_shb.sh
bash scripts/test_qnrf.sh
bash scripts/test_nwpu.sh
```

## 4. To demo the model for an image, run the following script:
```shell
bash scripts/demo.sh
```

## 5. Acknowledgement
* [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)
