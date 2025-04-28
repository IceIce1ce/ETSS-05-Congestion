# CSCA

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{zhang2022spatio,
  title={Spatio-channel attention blocks for cross-modal crowd counting},
  author={Zhang, Youjia and Choi, Soyun and Hong, Sungeun},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={90--107},
  year={2022}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for the RGBT-CC dataset, run the following scripts:
```shell
bash scripts/train_rgbtcc.sh
bash scripts/test_rgbtcc.sh
```

## 4. To train and test the model for the ShanghaiTechRGBD dataset, run the following scripts:
```shell
bash scripts/train_shanghaitechrgbd.sh
bash scripts/test_shanghaitechrgbd.sh
```

## 5. Acknowledgement
* [AIM-SKKU/CSCA](https://github.com/AIM-SKKU/CSCA)
* [svip-lab/RGBD-Counting](https://github.com/svip-lab/RGBD-Counting)
