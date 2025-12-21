# MovingDroneCrowd

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{MDC,
  title={Video Individual Counting for Moving Drones},
  author={Fan, Yaowu and Wan, Jia and Han, Tao and Chan, Antoni B and Ma, Andy J},
  booktitle={ICCV},
  year={2025}
}
```

## 2. To process the dataset, run the following script:
```shell
bash scripts/process_dataset.sh
```

## 3. To train and test the model for MovingDroneCrowd and UAVVIC datasets, run the following scripts:
```shell
bash scripts/train_drone.sh
bash scripts/train_uavvic.sh
bash scripts/test_drone.sh
bash scripts/test_uavvic.sh
```

## 4. Acknowledgement
* [fyw1999/MovingDroneCrowd](https://github.com/fyw1999/MovingDroneCrowd)
