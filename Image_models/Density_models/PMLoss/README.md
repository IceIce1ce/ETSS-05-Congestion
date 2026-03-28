# PMLoss

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{linproximal,
  title={Proximal mapping loss: Understanding loss functions in crowd counting \& localization},
  author={Lin, Wei and Wan, Jia and Chan, Antoni B},
  booktitle={The Thirteenth International Conference on Learning Representations}
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
```

## 4. Acknowledgement
* [Elin24/pml](https://github.com/Elin24/pml)
