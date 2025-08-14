### Automation Lab, Sungkyunkwan University

This is the official repository of 

**OpenCounting: An Open Source Implementation of Crowd Counting Methods.**

## 1. Setup
### 1.1. Using environment.yml
```bash
conda env create -f environment.yml
conda activate anomaly
```

### 1.2. Using requirements.txt
```bash
conda create --name anomaly python=3.10.13
conda activate anomaly
pip install -r requirements.txt
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

## 2. Dataset Preparation
### 2.1. Image Crowd Counting Datasets
For RGBT-CC dataset, please download it from this [link](https://lingboliu.com/RGBT_Crowd_Counting.html).

For ShanghaiTech RGB-D dataset, please download it from this [repo](https://github.com/svip-lab/RGBD-Counting).

For UCF-QNRF dataset, please download it from this [link](https://www.crcv.ucf.edu/data/ucf-qnrf/)

For NWPU-Crowd dataset, please download it from this [link](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)

For ShanghaiTech dataset, please download it from this [link](https://www.kaggle.com/datasets/tthien/shanghaitech/data)

### 2.2. Video Crowd Counting Datasets
For FDST dataset, please download it from this [repository](https://github.com/sweetyy83/Lstn_fdst_dataset)

## 3. Usage
### 3.1 Supported Models for Bayesian Crowd Counting
| Models        | UCF-QNRF           | ShanghaiTech       |
|---------------|--------------------|--------------------|
| BayesianCrowd | :heavy_check_mark: | :heavy_check_mark: |
| NoisyCC       | :heavy_check_mark: | :x:                |

### 3.1 Supported Models for Multimodal Crowd Counting
| Models | RGBT-CC            | ShanghaiTechRGBD   |
|--------|--------------------|--------------------|
| CSCA   | :heavy_check_mark: | :heavy_check_mark: |
| IADM   | :heavy_check_mark: | :heavy_check_mark: |

### 3.2 Supported Models for VLM Crowd Counting
| Models   | ShanghaiTech       | NWPU-Crowd         | UCF-QNRF           |
|----------|--------------------|--------------------|--------------------|
| CLIP-EBC | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.3 Supported Models for OT Crowd Counting
| Models          | ShanghaiTech       | NWPU-Crowd         | UCF-QNRF           | Arbitrary Image    |
|-----------------|--------------------|--------------------|--------------------|--------------------|
| DM-Count        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| OT-M            | :x:                | :x:                | :x:                | :heavy_check_mark: |
| GeneralizedLoss | :x:                | :x:                | :heavy_check_mark: | :x:                |

### 3.4 Supported Models for INR Crowd Counting
| Models | ShanghaiTech       | NWPU-Crowd         |
|--------|--------------------|--------------------|
| APGCC  | :heavy_check_mark: | :heavy_check_mark: |
| UNIC   | :heavy_check_mark: | :x:                |
| SI-INR | :heavy_check_mark: | :x:                |

### 3.5 Supported Models for Density Crowd Counting
| Models       | ShanghaiTech       | FDST               | UCF-QNRF           |
|--------------|--------------------|--------------------|--------------------|
| CSRNet       | :heavy_check_mark: | :x:                | :x:                |
| People-Flows | :x:                | :heavy_check_mark: | :x:                |
| S-DCNet      | :heavy_check_mark: | :x:                | :x:                |
| SS-DCNet     | :heavy_check_mark: | :x:                | :heavy_check_mark: |
| GCC-SFCN     | :x:                | :x:                | :heavy_check_mark: |
| CACC         | :heavy_check_mark: | :x:                | :x:                |
| SASNet       | :heavy_check_mark: | :x:                | :x:                |
| PAL          | :heavy_check_mark: | :x:                | :x:                |
| CUT          | :heavy_check_mark: | :x:                | :x:                |
| SGANet       | :heavy_check_mark: | :x:                | :x:                |
| RankBench    | :heavy_check_mark: | :x:                | :heavy_check_mark: |

### 3.6 Supported Models for Domain Generalization Crowd Counting
| Models  | ShanghaiTech       | UCF-QNRF           |
|---------|--------------------|--------------------|
| MPCount | :heavy_check_mark: | :heavy_check_mark: |
| DCCUS   | :heavy_check_mark: | :x:                |
| BLA     | :heavy_check_mark: | :x:                |

### 3.7 Supported Models for Video Crowd Analysis
| Models   | SDD                | IND-TIME           | FDST               | VSCROWD            | JRDB               | HT21               | ETHUCY             |
|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CrowdMAC | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.8 Supported Models for Memory Bank Crowd Counting
| Models   | JHU-Crowd++        |
|----------|--------------------|
| AWCC-Net | :heavy_check_mark: |

### 3.9 Supported Models for Transformer Crowd Counting
| Models     | JHU-Crowd++        | NWPU               | ShanghaiTech       | QNRF               |
|------------|--------------------|--------------------|--------------------|--------------------|
| CLTR       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| TransCrowd | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |

### 3.9 Supported Models for MoE Crowd Counting
| Models | ShanghaiTech       |
|--------|--------------------|
| HMoDE  | :heavy_check_mark: |

### 3.10 Supported Models for Knowledge Distillation Crowd Counting
| Models | ShanghaiTech       | UCF-QNRF           |
|--------|--------------------|--------------------|
| SKT    | :heavy_check_mark: | :heavy_check_mark: |

### 3.11 Supported Models for Domain Adaptation Crowd Counting
| Models | UCF-QNRF           | Shanghaitech       |
|--------|--------------------|--------------------|
| UGSDA  | :heavy_check_mark: | :x:                |
| CODA   | :x:                | :heavy_check_mark: |

### 3.12 Supported Models for GCN Crowd Counting
| Models     | UCF-QNRF           | JHU-Crowd++        | ShanghaiTech       |
|------------|--------------------|--------------------|--------------------|
| Gramformer | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| GAAL       | :x:                | :x:                | :heavy_check_mark: |
| MDGCN      | :x:                | :x:                | :x:                |

### 3.13 Supported Models for GCN Video Crowd Counting
| Models | FDST               |
|--------|--------------------|
| STGN   | :heavy_check_mark: |

### 3.14 Supported Models for Open-world Object Counting
| Models | ShanghaiTech       | FSC147             |
|--------|--------------------|--------------------|
| OVID   | :heavy_check_mark: | :heavy_check_mark: |

## 4. Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {OpenCrowd: An Open Source Implementation of Crowd Counting Methods},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/SKKU-AutoLab-VSW/ETSS-07-CongestionDetection},
  year         = {2023}
}
```

## 5. Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

## 6. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.