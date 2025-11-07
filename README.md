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
For the RGBT-CC dataset, please download it from this [link](https://lingboliu.com/RGBT_Crowd_Counting.html).

For the ShanghaiTech RGB-D dataset, please download it from this [repo](https://github.com/svip-lab/RGBD-Counting).

For the UCF-QNRF dataset, please download it from this [link](https://www.crcv.ucf.edu/data/ucf-qnrf/)

For the NWPU-Crowd dataset, please download it from this [link](https://gjy3035.github.io/NWPU-Crowd-Sample-Code/)

For the ShanghaiTech dataset, please download it from this [link](https://www.kaggle.com/datasets/tthien/shanghaitech/data)

For the STCrowd dataset, please download it from this [link](https://opendatalab.com/OpenDataLab/STCrowd)

For the NWPU-MOC dataset, please download it from this [repository](https://github.com/lyongo/NWPU-MOC)

For the Towards-vs-Away dataset, please download it from this [repository](https://github.com/rk620/Fine-Grained-CrowdCounting)

For the CARPK dataset, please download it from this [link](https://lafi.github.io/LPN/)

For the FSC147 dataset, please download it from this [link](https://www.kaggle.com/datasets/xuncngng/fsc147-0)

For the FruitNeRF dataset, please download it from this [link](https://zenodo.org/records/10869455)

For the StackCounting dataset, please download it from this [link](https://zenodo.org/records/15609540)

For the TRANCOS dataset, please download it from this [link](https://gram.web.uah.es/data/datasets/trancos/index.html)

For the PUCPR dataset, please download it from this [link](https://lafi.github.io/LPN/)

For the Crowd-SR dataset, please download it from this [repository](https://github.com/PRIS-CV/MSSRM)

For the Mall dataset, please download it from this [repository](https://github.com/fyw1999/LCSD)

### 2.2. Video Crowd Counting Datasets
For the FDST dataset, please download it from this [repository](https://github.com/sweetyy83/Lstn_fdst_dataset)

For the VSCrowd dataset, please download it from this [link](https://huggingface.co/datasets/HopLeeTop/VSCrowd)

For the CroHD dataset, please download it from this [link](https://motchallenge.net/data/Head_Tracking_21/)

For the CARLA dataset, please download it from this [repository](https://github.com/LeoHuang0511/FMDC)

For the MovingDroneCrowd dataset, please download it from this [repository](https://github.com/fyw1999/MovingDroneCrowd)

For the DroneBird dataset, please download it from this [repository](https://github.com/mast1ren/E-MAC)

## 3. Usage
### 3.1 Supported Models for Bayesian Crowd Counting
| Models        | UCF-QNRF           | ShanghaiTech       |
|---------------|--------------------|--------------------|
| BayesianCrowd | :heavy_check_mark: | :heavy_check_mark: |
| NoisyCC       | :heavy_check_mark: | :x:                |

### 3.1 Supported Models for Multimodal Crowd Counting
| Models  | RGBT-CC            | ShanghaiTechRGBD   |
|---------|--------------------|--------------------|
| CSCA    | :heavy_check_mark: | :heavy_check_mark: |
| IADM    | :heavy_check_mark: | :heavy_check_mark: |
| EAEFNet | :heavy_check_mark: | :x:                |

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
| Models       | ShanghaiTech       | FDST               | UCF-QNRF           | STCrowd            | CARPK | Towards-vs-Away    | Mall               | JHU-Crowd++        | NWPU-Crowd         | TRANCOS            | Crowd-SR           |
|--------------|--------------------|--------------------|--------------------|--------------------|-------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CSRNet       | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| People-Flows | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| S-DCNet      | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SS-DCNet     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GCC-SFCN     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CACC         | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SASNet       | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PAL          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CUT          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SGANet       | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RankBench    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| STCrowd      | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FGCC         | :x:                | :x:                | :x:                | :x:                | :x:   | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| P2PNet       | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UEPNet       | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FIDTM        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| PML          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| AutoScale    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| IIM          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:   | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| DPD          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MSSRGN       | :x:                | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| PMLoss       | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |

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
| Models     | JHU-Crowd++        | NWPU               | ShanghaiTech       | UCF-QNRF           | CARPK              | NWPU-MOC           | TRANCOS            |
|------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| CLTR       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| TransCrowd | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                |
| PET        | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                |
| NWPU-MOC   | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |

### 3.9 Supported Models for MoE Crowd Counting
| Models | ShanghaiTech       |
|--------|--------------------|
| HMoDE  | :heavy_check_mark: |

### 3.10 Supported Models for Knowledge Distillation Crowd Counting
| Models  | ShanghaiTech       | UCF-QNRF           | Mall               |
|---------|--------------------|--------------------|--------------------|
| SKT     | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| LCSD    | :x:                | :x:                | :heavy_check_mark: |
| P2RLoss | :heavy_check_mark: | :x:                | :x:                |

### 3.11 Supported Models for Domain Adaptation Crowd Counting
| Models | UCF-QNRF           | Shanghaitech       | CARPK              | PUCPR              |
|--------|--------------------|--------------------|--------------------|--------------------|
| UGSDA  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| CODA   | :x:                | :heavy_check_mark: | :x:                | :x:                |
| CBD    | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: |

### 3.12 Supported Models for GCN Crowd Counting
| Models     | UCF-QNRF           | JHU-Crowd++        | ShanghaiTech       |
|------------|--------------------|--------------------|--------------------|
| Gramformer | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| GAAL       | :x:                | :x:                | :heavy_check_mark: |
| MDGCN      | :x:                | :x:                | :x:                |
| DSGCNet    | :x:                | :x:                | :heavy_check_mark: |

### 3.13 Supported Models for GCN Video Crowd Counting
| Models | FDST               |
|--------|--------------------|
| STGN   | :heavy_check_mark: |

### 3.14 Supported Models for Open-world Object Counting
| Models | ShanghaiTech       | FSC147             |
|--------|--------------------|--------------------|
| OVID   | :heavy_check_mark: | :heavy_check_mark: |

### 3.15 Supported Models for Density Video Crowd Counting
| Models           | VSCrowd            | CroHD              | FDST               | CARLA              | MovingDroneCrowd   | DroneBird          |
|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| VSCrowd          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| DAANet           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                |
| AVCC             | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| OMAN             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| FMDC             | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                |
| CGNet            | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| DRNet            | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| MovingDroneCrowd | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| DroneBird        | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.16 Supported Models for Low-shot Crowd Counting
| Models | FSC147             |
|--------|--------------------|
| FamNet | :heavy_check_mark: |

### 3.17 Supported Models for 3D Crowd Counting
| Models    | FruitNeRF          | StackCounting      |
|-----------|--------------------|--------------------|
| FruitNeRF | :heavy_check_mark: | :x:                |
| 3DC       | :x:                | :heavy_check_mark: |

## 4. Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {OpenCrowd: An Open Source Implementation of Crowd Counting Methods},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/SKKU-AutoLab-VSW/ETSS-05-CongestionDetection},
  year         = {2025}
}
```

## 5. Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

## 6. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.