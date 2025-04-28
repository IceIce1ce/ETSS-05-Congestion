### Automation Lab, Sungkyunkwan University

This is the official repository of 

**OpenCounting: An open Source Implementation of Crowd Counting Methods.**

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

### 2.2. Video Crowd Counting Datasets
TBA.

## 3. Usage
### 3.1 Supported Models for Bayesian Crowd Counting
| Models        | UCF-QNRF           | ShanghaiTech       |
|---------------|--------------------|--------------------|
| BayesianCrowd | :heavy_check_mark: | :heavy_check_mark: |

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
| Models   | ShanghaiTech       | NWPU-Crowd         | UCF-QNRF           |
|----------|--------------------|--------------------|--------------------|
| DM-Count | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

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