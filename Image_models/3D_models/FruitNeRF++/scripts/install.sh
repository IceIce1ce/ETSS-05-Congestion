pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install pip==24.0 cython ninja setuptools==65.0
# pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.4#subdirectory=bindings/torch
pip install nerfstudio==1.1.5
pip install pyntcloud==0.3.1 hdbscan numba hausdorff
conda install docutils -y
pip install -e .
ns-install-cli
mkdir -p cf_nerf/segmentation
cd cf_nerf/segmentation 
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git groundedSAM
cd groundedSAM
git checkout fe24
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/local/cuda-11.7/
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install diffusers[torch]==0.30
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
pip install segment-anything-hq
cd ../../..
cd cf_nerf/segmentation 
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd Detic
pip install -r requirements.txt
cd ../../..
