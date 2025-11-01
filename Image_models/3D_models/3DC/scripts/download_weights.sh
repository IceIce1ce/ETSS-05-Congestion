mkdir -p weights
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P weights
mkdir ext
cd ext
git clone https://github.com/facebookresearch/dinov2
# cd dinov2
# git checkout b48308a394a04ccb9c4dd3a1f0a4daa1ce0579b8 
pip install fvcore omegaconf
# cd ..
git clone https://github.com/DepthAnything/Depth-Anything-V2/
mv Depth-Anything-V2 DepthAnythingV2