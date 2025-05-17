mkdir -p weights
cd weights
gdown https://drive.google.com/u/0/uc?id=1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu
wget https://download.pytorch.org/models/vgg16-397923af.pth
wget https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
cd ..