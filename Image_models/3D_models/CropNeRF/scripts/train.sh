export NERFSTUDIO_METHOD_CONFIGS=fruit_nerf=fruit_nerf.fruit_nerf_config:fruit_nerf_method
export PYTHONPATH=$PYTHONPATH:/home/vsw/Desktop/CropNeRF-A-Neural-Radiance-Field-Based-Framework/crop_nerf/fruit_nerf
cd crop_nerf
python debug/train.py fruit_nerf --data 3DCotton/plant_1 --output-dir outputs
ns-export pointcloud --load-config outputs/plant_1/fruit_nerf/2026-02-21_231922/config.yml --output-dir outputs/plant_1/exports/pcd/ --num-points 10000000 --remove-outliers True --normal-method open3d --save-world-frame False --obb_center -0.0571471367 0.1105365818 -0.5400721172 --obb_rotation 0.0000000000 0.0000000000 0.0000000000 --obb_scale 1.0000000000 1.0000000000 1.0000000000
cd ..
