export NERFSTUDIO_METHOD_CONFIGS=fruit_nerf=fruit_nerf.fruit_nerf_config:fruit_nerf_method
export PYTHONPATH=$PYTHONPATH:/home/vsw/Desktop/CropNeRF-A-Neural-Radiance-Field-Based-Framework/crop_nerf/fruit_nerf
cd crop_nerf
python segmentation/segmenter.py plant_1
python fruit_nerf/scripts/semantic_projection.py pointcloud --load-config outputs/plant_1/fruit_nerf/2026-02-21_231922/config.yml --output-dir outputs/exports/pcd/ # error code for creating full super_cluster_idx
python segmentation/merger.py --base_dir outputs --recording_name plant_1 --super_cluster_idx 0
cd ..
