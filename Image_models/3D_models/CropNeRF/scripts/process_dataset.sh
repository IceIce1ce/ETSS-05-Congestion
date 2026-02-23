export NERFSTUDIO_METHOD_CONFIGS=fruit_nerf=fruit_nerf.fruit_nerf_config:fruit_nerf_method
export PYTHONPATH=$PYTHONPATH:/home/vsw/Desktop/CropNeRF-A-Neural-Radiance-Field-Based-Framework/crop_nerf/fruit_nerf
cd crop_nerf
python fruit_nerf/utils/convert_segmentation_img_to_label.py 'plant_1'
cd ..
