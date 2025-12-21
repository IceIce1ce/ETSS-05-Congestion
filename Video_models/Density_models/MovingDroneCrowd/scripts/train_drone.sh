# train with single GPU
# python train.py --type_dataset MovingDroneCrowd --output_dir saved_drone
# train with multiple GPU
torchrun --master_port 29515 --nproc_per_node=2 train.py --type_dataset MovingDroneCrowd --output_dir saved_drone