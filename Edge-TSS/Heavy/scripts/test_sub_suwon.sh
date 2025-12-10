export NCCL_P2P_DISABLE=1
python test.py --type_dataset sub_suwon --pre saved_sub_suwon/model_best.pth --gpu_id 0,1 --num_queries 700 --output_dir saved_sub_suwon
# python test.py --type_dataset sub_suwon --pre saved_sub_suwon/model_best.pth --gpu_id 0,1 --num_queries 700 --output_dir saved_sub_suwon --threshold 0.875