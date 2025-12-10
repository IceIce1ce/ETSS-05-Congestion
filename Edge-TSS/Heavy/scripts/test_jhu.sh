export NCCL_P2P_DISABLE=1
python test.py --type_dataset jhu --pre saved_jhu/model_best.pth --gpu_id 0,1 --num_queries 500 --output_dir saved_jhu