export NCCL_P2P_DISABLE=1
python test.py --type_dataset trancos --pre saved_trancos/model_best.pth --gpu_id 0,1 --num_queries 700 --output_dir saved_trancos