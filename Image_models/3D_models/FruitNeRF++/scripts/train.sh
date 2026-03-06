export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1
export DATA_PATH='datasets/FruitNeRF_Real/tree_01'
export RESULT_PATH="results"
ns-train cf-nerf-small --data $DATA_PATH --output-dir $RESULT_PATH --viewer.camera-frustum-scale 0.2 --pipeline.model.temperature 0.1 --vis tensorboard
