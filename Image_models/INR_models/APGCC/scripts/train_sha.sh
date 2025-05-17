# train with basic decoder
# python main.py --config_file configs/SHHA_basic.yml DATASETS.DATA_ROOT 'data/sha/part_A' TEST.THRESHOLD 0.5 OUTPUT_DIR 'saved_sha'
# train with IFI decodewr
python main.py --config_file configs/SHHA_IFI.yml DATASETS.DATA_ROOT 'data/sha/part_A' TEST.THRESHOLD 0.5 MODEL.DECODER 'IFI' OUTPUT_DIR 'saved_sha'