python generate_mask.py --type_dataset sha --input_dir datasets/ShanghaiTech/part_A --mask_ratio 10
python generate_mask.py --type_dataset shb --input_dir datasets/ShanghaiTech/part_B --mask_ratio 10
python make_json.py --type_dataset sha --input_dir datasets/ShanghaiTech/part_A --train_json A_train.json --val_json A_val.json
python make_json.py --type_dataset shb --input_dir datasets/ShanghaiTech/part_B --train_json B_train.json --val_json B_val.json
python process_dataset.py --input_dir datasets/ShanghaiTech