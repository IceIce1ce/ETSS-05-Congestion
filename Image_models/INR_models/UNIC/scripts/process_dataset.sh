# process sha dataset
python dataset_preprocess.py --type_dataset sha --input_dir data/ShanghaiTech/part_A --output_dir data/sha
python generate_density_maps.py --type_dataset sha --input_dir data/sha
python generate_bayesian_gt.py --type_dataset sha --input_dir data/sha
# process shb dataset
python dataset_preprocess.py --type_dataset shb --input_dir data/ShanghaiTech/part_B --output_dir data/shb
python generate_density_maps.py --type_dataset shb --input_dir data/shb
python generate_bayesian_gt.py --type_dataset shb --input_dir data/shb