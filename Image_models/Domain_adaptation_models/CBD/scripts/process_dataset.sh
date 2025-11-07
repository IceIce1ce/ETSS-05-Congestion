python convert_bbox_to_point.py --type_dataset CARPK --input_dir datasets/CARPK/Annotations/ --output_dir datasets/CARPK/VGG_annotation_truth/
python convert_bbox_to_point.py --type_dataset PUCPR --input_dir datasets/PUCPR/Annotations/ --output_dir datasets/PUCPR/VGG_annotation_truth/
python process_dataset.py --type_dataset CARPK --input_dir datasets/CARPK
python process_dataset.py --type_dataset PUCPR --input_dir datasets/PUCPR