python data/prepare_jhu.py --input_dir data/jhu_crowd_v2.0
python data/prepare_nwpu.py --input_dir data/NWPU_CLTR
python data/prepare_trancos.py --type_dataset TRANCOS --input_dir data/TRANCOS
python data/process_drone_vehicle.py --type_dataset DroneVehicle --input_dir data/DroneVehicle
python data/prepare_trancos.py --type_dataset DroneVehicle --input_dir data/DroneVehicle
python data/prepare_suwon.py --type_dataset Suwon --input_dir data/Suwon
python data/prepare_suwon.py --type_dataset Sub_Suwon --input_dir data/Sub_Suwon
python data/make_npydata.py --jhu_path data/jhu_crowd_v2.0 --nwpu_path data/NWPU_CLTR --trancos_path data/TRANCOS --drone_vehicle_path data/DroneVehicle --suwon_path data/Suwon --output_dir npydata --sub_suwon_path data/Sub_Suwon