python process/fidt_generate_sh.py --type_dataset ShanghaiTech --input_dir datasets/ShanghaiTech
python process/fidt_generate_jhu.py --type_dataset JHU-Crowd++ --input_dir datasets/jhu_crowd_v2.0
python process/fidt_generate_qnrf.py --type_dataset UCF-QNRF --input_dir datasets/UCF-QNRF
python process/fidt_generate_nwpu.py --type_dataset NWPU-Crowd --input_dir datasets/NWPU_CLTR
python process/process_trancos.py --type_dataset TRANCOS --input_dir datasets/TRANCOS
python process/fidt_generate_trancos.py --type_dataset TRANCOS --input_dir datasets/TRANCOS
python process/process_trancos.py --type_dataset Suwon --input_dir datasets/Suwon
python process/fidt_generate_suwon.py --type_dataset Suwon --input_dir datasets/Suwon
python process/make_npydata.py --sh_dir datasets/ShanghaiTech --qnrf_dir datasets/UCF-QNRF --jhu_dir datasets/jhu_crowd_v2.0 --nwpu_dir datasets/NWPU_CLTR --trancos_dir datasets/TRANCOS --suwon_dir datasets/Suwon --output_dir npydata