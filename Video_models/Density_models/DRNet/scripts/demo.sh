export CUDA_VISIBLE_DEVICES=0
python demo/video2img.py --input_dir demo.mp4 --output_dir data/demo/img1
python demo/test_beijng.py --type_dataset HT21 --input_dir data --output_dir saved_demo --ckpt_dir saved_ht21/HT21.pth
python demo/test_CroHD.py --type_dataset HT21 --input_dir data/HT21/train --output_dir saved_ht21_demo --ckpt_dir saved_ht21/HT21.pth