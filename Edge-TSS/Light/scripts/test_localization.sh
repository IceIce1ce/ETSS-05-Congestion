cd local_eval
python eval.py --type_dataset sha
python eval.py --type_dataset shb
python eval.py --type_dataset jhu
python eval.py --type_dataset nwpu
python eval_qnrf.py --type_dataset qnrf --input_dir ../datasets/UCF-QNRF
cd ..