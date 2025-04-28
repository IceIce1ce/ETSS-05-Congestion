import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', default='qnrf', choices=['qnrf', 'nwpu'])
    parser.add_argument('--input_dir', default='data/UCF-QNRF', type=str)
    parser.add_argument('--output_dir', default='data/qnrf')
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    if args.type_dataset.lower() == 'qnrf':
        from preprocess.preprocess_dataset_qnrf import main
        main(args.input_dir, args.output_dir, 512, 2048)
    elif args.type_dataset.lower() == 'nwpu':
        from preprocess.preprocess_dataset_nwpu import main
        main(args.input_dir, args.output_dir, 384, 1920)
    else:
        print('This dataset does not exist')
        raise NotImplementedError
