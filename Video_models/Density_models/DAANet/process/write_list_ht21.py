import os

def divide_dataset(Root='data/HT21'):
    test_set = []
    val_set = []
    train_set = []
    train_path = os.path.join(Root + '/train')
    scenes = os.listdir(train_path)
    for i_scene in scenes:
        train_set.append(os.path.join('train/' + i_scene))
    train_path = os.path.join(Root + '/test')
    scenes = os.listdir(train_path)
    for i_scene in scenes:
        test_set.append(os.path.join('test/' + i_scene))
    train_set = set(train_set)
    val_set = set(val_set)
    train_set = train_set - val_set
    train_set = sorted(train_set)
    val_set = sorted(val_set)
    test_set = sorted(test_set)
    print('Number of training images:', len(train_set))
    print('Number of validating images:', len(val_set))
    print('Number of testing images:', len(test_set))
    with open(os.path.join(Root, 'train.txt'), "w") as f:
        for train_name in train_set:
            f.write(train_name + '\n')
    f.close()
    with open(os.path.join(Root, 'val.txt'), "w") as f:
        for valid_name in val_set:
            f.write(valid_name + '\n')
    f.close()
    with open(os.path.join(Root, 'test.txt'), "w") as f:
        for test_name in test_set:
            f.write(test_name + '\n')
    f.close()

if __name__ == '__main__':
    divide_dataset()