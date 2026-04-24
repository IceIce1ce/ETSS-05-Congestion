import random
from torch.utils.data import Dataset
from image import load_data_masked_data

class listDataset(Dataset):
    def __init__(self, root, transform=None, train=False, args=None):
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.lines[index]
        if self.args.is_eval:
            img, target, mask, target_20, img_path = load_data_masked_data(img_path, self.train, self.args)
            if self.transform is not None:
                img = self.transform(img)
            return img, target, mask, target_20, img_path
        img, target, mask, target_20 = load_data_masked_data(img_path, self.train, self.args)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, mask, target_20