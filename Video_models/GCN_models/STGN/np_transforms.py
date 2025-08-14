import torchvision.transforms.functional as TF

class ToTensor(object):
    def __call__(self, sample):
        return [TF.to_tensor(image) for image in sample]