import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from .crowd_dataset import CrowdDataset

def loading_data(args):
    transform = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_set = CrowdDataset(root_path=args.input_dir, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False, drop_last=False)
    return test_loader