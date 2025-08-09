import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import copy
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import cv2
import skimage.measure
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import scipy
import scipy.io
plt.ion()
from myInception_segLoss import headCount_inceptionv3
from generate_density_map import generate_density_map
import argparse

IMG_EXTENSIONS = ['.JPG','.JPEG','.jpg', '.jpeg', '.PNG', '.png', '.ppm', '.bmp', '.pgm', '.tif']
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    d = os.path.join(dir,'images')
    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                image_path = os.path.join(root, fname)
                head,tail = os.path.split(root)
                label_path = os.path.join(head,'ground_truth', 'GT_' + fname[:-4] + '.mat')
                item = [image_path, label_path]
                images.append(item)
    return images

class ShanghaiTechDataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train', extensions=IMG_EXTENSIONS, patch_size=128, num_patches_per_image=4):
        self.samples = make_dataset(data_dir,extensions)
        self.transform = transform
        self.phase = phase
        self.patch_size = patch_size
        self.numPatches = num_patches_per_image

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):        
        img_file, label_file = self.samples[idx]
        image = cv2.imread(img_file) # [435, 1024, 3]
        annPoints = scipy.io.loadmat(label_file)
        annPoints = annPoints['image_info'][0][0][0][0][0] # [2153, 2]
        positions = generate_density_map(shape=image.shape, points=annPoints, f_sz=15, sigma=4) # [750, 1000]
        fbs = generate_density_map(shape=image.shape, points=annPoints, f_sz=25, sigma=1) # [750, 1000]
        fbs = np.int32(fbs > 0)
        targetSize = [self.patch_size, self.patch_size]
        height, width, channel = image.shape
        if height < targetSize[0] or width < targetSize[1]:
            image = cv2.resize(image,(np.maximum(targetSize[0] + 2,height), np.maximum(targetSize[1] + 2, width)))
            count = positions.sum()
            max_value = positions.max()
            positions = cv2.resize(positions, (np.maximum(targetSize[0] + 2, height), np.maximum(targetSize[1] + 2, width)))
            count2 = positions.sum()
            positions = np.minimum(positions * count / (count2 + 1e-8), max_value * 10)
            fbs = cv2.resize(fbs,(np.maximum(targetSize[0] + 2, height), np.maximum(targetSize[1] + 2,width)))
            fbs = np.int32(fbs > 0)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
            image = np.concatenate((image, image, image), axis=2)
        image = image.transpose(2, 0, 1)
        numPatches = self.numPatches
        if self.phase == 'train':
            patchSet, countSet, fbsSet = getRandomPatchesFromImage(image, positions, fbs, targetSize, numPatches)
            x = np.zeros((patchSet.shape[0], 3, targetSize[0], targetSize[1]))
            if self.transform:
              for i in range(patchSet.shape[0]):
                x[i, :, :, :] = self.transform(np.uint8(patchSet[i, :, :, :]).transpose(1, 2, 0))
            patchSet = x
        if self.phase == 'val' or self.phase == 'test':
            patchSet, countSet, fbsSet = getAllFromImage(image, positions, fbs)
            patchSet[0, :, :, :] = self.transform(np.uint8(patchSet[0, :, :, :]).transpose(1, 2, 0))
        return patchSet, countSet, fbsSet # [4, 3, 128, 128], [4, 1, 128, 128], [4, 1, 128, 128]

def getRandomPatchesFromImage(image, positions, fbs, target_size, numPatches): # [3, 681, 1024], [681, 1024], [681, 1024], [128, 128], 4
    imageShape = image.shape
    if np.random.random() > 0.5:
        for channel in range(3):
            image[channel, :, :] = np.fliplr(image[channel, :, :])
        positions = np.fliplr(positions)
        fbs = np.fliplr(fbs)
    patchSet = np.zeros((numPatches, 3, target_size[0], target_size[1]))
    countSet = np.zeros((numPatches, 1, target_size[0], target_size[1]))
    fbsSet = np.zeros((numPatches, 1, target_size[0], target_size[1]))
    for i in range(numPatches):
        topLeftX = np.random.randint(imageShape[1] - target_size[0] + 1)
        topLeftY = np.random.randint(imageShape[2] - target_size[1] + 1)
        thisPatch = image[:,topLeftX:topLeftX+target_size[0],topLeftY:topLeftY+target_size[1]]
        patchSet[i, :, :, :] = thisPatch
        position = positions[topLeftX:topLeftX + target_size[0], topLeftY:topLeftY + target_size[1]]
        fb = fbs[topLeftX:topLeftX + target_size[0], topLeftY:topLeftY + target_size[1]]
        position = position.reshape((1, position.shape[0], position.shape[1]))
        fb = fb.reshape((1, fb.shape[0], fb.shape[1]))
        countSet[i, :, :, :] = position
        fbsSet[i, :, :, :] = fb
    return patchSet, countSet, fbsSet # [4, 3, 128, 128], [4, 1, 128, 128], [4, 1, 128, 128]

def getAllFromImage(image, positions, fbs): # [3, 768, 1024], [768, 1024, [768, 1024]
    nchannel, height, width = image.shape
    patchSet = np.zeros((1, 3, height, width))
    patchSet[0, :, :, :] = image[:, :, :]
    countSet = positions.reshape((1, 1, positions.shape[0], positions.shape[1])) # [1, 1, 768, 1024]
    fbsSet = fbs.reshape((1, 1, fbs.shape[0], fbs.shape[1])) # [1, 1, 768, 1024]
    return patchSet, countSet, fbsSet

data_transforms = {'train': transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def train_model(model, optimizer, scheduler, num_epochs=100, seg_loss=False, cl_loss=False, test_step=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae_val = 1e6
    best_mae_by_val = 1e6
    best_mae_by_test = 1e6
    best_mse_by_val = 1e6
    best_mse_by_test = 1e6
    best_epoch_val = 0
    best_epoch_test = 0
    criterion1 = nn.MSELoss(reduce=False) # for density map loss
    criterion2 = nn.BCELoss() # for segmentation map loss
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for index, (inputs, labels, fbs) in enumerate(dataloaders['train']): # [6, 4, 3, 128, 128], [6, 4, 1, 128, 128], [6, 4, 1, 128, 128]
            labels = labels * 100
            labels = skimage.measure.block_reduce(labels.cpu().numpy(),(1, 1, 1, 4, 4), np.sum) # [6, 4, 1, 32, 32]
            fbs = skimage.measure.block_reduce(fbs.cpu().numpy(), (1, 1, 1, 4, 4), np.max) # [6, 4, 1, 32, 32]
            fbs = np.float32(fbs > 0)
            labels = torch.from_numpy(labels)
            fbs = torch.from_numpy(fbs)
            labels = labels.cuda()
            fbs = fbs.cuda()
            inputs = inputs.cuda()
            inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]) # [24, 3, 128, 128]
            labels = labels.view(-1, labels.shape[3], labels.shape[4]) # [24, 32, 32]
            fbs = fbs.view(-1, fbs.shape[3], fbs.shape[4]) # [24, 32, 32]
            inputs = inputs.float()
            labels = labels.float()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output, fbs_out = model(inputs) # [24, 32, 32], [24, 32, 32]
                loss_den = criterion1(output, labels)
                loss_seg = criterion2(fbs_out, fbs)
                if cl_loss:
                    th = 0.1 * epoch + 5
                else:
                    th = 1000
                weights = th / (F.relu(labels - th) + th)
                loss_den = loss_den * weights
                loss_den = loss_den.sum() / weights.sum()
                if seg_loss:
                    loss = loss_den + 20 * loss_seg
                else:
                    loss = loss_den
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()    
        epoch_loss = running_loss / dataset_sizes['train']            
        print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        if epoch % test_step == 0:
            _, epoch_mae, epoch_mse, epoch_mre = test_model(model, optimizer, 'val')
            _, epoch_mae_test, epoch_mse_test, epoch_mre_test = test_model(model, optimizer, 'test')
            if epoch_mae < best_mae_val:
                best_mae_val = epoch_mae
                best_mae_by_val = epoch_mae_test
                best_mse_by_val = epoch_mse_test
                best_epoch_val = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if epoch_mae_test < best_mae_by_test:
                best_mae_by_test = epoch_mae_test
                best_mse_by_test = epoch_mse_test
                best_epoch_test = epoch
            print('[val]: Epoch: {}, Best MAE: {:.2f}, Best MSE: {:.2f}'.format(best_epoch_val + 1, best_mae_by_val, best_mse_by_val))
            print('[test]: Epoch: {}, Best MAE: {:.2f}, Best MSE: {:.2f}'.format(best_epoch_test + 1, best_mae_by_test, best_mse_by_test))
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, optimizer, phase):
    model.eval()
    mae = 0
    mse = 0
    mre = 0
    pred = np.zeros((3000,2))
    for index, (inputs, labels, fbs) in enumerate(dataloaders[phase]):
        inputs = inputs.cuda() # [1, 1, 3, 680, 1024]
        labels = labels.cuda() # [1, 1, 1, 680, 1024]
        inputs = inputs.float()
        labels = labels.float()
        inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]) # [1, 3, 680, 1024]
        labels = labels.view(-1, labels.shape[3], labels.shape[4]) # [1, 680, 1024]
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs, fbs_out = model(inputs) # [1, 170, 256], [1, 170, 256]
            outputs = outputs.to(torch.device("cpu")).numpy() / 100
            pred_count = outputs.sum()
        true_count = labels.to(torch.device("cpu")).numpy().sum()
        mse = mse + np.square(pred_count - true_count)
        mae = mae + np.abs(pred_count - true_count)
        mre = mre + np.abs(pred_count - true_count) / true_count
        pred[index, 0] = pred_count
        pred[index, 1] = true_count
    pred = pred[0:index + 1, :]
    mse = np.sqrt(mse / (index + 1))
    mae = mae / (index + 1)
    mre = mre / (index + 1)
    print('[{}]: MAE: {:.2f}, MSE: {:.2f}, MRE: {:.2f}'.format(phase, mae, mse, mre))
    return pred, mae, mse, mre

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='sha')
    parser.add_argument('--input_dir', type=str, default='datasets/ShanghaiTech/part_A_final')
    parser.add_argument('--output_dir', type=str, default='saved_sha')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_patches_per_image', type=int, default=4)
    parser.add_argument('--test_step', type=int, default=1)
    parser.add_argument('--seg_loss', type=bool, default=True)
    parser.add_argument('--cl_loss', type=bool, default=True)
    args = parser.parse_args()

    print('Training dataset:', args.type_dataset)
    # train, val and test loader
    image_datasets = {x: ShanghaiTechDataset(args.input_dir + '/' + x + '_data', phase=x, transform=data_transforms[x], patch_size=args.patch_size,
                                             num_patches_per_image=args.num_patches_per_image) for x in ['train','test']}
    image_datasets['val'] = ShanghaiTechDataset(args.input_dir + '/train_data', phase='val', transform=data_transforms['val'], patch_size=args.patch_size,
                                                num_patches_per_image=args.num_patches_per_image)
    indices = list(range(len(image_datasets['train'])))
    split = np.int(len(image_datasets['train']) * 0.2)
    val_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = indices
    test_idx = range(len(image_datasets['test']))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetSampler(test_idx)
    train_loader = torch.utils.data.DataLoader(dataset=image_datasets['train'], batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=image_datasets['val'], batch_size=1, sampler=val_sampler, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=image_datasets['test'], batch_size=1, sampler=test_sampler, num_workers=args.num_workers)
    dataset_sizes = {'train': len(train_idx), 'val': len(val_idx), 'test': len(image_datasets['test'])}
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    # model
    model = headCount_inceptionv3(pretrained=True).cuda()
    # model = MCNN()
    # model = SANet()
    # model = TEDNet(use_bn=True)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    model = train_model(model, optimizer, exp_lr_scheduler, num_epochs=501, seg_loss=args.seg_loss, cl_loss=args.cl_loss, test_step=args.test_step)
    pred, mae, mse, mre = test_model(model, optimizer, 'test')
    scipy.io.savemat('results.mat', mdict={'pred': pred, 'mse': mse, 'mae': mae,'mre': mre})
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pt'))