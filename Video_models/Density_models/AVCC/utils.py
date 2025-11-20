import os
import torch

def save_checkpoint(state, is_best, epoch, args):
    if is_best:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        filename = os.path.join(args.output_dir, 'best_' + str(epoch + 1) + '.pth.tar')
        torch.save(state, filename)