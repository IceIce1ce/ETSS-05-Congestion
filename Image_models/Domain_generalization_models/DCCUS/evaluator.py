import torch

def evaluate_all(model, data_loader):
    model.eval()
    mseloss = torch.nn.MSELoss()
    MAEs = 0.
    MSEs = 0.
    val_loss = []
    with torch.no_grad():
        for i, (imgs, gts) in enumerate(data_loader):
            imgs = imgs.cuda() # [1, 3, 1280, 1920]
            gts = gts.cuda() # [1, 1280, 1920]
            dens = model(imgs) # [1, 1, 1280, 1920]
            for j, (den, gt) in enumerate(zip(dens, gts)):
                loss = mseloss(den, gt)
                val_loss.append(loss.item())
                den = torch.sum(den) / 1000.
                gt = torch.sum(gt) / 1000.
                MAEs += abs(gt - den)
                MSEs += ((gt-den)*(gt-den))
    mae = MAEs / len(data_loader)
    mse = torch.sqrt(MSEs / len(data_loader))
    loss = torch.mean(torch.Tensor(val_loss))
    print('Testing loss: {:.4f}, MAE: {:.4f}, MSE: {:.4f}'.format(loss, mae, mse))
    return mae, mse

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader):
        return evaluate_all(self.model, data_loader)