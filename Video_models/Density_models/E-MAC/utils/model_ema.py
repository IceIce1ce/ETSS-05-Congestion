from collections import OrderedDict
from copy import deepcopy
import torch

class ModelEma:
    def __init__(self, model, decay=0.9999, device="", resume=""):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, "module")
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert isinstance(checkpoint, dict)
        if "state_dict_ema" in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict_ema"].items():
                if self.ema_has_module:
                    name = "module." + k if not k.startswith("module") else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)
            print("Loaded state_dict_ema")
        else:
            print("Failed to find state_dict_ema, starting from loaded model weights")

    def update(self, model):
        needs_module = hasattr(model, "module") and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = "module." + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1.0 - self.decay) * model_v)