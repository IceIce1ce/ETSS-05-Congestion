import torch
import torch.nn as nn
from functools import partial
import sys
sys.path.append("ext/dinov2")
from dinov2.eval.linear import create_linear_input
from dinov2.eval.linear import LinearClassifier
from dinov2.eval.utils import ModelWithIntermediateLayers

class DinoRegression(nn.Module):
    def __init__(self, type, exp_config):
        super().__init__()
        self.config = exp_config
        self.device = exp_config["device"]
        model = torch.hub.load("facebookresearch/dinov2", type, pretrained=True).to(self.device)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
        self.feature_model = ModelWithIntermediateLayers(model, n_last_blocks=1, autocast_ctx=autocast_ctx).to(self.device)
        use_simple_linear = False
        if use_simple_linear:
            out_dim = 256 * 768
            self.regressor = nn.Linear(out_dim, 1).to(self.device)
            torch.nn.init.xavier_uniform_(self.regressor.weight)
        else:
            layers = [nn.Conv2d(self.config["MODEL_CONFIG"]["feats_dim"], 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                      nn.Conv2d(128, 64, kernel_size=2), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
            layers.append(nn.Linear(64, 1))
            layers.append(nn.Sigmoid())
            self.regressor = nn.Sequential(*layers).to(self.device)
        self.print_model_size()

    def forward(self, x, volume=None):
        with torch.no_grad():
            features = self.feature_model(x)
        feats = features[0][0].clone()
        feats = feats.permute(0, 2, 1)
        feats = feats.view((-1, self.config["MODEL_CONFIG"]["feats_dim"], self.config["MODEL_CONFIG"]["encoded_image_size"], self.config["MODEL_CONFIG"]["encoded_image_size"]))
        return self.regressor(feats)

    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total number of weights: {total_params}")
        regressor_params = sum(p.numel() for p in self.regressor.parameters())
        print(f"regressor number of weights: {regressor_params}")