import torch.nn as nn
if __name__ == '__main__':
    from decoder import SimpleDecoder
    from encoders import build_encoder
else:
    from .decoder import SimpleDecoder
    from .encoders import build_encoder

class Counter(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = build_encoder(encoder_name)
        self.decoders = SimpleDecoder(in_channel=self.encoder.last_channel,  fea_channel=self.encoder.last_channel, out_channel=1, up_sample=self.encoder.last_scale // 2)

    def forward(self, image): # [16, 3, 256, 256]
        fea = self.encoder(image)
        denmap = self.decoders(fea)
        if denmap.size(1) == 2:
            den1, den2 = denmap[:, :1], denmap[:, 1:]
            den = (den2 - den1).relu()
        else:
            den = denmap.relu()
        return den # [16, 1, 128, 128]