from .vgg import VGGFPN
from .hrnet import HRNetFPN
from .utils import conv_3x3

def build_encoder(name):
    name = name.lower()
    if name.startswith("vgg"):
        return VGGFPN(name)
    elif name.startswith("hrnet"):
        return HRNetFPN(name)
    else:
        print('This encoder does not exist')
        raise NotImplementedError