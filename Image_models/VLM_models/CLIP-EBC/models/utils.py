import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, Any, List, TypeVar, List
from types import FunctionType
from itertools import repeat
import warnings
import os
from collections.abc import Iterable

V = TypeVar("V")
curr_dir = os.path.dirname(os.path.abspath(__file__))
vgg_urls = {"vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth", "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth", "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth", "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"}
vgg_cfgs = {"A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512], "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512], "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]}

def _log_api_usage_once(obj: Any) -> None:
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")

def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(repeat(x, n))

class ConvNormActivation(torch.nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]] = 3, stride: Union[int, Tuple[int, ...]] = 1, padding: Optional[Union[int, Tuple[int, ...], str]] = None,
                 groups: int = 1, norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 dilation: Union[int, Tuple[int, ...]] = 1, inplace: Optional[bool] = True, bias: Optional[bool] = None, conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None
        layers = [conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        _log_api_usage_once(self)
        self.out_channels = out_channels
        if self.__class__ == ConvNormActivation:
            warnings.warn("Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead.")

class Conv2dNormActivation(ConvNormActivation):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]] = 3, stride: Union[int, Tuple[int, int]] = 1, padding: Optional[Union[int, Tuple[int, int], str]] = None,
                 groups: int = 1, norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 dilation: Union[int, Tuple[int, int]] = 1, inplace: Optional[bool] = True, bias: Optional[bool] = None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, norm_layer, activation_layer, dilation, inplace, bias, torch.nn.Conv2d)

class MLP(torch.nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: List[int], norm_layer: Optional[Callable[..., torch.nn.Module]] = None, activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, bias: bool = True, dropout: float = 0.0):
        params = {} if inplace is None else {"inplace": inplace}
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))
        super().__init__(*layers)
        _log_api_usage_once(self)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, base_width: int = 64, dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, **kwargs: Any) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.stride = stride
        if in_channels != out_channels:
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, base_width: int = 64, dilation: int = 1, expansion: int = 4,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, **kwargs: Any) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        self.expansion = expansion
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if in_channels != out_channels:
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels), nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

def _init_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

class Upsample(nn.Module):
    def __init__(self, size: Union[int, Tuple[int, int]] = None, scale_factor: Union[float, Tuple[float, float]] = None, mode: str = "nearest", align_corners: bool = False, antialias: bool = False) -> None:
        super().__init__()
        self.interpolate = partial(F.interpolate, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, antialias=antialias)

    def forward(self, x: Tensor) -> Tensor:
        return self.interpolate(x)

def make_vgg_layers(cfg: List[Union[str, int]], in_channels: int = 3, batch_norm: bool = False, dilation: int = 1) -> nn.Sequential:
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "U":
            layers += [Upsample(scale_factor=2, mode="bilinear")]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_resnet_layers(block: Union[BasicBlock, Bottleneck], cfg: List[Union[int, str]], in_channels: int, dilation: int = 1, expansion: int = 1) -> nn.Sequential:
    layers = []
    for v in cfg:
        if v == "U":
            layers.append(Upsample(scale_factor=2, mode="bilinear"))
        else:
            layers.append(block(in_channels=in_channels, out_channels=v, dilation=dilation, expansion=expansion))
            in_channels = v
    layers = nn.Sequential(*layers)
    layers.apply(_init_weights)
    return layers