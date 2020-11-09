from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from collections import OrderedDict
from typing import List, Callable
from functools import partial
from ....blocks import ConvBnAct
from ....models.VisionModule import VisionModule
from glasses.utils.Storage import ForwardModuleStorage


class UNetBasicBlock(nn.Sequential):
    """Basic Block for UNet. It is composed by a double 3x3 conv.
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module = partial(nn.ReLU, inplace=True), *args, **kwargs):
        super().__init__(ConvBnAct(in_features, out_features, kernel_size=3, activation=activation, *args, **kwargs),
                         ConvBnAct(
            out_features, out_features, kernel_size=3, activation=activation, *args, **kwargs))


DownBlock = UNetBasicBlock
UpBlock = UNetBasicBlock


class DownLayer(nn.Module):
    """UNet down layer (left side). 

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        donwsample (bool, optional): If true maxpoll will be used to reduce the resolution of the input. Defaults to True.
        block (nn.Module, optional): Block used. Defaults to DownBlock.

    """

    def __init__(self, in_features: int, out_features: int, donwsample: bool = True, block: nn.Module = DownBlock, *args, **kwargs):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2, stride=2) if donwsample else nn.Identity(),
            block(in_features, out_features, *args, **kwargs))

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class UpLayer(nn.Module):
    """UNet up layer (right side). 

    Args:
        out_features (int): Number of input features
        out_features (int): Number of output features
        block (nn.Module, optional): Block used. Defaults to UpBlock.

    """

    def __init__(self, in_features: int, out_features: int, lateral_features: int = None, block: nn.Module = UpBlock, *args, **kwargs):
        super().__init__()
        lateral_features = out_features if lateral_features is None else lateral_features
        self.up = nn.ConvTranspose2d(
            in_features, out_features, 2, 2)

        self.block = block(out_features + lateral_features,
                           out_features, *args, **kwargs)

    def forward(self, x: Tensor, res: Tensor) -> Tensor:
        x = self.up(x)
        if res is not None:
            x = torch.cat([res, x], dim=1)
        out = self.block(x)

        return out


class SegmentationEncoder(nn.Module):
    
    def __init__(self, backbone, stages : List[nn.Module] = None, widths : List[int] = None):
        super().__init__()
        self.backbone = backbone
        self.stages = backbone.layers if stages is None else stages  
        self.widths = backbone.widths if widths is None else widths
        self.storage = ForwardModuleStorage(self.backbone, [*self.stages])
        
    def forward(self, x):
        return self.backbone(x)
    
    @property
    def features(self):
        return list(self.storage.state.values())


class UNetEncoder(nn.Module):
    """UNet Encoder composed of several layers of convolutions aimed to increased the features space and decrease the resolution.
    """

    def __init__(self, in_channels: int,  widths: List[int] = [64, 128, 256, 512, 1024], *args, **kwargs):
        self.in_out_block_sizes = list(zip(widths, widths[1:]))
        self.widths = widths
        self.stem = nn.Identity()

        self.layers = nn.ModuleList([
            DownLayer(in_channels, widths[0],
                      donwsample=False, *args, **kwargs),
            *[DownLayer(in_features,
                        out_features, *args, **kwargs)
              for (in_features, out_features) in self.in_out_block_sizes]
        ])


    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        return x

class UNetDecoder(nn.Module):
    """
    UNet Decoder composed of several layer of upsampling layers aimed to decrease the features space and increase the resolution.
    """

    def __init__(self, widths: List[int] = [512, 256, 128, 64, 32], lateral_widths: List[int] = None, *args, **kwargs):
        super().__init__()
        self.widths = widths
        lateral_widths = widths if lateral_widths is None else lateral_widths
        lateral_widths.extend([0] * (len(widths) - len(lateral_widths)))
        print(lateral_widths)

        self.in_out_block_sizes = list(zip(widths, widths[1:]))
        self.layers = nn.ModuleList([
            UpLayer(in_features,
                    out_features, lateral_features, **kwargs)
            for (in_features, out_features), lateral_features in zip(self.in_out_block_sizes, lateral_widths)
        ])

class UNet(VisionModule):
    """Implementation of Unet proposed in `U-Net: Convolutional Networks for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_

    .. image:: https://github.com/FrancescoSaverioZuppichini/glasses/blob/develop/docs/_static/images/UNet.png?raw=true

    Examples:

       Create a default model

        >>> UNet()

        You can easily customize your model

        >>> # change activation
        >>> UNet(activation=nn.SELU)
        >>> # change number of classes (default is 2 )
        >>> UNet(n_classes=2)
        >>> # pass a different block
        >>> UNet(encoder=partial(UNetEncoder, block=SENetBasicBlock))
        >>> # change the encoder
        >>> unet = UNet(encoder=partial(ResNetEncoder, block=ResNetBottleneckBlock, depths=[2,2,2,2]))


    Args:

        in_channels (int, optional): [description]. Defaults to 1.
        n_classes (int, optional): [description]. Defaults to 2.
        encoder (nn.Module, optional): Model's encoder (left part). It have a `.stem` and `.block : nn.ModuleList` fields. Defaults to UNetEncoder.
        decoder (nn.Module, optional): Model's decoder (left part). It must have a `.layers : nn.ModuleList` field. Defaults to UNetDecoder.
        widths (List[int], optional): [description]. Defaults to [64, 128, 256, 512, 1024].
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 2,
                 encoder: nn.Module = UNetEncoder,
                 decoder: nn.Module = UNetDecoder, 
                  **kwargs):

        super().__init__()
        self.encoder = encoder(in_channels=in_channels, **kwargs)
        self.decoder = decoder(lateral_widths=self.encoder.widths[::-1], **kwargs)
        self.head = nn.Conv2d(
            self.decoder.widths[-1], n_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:       
        x = self.encoder(x)
        self.residuals = self.encoder.features[::-1]
        # reverse the residuals and remove the last one
        # if decoder has more layers than residuals, just pad residual with None
        self.residuals.extend(
            [None] * (len(self.decoder.layers) - len(self.residuals)))

        for layer, res in zip(self.decoder.layers, self.residuals):
            x = layer(x, res)

        x = self.head(x)
        return x
