"""
The Unet based convolutional block. The code was written based on Medical Open Network for AI (MONAI).
- Ronneberger O, Fischer P, and Brox T 2015 U-Net: Convolutional networks for biomedical image segmentation in Proc. Int. Conf. Med. Image Comput. Comput.-Assisted Intervention 234–241
- Kerfoot E, Clough J, Oksuz I, Lee J, King A P, and Schnabel J A 2018 Left-ventricle quantification using residual U-Net in Proc. Int. Workshop Stat. Atlases Comput. Models Heart 371–380
- Cardoso M J et al 2022 MONAI: An open-source framework for deep learning in healthcare arXiv:2211.02701.
"""

from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = [
    "Unet_en",
    "Unet_de",
    "Unet_out",
    "Dual_en"
] 

class Unet_en(nn.Module):
    """
    Encoder for Unet
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        if self.num_res_units > 0:
            self.down = ResidualUnit(
                self.dimensions,
                self.in_channels,
                self.out_channels,
                strides=self.strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        else:
            self.down = Convolution(
                self.dimensions,
                self.in_channels,
                self.out_channels,
                strides=self.strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )

    def forward(self, enc):

        out = self.down(enc)

        return out


class Unet_de(nn.Module):
    """
    Decoder of Unet
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        is_top: bool = False,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_top = is_top

        self.conv_up = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=self.strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=self.is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        self.conv = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        self.ru = ResidualUnit(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=1,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=self.is_top,
            adn_ordering=self.adn_ordering,
        )

    def forward(self, dec, enc):

        up = self.conv_up(dec)
        skip = torch.cat((up, enc), dim=1)
        if self.num_res_units > 0:
            out = self.ru(skip)
        else:
            out = self.conv(skip)

        return out

class Unet_out(nn.Module):
    """
    Covolution before output
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        is_top: bool = False,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_top = is_top

        self.conv_up = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=self.strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=self.is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        self.conv = Convolution(
            self.dimensions,
            self.out_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        self.ru = ResidualUnit(
            self.dimensions,
            self.out_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=1,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=self.is_top,
            adn_ordering=self.adn_ordering,
        )

    def forward(self, dec):

        up = self.conv_up(dec)

        if self.num_res_units > 0:
            out = self.ru(up)
        else:
            out = self.conv(up)

        return out

class Dual_en(nn.Module): 
    """
    Convolution for fusion block
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        is_top: bool = False,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_top = is_top

        self.conv = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        self.ru = ResidualUnit(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=1,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=self.is_top,
            adn_ordering=self.adn_ordering,
        )

    def forward(self, br_1, br_2):

        skip = torch.cat((br_1, br_2), dim=1)
        if self.num_res_units > 0:
            out = self.ru(skip)
        else:
            out = self.conv(skip)

        return out

