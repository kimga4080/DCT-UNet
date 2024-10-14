"""
DCTUnet
"""

import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch.nn as nn


from monai.networks.layers.factories import Act, Norm
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from einops import rearrange

from block_transformer import *
from block_convolution import *

class DCTUNet(nn.Module):
    """
    DCTUNet with convolution and transformer blocks
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (16, 32, 64),
        strides: Sequence[int]=(2, 2, 2),
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        bias: bool = True,
        adn_ordering: str = "NDA",
        depths: Sequence[int] = (2, 2, 2, 2),
        self_atts: Sequence[str] = ["Local", "Local", "Local", "Local"],
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        drop_rate_conv: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        patch_size: Sequence[int] = (2, 2, 2),
        window_size: Sequence[int] = (7, 7, 7),
    ) -> None:
        
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            drop_rate: dropout rate.
            channels: sequence of channels for unet. Top block first. The length of `channels` should be no less than 2.
            strides: sequence of convolution strides for unet. The length of `stride` should equal to `len(channels) - 1`.
            kernel_size: convolution kernel size, the value(s) should be odd. If sequence, its length should equal to dimensions. Defaults to 3.
            up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence, its length should equal to dimensions. Defaults to 3.
            num_res_units: int = 0,
            act: Union[Tuple, str] = Act.PRELU,
            norm: Union[Tuple, str] = Norm.INSTANCE,
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims) 
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size): 
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        if not len(depths)+1 == len(channels):
            raise ValueError("len(channels) and len(depths)+1 should be same")

        self.num_layers = len(channels)
        self.kernel_size = kernel_size

        self.normalize = normalize

        self.res_block = True

        self.swinViT = SwinTransformer_V2(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            self_atts=self_atts,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.InstanceNorm3d,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )
        
        self.encoder0 = Unet_en(
                spatial_dims = spatial_dims,
                in_channels_layer = in_channels,
                out_channels_layer = channels[0],
                strides = strides[0],
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
        )
        
        self.encoders = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            encoder = Unet_en(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[i_layer],
                out_channels_layer = channels[i_layer+1],
                strides = strides[i_layer+1],
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
            )
            self.encoders.append(encoder)
        
        
        self.endecoder_1 = Unet_en(
            spatial_dims = spatial_dims,
            in_channels_layer = channels[-1],
            out_channels_layer = channels[-1]*2,
            strides = 1,
            kernel_size = kernel_size,
            num_res_units = num_res_units,
            act = act,
            norm = norm,
            dropout = drop_rate_conv,
            bias = bias,
            adn_ordering = adn_ordering,
        )

        self.dualencoders = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dualencoder = Dual_en(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[i_layer]+feature_size*(2**i_layer),
                out_channels_layer = channels[i_layer],
                strides = 1,
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
            )
            self.dualencoders.append(dualencoder)

        self.endecoder_2 = Unet_de(
            spatial_dims = spatial_dims,
            in_channels_layer = channels[-1]*2,
            out_channels_layer = channels[-1],
            strides = 1,
            up_kernel_size = up_kernel_size,
            kernel_size = kernel_size,
            num_res_units = num_res_units,
            act = act,
            norm = norm,
            dropout = drop_rate_conv,
            bias = bias,
            adn_ordering = adn_ordering,
            is_top = False,
        )

        self.decoders = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            decoder = Unet_de(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[-(i_layer+1)],
                out_channels_layer = channels[-(i_layer+2)],
                strides = strides[-(i_layer+1)],
                up_kernel_size = up_kernel_size,
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
                is_top = False,
            )
            self.decoders.append(decoder)

        self.out = Unet_out(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[0],
                out_channels_layer = out_channels,
                strides = strides[0],
                up_kernel_size = up_kernel_size,
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
                is_top = True,
            )

    def forward(self, x_in):

        # Encoder with transformer block
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # Encoder with convolution block
        encs = []
        enc = self.encoder0(x_in) 
        encs.append(enc)
        for i_layer in range(self.num_layers - 1):
            enc = self.encoders[i_layer](encs[i_layer])
            encs.append(enc)

        # Last layer
        ende_1 = self.endecoder_1(encs[-1])

        # Fusion block
        dual_encs = []
        for i_layer in range(self.num_layers):
            dual_enc = self.dualencoders[i_layer](encs[i_layer], hidden_states_out[i_layer])
            dual_encs.append(dual_enc)

        # Decoder 
        decs = []
        dec = self.endecoder_2(ende_1, dual_encs[-1]) 
        decs.append(dec)
        for i_layer in range(self.num_layers - 1):
            dec = self.decoders[i_layer](decs[i_layer], dual_encs[-(i_layer+2)])
            decs.append(dec)

        out = self.out(decs[-1])

        return out
