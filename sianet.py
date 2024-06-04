VERBOSE=False
import copy
import itertools
from typing import Sequence, Union, Tuple, Optional
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes,  kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=0.01)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=0.01)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Learnable_Filter(nn.Module):
    """Refinement module of MagNet
    Args:
        n_classes (int): no. classes
        use_bn (bool): use batch normalization on the input
    """

    def __init__(self, n_classes=1):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64, momentum=0.01)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        # 2 residual blocks
        self.residual = self._make_layer(Bottleneck, 64, 32, 2)
        #self.weight_mask_conv = self._make_layer(BasicBlock, 1, 64, 1)

        # Prediction head
        self.seg_conv = nn.Conv3d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make residual block"""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion, momentum=0.01),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)# + self.weight_mask_conv(weight_mask)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.residual(x)

        return self.seg_conv(x)

class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, 1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv3d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model=1, d_model2=64):
        super().__init__()

        self.proj_1 = nn.Conv3d(d_model, d_model2, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model2)
        self.proj_2 = nn.Conv3d(d_model2, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

def get_conv(dim=3):
    """Chooses an implementation for a convolution layer."""
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_convtranspose(dim=3):
    """Chooses an implementation for a transposed convolution layer."""
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_maxpool(dim=3):
    """Chooses an implementation for a max-pooling layer."""
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_normalization(normtype: str, num_channels: int, dim: int = 3):
    """Chooses an implementation for a batch normalization layer."""
    if normtype is None or normtype == 'none':
        return nn.Identity()
    elif normtype.startswith('group'):
        if normtype == 'group':
            num_groups = 8
        elif len(normtype) > len('group') and normtype[len('group'):].isdigit():
            num_groups = int(normtype[len('group'):])
        else:
            raise ValueError(
                f'normtype "{normtype}" not understood. It should be "group<G>",'
                f' where <G> is the number of groups.'
            )
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normtype == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    elif normtype == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
        else:
            raise ValueError('dim has to be 2 or 3')
    else:
        raise ValueError(
            f'Unknown normalization type "{normtype}".\n'
            'Valid choices are "batch", "instance", "group" or "group<G>",'
            'where <G> is the number of groups.'
        )


def planar_kernel(x):
    """Returns a "planar" kernel shape (e.g. for 2D convolution in 3D space)
    that doesn't consider the first spatial dim (D)."""
    if isinstance(x, int):
        return (1, x, x)
    else:
        return x


def planar_pad(x):
    """Returns a "planar" padding shape that doesn't pad along the first spatial dim (D)."""
    if isinstance(x, int):
        return (0, x, x)
    else:
        return x


def conv3(in_channels, out_channels, kernel_size=3, stride=1,
          padding=1, bias=True, planar=False, dim=3):
    """Returns an appropriate spatial convolution layer, depending on args.
    - dim=2: Conv2d with 3x3 kernel
    - dim=3 and planar=False: Conv3d with 3x3x3 kernel
    - dim=3 and planar=True: Conv3d with 1x3x3 kernel
    """
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim)(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )


def upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3):
    """Returns a learned upsampling operator depending on args."""
    kernel_size = 2
    stride = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
    if mode == 'transpose':
        return get_convtranspose(dim)(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    elif 'resizeconv' in mode:
        if 'linear' in mode:
            upsampling_mode = 'trilinear' if dim == 3 else 'bilinear'
        else:
            upsampling_mode = 'nearest'
        rc_kernel_size = 1 if mode.endswith('1') else 3
        return ResizeConv(
            in_channels, out_channels, planar=planar, dim=dim,
            upsampling_mode=upsampling_mode, kernel_size=rc_kernel_size
        )


def conv1(in_channels, out_channels, dim=3):
    """Returns a 1x1 or 1x1x1 convolution, depending on dim"""
    return get_conv(dim)(in_channels, out_channels, kernel_size=1)


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'silu':
            return nn.SiLU()
        elif activation == 'lin':
            return nn.Identity()
    else:
        # Deep copy is necessary in case of paremtrized activations
        return copy.deepcopy(activation)

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, dropout_rate, pooling=True, planar=False, activation='relu',
                 normalization=None, full_norm=True, dim=3, conv_mode='same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.normalization = normalization
        self.dim = dim
        padding = 1 if 'same' in conv_mode else 0

        #self.conv1 = conv3(
        #    self.in_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        #)

        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=True)
        self.conv1_2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=True)

        #self.conv2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=True)
        #self.conv2_1 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=True)


        self.conv2 = conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        )

        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = get_maxpool(dim)(kernel_size=kernel_size, ceil_mode=True)
            self.pool_ks = kernel_size
        else:
            self.pool = nn.Identity()
            self.pool_ks = -123  # Bogus value, will never be read. Only to satisfy TorchScript's static type system

        self.dropout = nn.Dropout3d(dropout_rate)

        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)

        self.norm01 = get_normalization(normalization, self.out_channels, dim=dim)
        self.norm02 = get_normalization(normalization, self.out_channels, dim=dim)

        #self.norm02 = get_normalization(normalization, int(self.out_channels/2), dim=dim)
        #self.norm03 = get_normalization(normalization, int(self.out_channels/2), dim=dim)

        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
            if VERBOSE: print("DownConv, full_norm, norm0 =",normalization)
        else:
            self.norm0 = nn.Identity()
            if VERBOSE: print("DownConv, no full_norm")
        self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
        if VERBOSE: print("DownConv, norm1 =",normalization)

    def forward(self, x):
        #
        y = self.conv1(x)
        y = self.norm01(y)
        y = self.act1(self.dropout(y))
        y = self.conv1_2(y)
        #
        y = self.conv2(y)
        y = self.norm1(y)
        y =  self.dropout(y)
        y = self.act2(y)
        #y = self.conv2_1(y)
        before_pool = y
        y = self.pool(y)
        return y, before_pool
@torch.jit.script
def autocrop(from_down: torch.Tensor, from_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Crops feature tensors from the encoder and decoder pathways so that they
    can be combined.
    - If inputs from the encoder pathway have shapes that are not divisible
      by 2, the use of ``nn.MaxPool(ceil_mode=True)`` leads to the 2x
      upconvolution results being too large by one element in each odd
      dimension, so they need to be cropped in these dimensions.
    - If VALID convolutions are used, feature tensors get smaller with each
      convolution, so we need to center-crop the larger feature tensors from
      the encoder pathway to make features combinable with the smaller
      decoder feautures.
    Args:
        from_down: Feature from encoder pathway (``DownConv``)
        from_up: Feature from decoder pathway (2x upsampled)
    Returns:
    """
    ndim = from_down.dim()  # .ndim is not supported by torch.jit

    if from_down.shape[2:] == from_up.shape[2:]:  # No need to crop anything
        return from_down, from_up

    ds = from_down.shape[2:]
    us = from_up.shape[2:]
    upcrop = [u - ((u - d) % 2) for d, u in zip(ds, us)]

    if ndim == 4:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1]]
    if ndim == 5:
        from_up = from_up[:, :, :upcrop[0], :upcrop[1], :upcrop[2]]

    # Step 2: Handle center-crop resulting from valid convolutions
    ds = from_down.shape[2:]
    us = from_up.shape[2:]

    assert ds[0] >= us[0], f'{ds, us}'
    assert ds[1] >= us[1]
    if ndim == 4:
        from_down = from_down[
            :,
            :,
            (ds[0] - us[0]) // 2:(ds[0] + us[0]) // 2,
            (ds[1] - us[1]) // 2:(ds[1] + us[1]) // 2
        ]
    elif ndim == 5:
        assert ds[2] >= us[2]
        from_down = from_down[
            :,
            :,
            ((ds[0] - us[0]) // 2):((ds[0] + us[0]) // 2),
            ((ds[1] - us[1]) // 2):((ds[1] + us[1]) // 2),
            ((ds[2] - us[2]) // 2):((ds[2] + us[2]) // 2),
        ]
    return from_down, from_up


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    att: Optional[torch.Tensor]

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', planar=False,
                 activation='relu', normalization=None, full_norm=True, dim=3, conv_mode='same',
                 attention=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.normalization = normalization
        padding = 1 if 'same' in conv_mode else 0

        self.upconv = upconv2(self.in_channels, self.out_channels,
                              mode=self.up_mode, planar=planar, dim=dim)
        # hi minseok
        #if self.merge_mode == 'concat':
        #    #if self.out_channels == 64:
        #    #    self.conv1 = conv3(self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        #    #else:
        #    self.conv1 = conv3(2*self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding)
        #else:
        #    # num of input channels to conv2 is same
        #    self.conv1 = conv3(
        #        self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        #    )

        self.conv1 = nn.Conv3d(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=True)
        self.conv1_2 = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=True)

        self.conv2 = conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding
        )

        self.act0 = get_activation(activation)
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)

        if full_norm:
            self.norm0 = get_normalization(normalization, self.out_channels, dim=dim)
            self.norm1 = get_normalization(normalization, self.out_channels, dim=dim)
        else:
            self.norm0 = nn.Identity()
            self.norm1 = nn.Identity()
        self.norm2 = get_normalization(normalization, self.out_channels, dim=dim)
        if attention:
            self.attention = GridAttention(
                in_channels=in_channels // 2, gating_channels=in_channels, dim=dim
            )
        else:
            self.attention = DummyAttention()
        self.att = None  # Field to store attention mask for later analysis

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """

        updec = self.upconv(dec)
        enc, updec = autocrop(enc, updec)
        genc, att = self.attention(enc, dec)
        if not torch.jit.is_scripting():
            self.att = att
        updec = self.norm0(updec)
        updec = self.act0(updec)
        #if self.merge_mode == 'concat':
        #    mrg = torch.cat((updec, genc), 1)
        #else:
        #    mrg = updec + genc
        #if updec.shape[1] == 64:
        #    mrg = updec
        #else:
        mrg = torch.cat((updec, genc), 1)

        y = self.conv1(mrg)
        y = self.norm1(y)
        y = self.act1(y)
        y = self.conv1_2(y)

        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act2(y)
        return y
    
class ResizeConv(nn.Module):
    """Upsamples by 2x and applies a convolution.
    This is meant as a replacement for transposed convolution to avoid
    checkerboard artifacts. See
    - https://distill.pub/2016/deconv-checkerboard/
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, planar=False, dim=3,
                 upsampling_mode='nearest'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.scale_factor = 2
        if dim == 3 and planar:  # Only interpolate (H, W) dims, leave D as is
            self.scale_factor = planar_kernel(self.scale_factor)
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.upsampling_mode)
        # TODO: Investigate if 3x3 or 1x1 conv makes more sense here and choose default accordingly
        # Preliminary notes:
        # - conv3 increases global parameter count by ~10%, compared to conv1 and is slower overall
        # - conv1 is the simplest way of aligning feature dimensions
        # - conv1 may be enough because in all common models later layers will apply conv3
        #   eventually, which could learn to perform the same task...
        #   But not exactly the same thing, because this layer operates on
        #   higher-dimensional features, which subsequent layers can't access
        #   (at least in U-Net out_channels == in_channels // 2).
        # --> Needs empirical evaluation
        if kernel_size == 3:
            self.conv = conv3(
                in_channels, out_channels, padding=1, planar=planar, dim=dim
            )
        elif kernel_size == 1:
            self.conv = conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(f'kernel_size={kernel_size} is not supported. Choose 1 or 3.')

    def forward(self, x):
        return self.conv(self.upsample(x))

class DummyAttention(nn.Module):
    def forward(self, x, g):
        return x, None


class sianet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3, # 10
            out_channels: int = 1, # 6*3 NEW: number of time slots to predict
            dropout_rate: float = 0.4,
            n_blocks: int = 5,
            start_filts: int = 32,
            up_mode: str = 'transpose',
            merge_mode: str = 'concat',
            planar_blocks: Sequence = (),
            batch_norm: str = 'unset',
            attention: bool = False,
            activation: Union[str, nn.Module] = 'relu',
            normalization: str = 'batch',
            full_norm: bool = True,
            dim: int = 3,
            conv_mode: str = 'same',
    ):
        super().__init__()
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        self.normalization = normalization
        self.attention = attention
        self.conv_mode = conv_mode
        self.activation = activation
        self.dim = dim

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.planar_blocks = planar_blocks

        # create the encoder pathway and add to a list
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks - 1 else False
            planar = i in self.planar_blocks

            down_conv = DownConv(
                ins,
                outs,
                dropout_rate,
                pooling=pooling,
                planar=planar,
                activation=activation,
                normalization=normalization,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
            )
            self.down_convs.append(down_conv)

        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = n_blocks - 2 - i in self.planar_blocks

            up_conv = UpConv(
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                planar=planar,
                activation=activation,
                normalization=normalization,
                attention=attention,
                full_norm=full_norm,
                dim=dim,
                conv_mode=conv_mode,
            )
            self.up_convs.append(up_conv)
        self.reduce_channels = conv1(192, ## 4 = experiment / len_seq_in # 160
                                     self.out_channels, dim=dim)

        self.dropout = nn.Dropout3d(dropout_rate)
        self.apply(self.weight_init)
        self.up = nn.Upsample(size=(32,252,252), mode='trilinear')
        self.filter = Learnable_Filter()
        self.lka = Attention()

    @staticmethod
    def weight_init(m):
        #if isinstance(m, GridAttention):
        #    return
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if getattr(m, 'bias') is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_outs = []

        # Encoder pathway, save outputs for merging
        i = 0  # Can't enumerate because of https://github.com/pytorch/pytorch/issues/16123
        for module in self.down_convs:
            x, before_pool = module(x)
            before_pool =  self.dropout(before_pool)  # for skip connections
            encoder_outs.append(before_pool)
            i += 1

        x = self.dropout(x)  # at bottom of the U, as in the original U-Net
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
            i += 1

        xs = x.shape
        # print("x shape",xs)
        x = torch.reshape(x,(xs[0],xs[1]*xs[2],1,xs[3],xs[4]));
        x = self.reduce_channels(x)
        xs = x.shape
        # print("reduce channels shape",xs)
        x = torch.reshape(x, (xs[0], 1, 1, xs[3], xs[4])) # (xs[0], 3, 6, xs[3], xs[4])
        return x
    
    @torch.jit.unused
    def forward_gradcp(self, x):
        """``forward()`` implementation with gradient checkpointing enabled.
        Apart from checkpointing, this behaves the same as ``forward()``."""
        encoder_outs = []
        i = 0
        for module in self.down_convs:
            x, before_pool = checkpoint(module, x)
            encoder_outs.append(before_pool)
            i += 1
        i = 0
        for module in self.up_convs:
            before_pool = encoder_outs[-(i+2)]
            x = checkpoint(module, before_pool, x)
            i += 1
        x = self.conv_final(x)
        # self.feature_maps = [x]  # Currently disabled to save memory
        return x


if __name__ == '__main__':
    inputs = torch.randn(8, 3, 6, 150, 150) #(b, c, t, w , h) # (3, 10, 5, 256, 256)
    model = sianet()
    output = model(inputs)
    print(output.shape)
    print('All tests sucessful!')