import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.instancenorm import InstanceNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models
from torchvision.models import inception, resnet
import numpy as np
from .commons import Mean


###############################################################################
# Helper Functions
###############################################################################
def define_generator(
    netG,
    res_blocks,
    norm,
    freeze_encoder=False,
    attn_kernel_size=3,
    attn_normalize=False,
):
    model = SAGenerator(
        concat=False,
        freeze_encoder=freeze_encoder,
        attn_kernel_size=attn_kernel_size,
        attn_normalize=attn_normalize,
    )
    return model


def conv3x3(in_channel, out_channel, stride=1, padding=1, use_sn=False):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size=3, stride=stride, padding=padding
    )
    if use_sn:
        conv = spectral_norm(conv)
    return conv


def conv1x1(in_channel, out_channel, stride=1, padding=0):
    return nn.Conv2d(in_channel, out_channel, 1, stride=stride, padding=padding)


def upsample_x2(in_channel, out_channel):
    return nn.ConvTranspose2d(
        in_channel, out_channel, 3, stride=2, padding=1, output_padding=1
    )
    # return nn.ConvTranspose2d(in_channel, out_channel, 4, stride=2, padding=1)


def get_norm_layer(norm_type):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | layer
    """
    if norm_type == "layer":
        return functools.partial(nn.GroupNorm, 1, affine=True)  # set num_groups to 1
    elif norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError


class Discriminator(nn.Module):
    """Patch-level Discriminator architecture coming from CartoonGAN
    Discriminator output a tensor with (img_size // 4, img_size // 4),
    such that each item indicates a classification about a local patch.
    """

    def __init__(self, use_sn=True, norm="layer"):
        super().__init__()
        use_sn = True
        norm_layer = get_norm_layer(norm)
        self.networks = nn.Sequential(
            # flat layers
            conv3x3(3, 32, use_sn=use_sn),  # (32, 256, 256)
            nn.LeakyReLU(
                0.2
            ),  # not add instance normalization to keep color information
            # Two strided convolution blocks to encode essential local features
            conv3x3(32, 64, stride=2, use_sn=use_sn),  # (64, 128, 128)
            nn.LeakyReLU(0.2),
            conv3x3(64, 128, use_sn=use_sn),  # (128, 128, 128)
            norm_layer(128),
            nn.LeakyReLU(0.2),
            conv3x3(128, 128, stride=2, use_sn=use_sn),  # (128, 64, 64)
            nn.LeakyReLU(0.2),
            conv3x3(128, 256, use_sn=use_sn),  # (256, 64, 64)
            norm_layer(256),
            nn.LeakyReLU(0.2),
            # feature construction block
            conv3x3(256, 256, use_sn=use_sn),  # (256, 64, 64)
            norm_layer(256),
            nn.LeakyReLU(0.2),
            conv3x3(256, 1),
        )

    def forward(self, x):
        """output a tensor whose item indicates which domain a local patch comes from
        @args:
            x: image tensor of shape (3, 256, 256)
        """
        return self.networks(x)


class ResBlock(nn.Module):
    def __init__(self, channel, norm=None, act="relu"):
        super().__init__()
        self.norm = norm
        self.conv1 = conv3x3(channel, channel)
        if norm is not None:
            self.norm1 = get_norm_layer(norm)(channel)
        if act == "relu":
            self.relu = nn.ReLU(True)
        elif act == "leaky_relu":
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = conv3x3(channel, channel)
        if norm is not None:
            self.norm2 = get_norm_layer(norm)(channel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm is not None:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm is not None:
            out = self.norm2(out)

        out = self.relu(out + identity)
        return out


class LocalAttention(nn.Module):
    def __init__(
        self,
        r=3,
        normalize=False,
        projection=False,
        in_channel=None,
        mid_channel=None,
        soft_max=True,
    ):
        super().__init__()
        self.unfold = nn.Unfold(r, dilation=1, padding=(r - 1) // 2, stride=1)
        self.r = r
        self.normalize = normalize
        self.projection = projection
        self.soft_max = soft_max
        if projection:
            self.key_proj = nn.Conv2d(in_channel, mid_channel, kernel_size=1)
            self.query_proj = nn.Conv2d(in_channel, mid_channel, kernel_size=1)

    def forward(self, key, query):
        raw_key = key
        if self.projection:
            if self.normalize:
                key = F.normalize(self.key_proj(key), dim=1)
                query = F.normalize(self.query_proj(query), dim=1)
            else:
                key = self.key_proj(key)
                query = self.query_proj(query)

        N, C, H, W = key.shape
        key = (
            self.unfold(key).permute(0, 2, 1).reshape(N * H * W, C, self.r ** 2)
        )  # (N, C*9, H*W) -> (N*H*W, C, 9)
        query = query.permute(0, 2, 3, 1).reshape(
            N * H * W, 1, C
        )  # (N, C, H, W) -> (NHW, 1, C)
        prod = torch.matmul(
            query, key
        ).squeeze()  # (NHW, 1, C) x (NHW, C, 9) -> (NHW, 1, 9) -> (NHW, 9)
        if self.soft_max:
            map = torch.softmax(prod, dim=1)  # (NHW, 9)
            if self.projection:
                key = (
                    self.unfold(raw_key)
                    .permute(0, 2, 1)
                    .reshape(N * H * W, -1, self.r ** 2)
                )  # (N, C*9, H*W) -> (N*H*W, C, 9)
            out = torch.matmul(
                key, map.unsqueeze(2)
            ).squeeze()  # (NHW, C, 9) x (NHW, 9) -> (NHW, C, 1) -> (NHW, C)
            out = out.reshape(N, H, W, -1).permute(0, 3, 1, 2)
        else:
            map = torch.argmax(prod, dim=1).reshape(-1, 1, 1).repeat(1, C, 1)  # (NHW)
            out = key.gather(2, map).reshape(N, H, W, C).permute(0, 3, 1, 2)  # (NHW, C)
        return out


def get_filter(filt_size=3):
    if filt_size == 1:
        a = np.array([1.0,])
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filt_size == 6:
        a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filt_size == 7:
        a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        else:
            return F.conv2d(
                self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=self.factor, mode=self.mode
        )


class Upsample(nn.Module):
    def __init__(self, channels, pad_type="repl", filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad(inp),
            self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer

class SAGenerator(nn.Module):
    "Generator with Spatially-adaptive semantic alignment"
    def __init__(
        self,
        channel=32,
        num_blocks=4,
        concat=False,
        freeze_encoder=False,
        attn_kernel_size=3,
        attn_normalize=False,
        soft_max=True,
        skip=False,
    ):
        super().__init__()
        self.skip = skip

        def Conv2d(in_channel, out_channel, kernel_size, padding=1, stride=1):
            model = nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size, padding=padding, stride=stride
                ),
                # nn.InstanceNorm2d(out_channel)
            )
            return model

        self.concat = concat
        self.local_attn = LocalAttention(
            r=attn_kernel_size, normalize=attn_normalize, soft_max=soft_max
        )
        self.conv = Conv2d(3, channel, [7, 7], padding=3)  # same 256,256
        self.conv1 = Conv2d(
            channel, channel, [3, 3], stride=2, padding=1
        )  # same 128,128
        self.conv2 = Conv2d(channel, channel * 2, [3, 3], padding=1)  # 128,128
        self.conv3 = Conv2d(
            channel * 2, channel * 2, [3, 3], stride=2, padding=1
        )  # 64,64
        self.conv4 = Conv2d(channel * 2, channel * 4, [3, 3], padding=1)  # 64,64

        self.resblock = nn.Sequential(
            *[ResBlock(channel * 4, act="leaky_relu") for i in range(num_blocks)]
        )

        self.conv5 = Conv2d(channel * 4, channel * 2, [3, 3], padding=1)  # 64,64
        if concat:
            self.conv6 = Conv2d(
                channel * 2 * 2, channel * 2, [3, 3], padding=1
            )  # 64,64
        else:
            self.conv6 = Conv2d(channel * 2, channel * 2, [3, 3], padding=1)  # 64,64
        self.conv7 = Conv2d(channel * 2, channel, [3, 3], padding=1)  # 64,64
        if concat:
            self.conv8 = Conv2d(2 * channel, channel, [3, 3], padding=1)  # 64,64
        else:
            self.conv8 = Conv2d(channel, channel, [3, 3], padding=1)  # 64,64
        self.conv9 = Conv2d(channel, 3, [7, 7], padding=3)  # 64,64

        self.leak_relu = nn.LeakyReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.act = nn.Tanh()

        if freeze_encoder:
            for layer in [self.conv, self.conv1, self.conv2, self.conv3, self.conv4]:
                for parameter in layer.parameters():
                    parameter.requires_grad_ = False

    def forward(self, inputs):
        x0 = self.conv(inputs)
        x0 = self.leak_relu(x0)  # 256, 256, 32

        if self.skip:
            x4 = self.conv8(x0)
            x4 = self.leak_relu(x4)
            x4 = self.conv9(x4)  # 256, 256, 32

            return self.act(x4)

        x1 = self.conv1(x0)
        x1 = self.leak_relu(x1)
        x1 = self.conv2(x1)
        x1 = self.leak_relu(x1)  # 128, 128, 64

        x2 = self.conv3(x1)
        x2 = self.leak_relu(x2)
        x2 = self.conv4(x2)
        x2 = self.leak_relu(x2)  # 64, 64, 128

        x2 = self.resblock(x2)  # 64, 64, 128
        x2 = self.conv5(x2)
        x2 = self.leak_relu(x2)  # 64, 64, 64

        x3 = self.upsample(x2)
        if self.concat:
            x3 = self.conv6(
                torch.cat([self.local_attn(x3, x1), x1], dim=1)
            )  # local attention
        else:
            x3 = self.conv6(x1 + self.local_attn(x3, x1))  # local attention

        x3 = self.leak_relu(x3)
        x3 = self.conv7(x3)
        x3 = self.leak_relu(x3)  # 128, 128, 32

        x4 = self.upsample(x3)
        if self.concat:
            x4 = self.conv8(
                torch.cat([self.local_attn(x4, x0), x0], dim=1)
            )  # local attention
        else:
            x4 = self.conv8(x0 + self.local_attn(x4, x0))  # local attention
        x4 = self.leak_relu(x4)
        x4 = self.conv9(x4)  # 256, 256, 32

        return self.act(x4)

class Vgg19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(pretrained=True).features[:26]  # conv_4_4
        self.model.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
        for param in self.model.parameters():
            param.require_grad = False

    def forward(self, x):
        # denoramlize from [-1, 1] to [0, 1]
        x = (x + 1) * 0.5
        # normalize with ImageNet mean and variance
        x = (x - self.mean) / self.std
        out = self.model(x)
        return out


class VGG16(nn.Module):
    def __init__(self, after_relu=True):
        super(VGG16, self).__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
        features = models.vgg16(pretrained=True).features
        # print(features[0].weight)
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, layers=None, encode_only=False, resize=False):
        # denoramlize from [-1, 1] to [0, 1]
        x = (x + 1) * 0.5
        # normalize with ImageNet mean and variance
        x = (x - self.mean) / self.std

        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            "relu1_1": relu1_1,
            "relu1_2": relu1_2,
            "relu2_1": relu2_1,
            "relu2_2": relu2_2,
            "relu3_1": relu3_1,
            "relu3_2": relu3_2,
            "relu3_3": relu3_3,
            "relu4_1": relu4_1,
            "relu4_2": relu4_2,
            "relu4_3": relu4_3,
            "relu5_1": relu5_1,
            "relu5_2": relu5_2,
            "relu5_3": relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out["relu3_1"]
        return out


class SpectNormDiscriminator(nn.Module):
  def __init__(self, channel=32, patch=True, in_channel=3):
    super().__init__()
    self.channel = channel
    self.patch = patch
    l = []
    for idx in range(3):
      l.extend([
          spectral_norm(nn.Conv2d(in_channel, channel * (2**idx), 3, stride=2, padding=1)),
          nn.LeakyReLU(inplace=True),
          spectral_norm(nn.Conv2d(channel * (2**idx), channel * (2**idx), 3, stride=1, padding=1)),
          nn.LeakyReLU(inplace=True),
      ])
      in_channel = channel * (2**idx)
    self.body = nn.Sequential(*l)
    if self.patch:
      self.head = spectral_norm(nn.Conv2d(in_channel, 1, 1, padding=0))
    else:
      self.head = nn.Sequential(Mean([1, 2]), nn.Linear(in_channel, 1))

  def forward(self, x):
    x = self.body(x)
    x = self.head(x)
    return x
