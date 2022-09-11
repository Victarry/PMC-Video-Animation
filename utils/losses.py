from argparse import Namespace
from sys import path
from models.networks import VGG16
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd, nn
from torch.autograd import Variable
from torch.nn import functional as F


def compute_gradient_penalty(real_sample, fake_sample, netD):
    alpha = torch.zeros(real_sample.size(0), 1, 1,
                        1).uniform_(0, 1).to(real_sample.device)
    # interpolated = alpha * real_sample + (1-alpha) * fake_sample
    interpolated = alpha * (real_sample - fake_sample) + fake_sample
    d_interpolated = netD(interpolated)

    fake = torch.ones_like(d_interpolated)
    # Get gradients w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=fake,
        create_graph=
        True,  # created graph for gradient to get higher derivatives
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    return gradient_penalty


class GANLoss(nn.Module):
    def __init__(self, gan_mode):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.gan_mode = gan_mode

        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan-gp':
            self.loss = None
        else:
            raise NotImplementedError(
                f'Gan mode {gan_mode} is not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        NOTE: use expand_as will not allocate new memory like broadcast.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).type_as(prediction)

    def get_loss(self, prediction, target_is_real):
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan-gp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def get_multiple_losses(self, real_logits: List[torch.Tensor],
                 fake_logits: List[torch.Tensor]):
        real_losses = []
        for real_logit in real_logits:
            loss = self.get_loss(real_logit, True)
            real_losses.append(loss)

        fake_losses = []
        for fake_logit in fake_logits:
            loss = self.get_loss(fake_logit, False)
            fake_losses.append(loss)

        return real_losses, fake_losses


class AnimeDiscriminatorLoss():
    def __init__(self, gan_mode, style):
        self.loss_func = GANLoss(gan_mode)

        if style == 'Hayao':
            # self.content_weight = 1.5
            # self.style_weight = 2.5
            # self.color_weight = 15
            # self.tv_weight = 1
            self.real_weight = 1.2
            self.fake_weight = 1.2
            self.gray_weight = 1.2
            self.smooth_weight = 0.8
        elif style == 'Shinkai':
            # self.content_weight = 1.2
            # self.style_weight = 2.0
            # self.color_weight = 10
            # self.tv_weight = 1
            self.real_weight = 1.7
            self.fake_weight = 1.7
            self.gray_weight = 1.7
            self.smooth_weight = 1
        elif style == 'Paprika':
            # self.content_weight = 1.5
            # self.style_weight = 2.5
            # self.color_weight = 15
            # self.tv_weight = 0.1
            self.real_weight = 1.0
            self.fake_weight = 1.0
            self.gray_weight = 1.0
            self.smooth_weight = 0.005
        self.gray_weight = 0

    def __call__(self, anime, fake_anime, gray_anime, smooth_anime):
        (real_loss, ), (fake_loss, gray_loss, smooth_anime_loss) = self.loss_func.get_multiple_losses(
            real_logits=(anime, ),
            fake_logits=(fake_anime, gray_anime, smooth_anime))

        d_loss = self.real_weight * real_loss + self.fake_weight * fake_loss + self.gray_weight * gray_loss + self.smooth_weight * smooth_anime_loss
        return d_loss, (real_loss, fake_loss, gray_loss, smooth_anime_loss)

class AnimeGeneratorLoss():
    def __init__(self) -> None:
        pass
    
    def __call__(self):
        pass

def generator_adv_loss(gan_type, fake):
    if gan_type == 'gan':
        loss = F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))

    elif gan_type == 'lsgan':
        loss = torch.mean((fake - 1)**2)

    elif gan_type == 'wgan-gp':
        loss = -torch.mean(fake)

    return loss


def gram_matrix(input):
    N, C, H, W = input.size()

    features = input.view(N, C, H * W)  # (N, C, H*W)

    G = torch.bmm(features, features.transpose(2, 1))  # (N, C, C)

    # normalization
    return G.div(C * H * W)


def style_loss_func(input_feature, target_feature):
    return F.l1_loss(gram_matrix(input_feature), gram_matrix(target_feature))


def total_variation(input):
    N, C, H, W = input.shape
    tv_h = torch.pow(input[:, :, :-1] - input[:, :, 1:], 2).mean()
    tv_w = torch.pow(input[:, :, :, :-1] - input[:, :, :, 1:], 2).mean()
    return tv_h + tv_w


def rgb2yuv(img):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py#L3884
    """
    _rgb_to_yuv_kernel = np.array([[0.299, -0.147, 0.615],
                      [0.587, -0.289, -0.515],
                      [0.114, 0.436, -0.100]], dtype=np.float32)
    rgb2yuv_filter = torch.tensor(_rgb_to_yuv_kernel.T).reshape(3, 3, 1, 1).to(img.device) # (out_channel, in_channel, H, W)
    return F.conv2d(img, rgb2yuv_filter)

def color_reconstruction_loss(fake, photo):
    fake = rgb2yuv((fake+1)/2)
    photo = rgb2yuv((photo+1)/2)
    y_loss = F.l1_loss(fake[:, :, :, 0], photo[:, :, :, 0])
    u_loss = F.smooth_l1_loss(fake[:, :, :, 1], photo[:, :, :, 1])
    v_loss = F.smooth_l1_loss(fake[:, :, :, 2], photo[:, :, :, 2])
    return y_loss + u_loss + v_loss


def warp(x, flo, padding_mode='border', interpolation='bilinear', align_corners=True):
    """
    warp img2 to img1 using forward flow
    @args:
        x: tensor of shape(N, C, H, W), the source image to be processed with optical flow
        flo: shape (N, 2, H, W) optical flow between first and second frame, each element contains relative motion.
    """
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()  # (N, 2, H, W)
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0 # Note: use W-1 instead of W
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # (N, H, W, 2) indicates x and y to sample
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear', align_corners=align_corners)
    return output

class TemporalLoss(nn.Module):
    def __init__(self,
                 data_sigma=True,
                 data_w=True,
                 noise_level=0.001,
                 motion_level=8,
                 shift_level=10, loss_type='pixel'):

        super().__init__()
        self.MSE = torch.nn.MSELoss()

        self.data_sigma = data_sigma
        self.data_w = data_w
        self.noise_level = noise_level
        self.motion_level = motion_level
        self.shift_level = shift_level
        self.loss_type = loss_type
        config = Namespace(small=False, mixed_precision=False)

    """ Flow should have most values in the range of [-1, 1].
        For example, values x = -1, y = -1 is the left-top pixel of input,
        and values  x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame """

    def GaussianNoise(self, ins, mean=0, stddev=0.001):
        stddev = stddev + np.random.random() * stddev
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))

        if ins.is_cuda:
            noise = noise.cuda()
        return ins + noise

    def GenerateFakeFlow(self, height, width):
        ''' height: img.shape[0]
            width:  img.shape[1] '''

        if self.motion_level > 0:
            flow = np.random.normal(0,
                                    scale=self.motion_level,
                                    size=[height // 100, width // 100, 2])
            flow = cv2.resize(flow, (width, height))
            flow[:, :, 0] += np.random.randint(-self.shift_level,
                                            self.shift_level)
            flow[:, :, 1] += np.random.randint(-self.shift_level,
                                            self.shift_level)
            flow = cv2.blur(flow, (100, 100))
        else:
            flow = np.ones([width, height, 2])
            flow[:, :, 0] = np.random.randint(-self.shift_level, self.shift_level)
            flow[:, :, 1] = np.random.randint(-self.shift_level, self.shift_level)

        return torch.from_numpy(flow.transpose((2, 0, 1))).float()

    def GenerateFakeData(self, first_frame):
        ''' Input should be a (N, 3, H, W) tensor of value range [0,1]. '''

        if self.data_w:
            forward_flow = self.GenerateFakeFlow(first_frame.shape[2],
                                                 first_frame.shape[3])
            if first_frame.is_cuda:
                forward_flow = forward_flow.cuda()
            forward_flow = forward_flow.expand(first_frame.shape[0], 2,
                                               first_frame.shape[2],
                                               first_frame.shape[3])
            second_frame = warp(first_frame, forward_flow)
        else:
            second_frame = first_frame.clone()
            forward_flow = None

        if self.data_sigma:
            second_frame = self.GaussianNoise(second_frame,
                                              stddev=self.noise_level)

        return second_frame, forward_flow

    def forward(self, first_frame, second_frame, forward_flow):
        if self.loss_type == 'flow':
            _, out_flow = self.raft(second_frame, first_frame, test_mode=True)
            temporalloss = torch.mean(torch.abs(out_flow-forward_flow))
        elif self.loss_type == 'pixel':
            if self.data_w:
                first_frame = warp(first_frame, forward_flow)
            temporalloss = torch.mean(torch.abs(first_frame - second_frame))
        else:
            raise NameError()

        return temporalloss


class Spatial_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VGG16()
        self.attn_layers = [4, 7, 9]
        self.spaital_correlative_loss = SpatialCorrelativeLoss(use_conv=False)


    def forward(self, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = self.net(src, self.attn_layers, encode_only=True)
        feats_tgt = self.net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = self.net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.spaital_correlative_loss.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.spaital_correlative_loss.conv_init:
            self.spaital_correlative_loss.update_init_()

        return total_loss / n_layers

class SpatialTemporal_Loss(nn.Module):
    def __init__(self,  attn_layers = [4, 7, 9], **kargs):
        super().__init__()
        self.net = VGG16()
        self.attn_layers = attn_layers
        self.loss_func = SpatialTemporalCorrelativeLoss(**kargs)


    def forward(self, src1, src2, tgt1, tgt2, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src1 = self.net(src1, self.attn_layers, encode_only=True)
        feats_src2 = self.net(src2, self.attn_layers, encode_only=True)
        feats_tgt1 = self.net(tgt1, self.attn_layers, encode_only=True)
        feats_tgt2 = self.net(tgt2, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = self.net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src1, feat_src2, feat_tgt1, feat_tgt2, feat_oth) in enumerate(zip(feats_src1, feats_src2, feats_tgt1, feats_tgt2, feats_oth)):
            loss = self.loss_func.loss(feat_src1, feat_src2, feat_tgt1, feat_tgt2, feat_oth, i)
            total_loss += loss.mean()

        if not self.loss_func.conv_init:
            self.loss_func.update_init_()

        return total_loss / n_layers

class PatchSim(nn.Module):
    """Calculate the similarity in selected patches"""
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(PatchSim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, other=None, patch_ids=None):
        """
        Calculate the similarity for selected patches
        """
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        if other is not None: # 
            query1, key1, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
            query2, key2, patch_ids = self.select_patch(other, patch_ids=patch_ids)
            patch_sim = query1.bmm(key2)
        else:
            query, key, patch_ids = self.select_patch(feat, patch_ids=patch_ids)
            patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def select_patch(self, feat, patch_ids=None):
        """
        Select the patches
        """
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if patch_ids is None:
                patch_ids = torch.randperm(feat_reshape.size(1), device=feat.device)
                patch_ids = patch_ids[:int(min(self.patch_nums, patch_ids.size(0)))]
            feat_query = feat_reshape[:, patch_ids, :]       # B*Num*C
            feat_key = []
            Num = feat_query.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = patch_ids // W, patch_ids % W
                # patch should in the feature
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    feat_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                feat_key = torch.stack(feat_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                feat_key = feat_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                feat_query = feat_query.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                feat_key = feat.reshape(B, C, W*H)
        else:
            feat_query = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            feat_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return feat_query, feat_key, patch_ids


class SpatialCorrelativeLoss(nn.Module):
    """
    learnable patch-based spatially-correlative loss with contrastive learning
    """
    def __init__(self, loss_mode='cos', patch_nums=256, patch_size=32, norm=True, use_conv=True,
                 init_type='normal', init_gain=0.02, gpu_ids=[], T=0.1, rank=True):
        super(SpatialCorrelativeLoss, self).__init__()
        self.patch_sim = PatchSim(patch_nums=patch_nums, patch_size=patch_size, norm=norm)
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.norm = norm
        self.use_conv = use_conv
        self.conv_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.rank = rank

    def update_init_(self):
        self.conv_init = True

    # def create_conv(self, feat, layer):
    #     """
    #     create the 1*1 conv filter to select the features for a specific task
    #     :param feat: extracted features from a pretrained VGG or encoder for the similarity and dissimilarity map
    #     :param layer: different layers use different filter
    #     :return:
    #     """
    #     input_nc = feat.size(1)
    #     output_nc = max(32, input_nc // 4)
    #     conv = nn.Sequential(*[nn.Conv2d(input_nc, output_nc, kernel_size=1),
    #                            nn.ReLU(),
    #                            nn.Conv2d(output_nc, output_nc, kernel_size=1)])
    #     conv.to(feat.device)
    #     setattr(self, 'conv_%d' % layer, conv)
    #     init_net(conv, self.init_type, self.init_gain, self.gpu_ids)

    def cal_sim(self, f_src, f_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        if self.use_conv:
            if not self.conv_init:
                self.create_conv(f_src, layer)
            conv = getattr(self, 'conv_%d' % layer)
            f_src, f_tgt = conv(f_src), conv(f_tgt)
            f_other = conv(f_other) if f_other is not None else None
        sim_src, patch_ids = self.patch_sim(f_src, patch_ids=patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f_tgt, patch_ids=patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids=patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other

    def compare_sim(self, sim_src, sim_tgt, sim_other):
        """
        measure the shape distance between the same shape and different inputs
        :param sim_src: the shape similarity map from source input image
        :param sim_tgt: the shape similarity map from target output image
        :param sim_other: the shape similarity map from other input image
        :return:
        """
        B, Num, N = sim_src.size()
        if self.loss_mode == 'info' or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(sam_self, torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % (Num))
        else:
            if self.rank:
                tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
                num = int(N / 4)
                src = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
                tgt = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
            else:
                num = N
                src = sim_src
                tgt = sim_tgt
            if self.loss_mode == 'l1':
                loss = self.criterion((N / num) * src, (N / num) * tgt)
            elif self.loss_mode == 'cos':
                sim_pos = F.cosine_similarity(src, tgt, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)

        return loss

    def loss(self, f_src, f_tgt, f_other=None, layer=0):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :param layer:
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f_src, f_tgt, f_other, layer)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        return loss


class SpatialTemporalCorrelativeLoss(SpatialCorrelativeLoss):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def loss(self, f1_src, f2_src, f1_tgt, f2_tgt, f_other=None, layer=0):
        """
        calculate the spatial similarity and dissimilarity loss for given features from source and target domain
        :param f_src: source domain features
        :param f_tgt: target domain features
        :param f_other: other random sampled features
        :param layer:
        :return:
        """
        sim_src, sim_tgt, sim_other = self.cal_sim(f1_src, f2_src, f1_tgt, f2_tgt, f_other, layer)
        # calculate the spatial similarity for source and target domain
        loss = self.compare_sim(sim_src, sim_tgt, sim_other)
        return loss

    def cal_sim(self, f1_src, f2_src, f1_tgt, f2_tgt, f_other=None, layer=0, patch_ids=None):
        """
        calculate the similarity map using the fixed/learned query and key
        :param f_src: feature map from source domain
        :param f_tgt: feature map from target domain
        :param f_other: feature map from other image (only used for contrastive learning for spatial network)
        :return:
        """
        sim_src, patch_ids = self.patch_sim(f1_src, f2_src, patch_ids)
        sim_tgt, patch_ids = self.patch_sim(f1_tgt, f2_tgt, patch_ids)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, patch_ids)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other