from argparse import ArgumentParser

from utils.losses import SpatialTemporal_Loss, TemporalLoss
from pathlib import Path

import pytorch_lightning as pl
from models.networks import define_generator, SpectNormDiscriminator
from models.pretrainnet import VGGCaffePreTrained
from losses.gan_loss import LSGanLoss
import torch
import torch.nn as nn
import torch.nn.functional as nf
import numpy as np
from joblib import Parallel, delayed
import itertools
from torch.distributions import Distribution
from torchvision import transforms
from typing import List, Tuple
from utils.superpix import slic, adaptive_slic, sscolor
from functools import partial
from utils.util import log_images, save_images
import torch.nn.functional as F


class ClipLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=-1, max=1)


class WhiteBoxGANModel(pl.LightningModule):
    SuperPixelDict = {
        'slic': slic,
        'adaptive_slic': adaptive_slic,
        'sscolor': sscolor
    }

    def __init__(self,
                 lrG: float = 2e-4,
                 lrD: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.99,
                 tv_weight: float = 10000.0,
                 g_blur_weight: float = 0.1,
                 g_gray_weight: float = 0.1,
                 recon_weight: float = 200,
                 superpixel_fn: str = 'sscolor',
                 superpixel_kwarg: dict = {'seg_num': 200},
                 freeze_encoder: bool = False,
                 logging_dir=None,
                 attn_kernel_size=3,
                 stc_mode='cos',
                 adaptive_power=1.2,
                 attn_layers=[4, 7, 9],
                 stc_rank=True,
                 attn_normalize=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # networks
        self.logging_dir = Path(logging_dir)
        self.generator = define_generator(self.hparams.netG,
                                          self.hparams.res_blocks,
                                          self.hparams.g_norm, freeze_encoder,
                                          self.hparams.attn_kernel_size,
                                          self.hparams.attn_normalize
                                          )
        self.disc_gray = SpectNormDiscriminator(in_channel=1)
        self.disc_blur = SpectNormDiscriminator()
        self.guided_filter = GuidedFilter()
        self.lsgan_loss = LSGanLoss()
        self.colorshift = ColorShift()
        self.pretrained = VGGCaffePreTrained()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.variation_loss = VariationLoss(1)
        superpixel_kwarg['power'] = adaptive_power

        self.superpixel_fn = partial(self.SuperPixelDict[superpixel_fn],
                                     **superpixel_kwarg)
        if self.hparams.temporal_weight > 0:
            self.temporal_loss_handler = TemporalLoss(
                data_sigma=self.hparams.data_sigma,
                data_w=self.hparams.data_w,
                noise_level=self.hparams.data_noise_level,
                motion_level=self.hparams.data_motion_level,
                shift_level=self.hparams.data_shift_level,
                loss_type='pixel')
            if self.hparams.temporal_loss_type == 'STC':
                self.spatiotemporal_loss = SpatialTemporal_Loss(
                    attn_layers=self.hparams.attn_layers,
                    use_conv=False,
                    loss_mode=self.hparams.stc_mode,
                    rank=self.hparams.stc_rank)
                self.random_transform = transforms.RandomAffine(
                    degrees=1,
                    translate=(0.01, 0.01),
                    scale=(0.97, 1.03),
                    shear=1)
                self.color_transform = transforms.ColorJitter(brightness=.5, hue=.3)

                self.random_transform2 = transforms.RandomAffine(
                    degrees=20, translate=(0.1, 0.1), scale=(0.8, 8), shear=20)
            elif self.hparams.temporal_loss_type == 'flow':
                pass
            elif self.hparams.temporal_loss_type == 'affine':
                self.random_transform = transforms.RandomAffine(
                    degrees=20, translate=(0.1, 0.1), scale=(0.8, 8), shear=20)
            else:
                raise NotImplementedError

    @staticmethod
    def modify_commandline_options(parser: ArgumentParser, is_train):
        def str2list_int(s):
            return [int(x) for x in s.split(',')]

        def str2bool(s):
            if s == 'True':
                return True
            if s == 'False':
                return False
            raise NameError

        parser.set_defaults(tv_weight=10000,
                            res_blocks=4,
                            netG='novel',
                            lrG=2e-4,
                            lrD=2e-4,
                            epochs=20,
                            batch_size=16,
                            dataset_mode='whiteboxgan',
                            accelerator='dp')
        parser.add_argument('--g_blur_weight', type=float, default=0.1)
        parser.add_argument('--g_gray_weight', type=float, default=1)
        parser.add_argument('--recon_weight', type=float, default=200)
        # parser.add_argument('--superpixel_fn', default='sscolor')
        parser.add_argument('--superpixel_fn', default='adaptive_slic')
        parser.add_argument('--freeze_encoder', type=bool, default=False)
        parser.add_argument('--g_superpixel_weight', type=float, default=200)
        parser.add_argument('--identity_weight', type=float, default=0)
        parser.add_argument('--output_guided', type=str2bool, default=False)
        parser.add_argument('--attn_kernel_size', type=int, default=3)
        parser.add_argument('--transform_type',
                            type=str,
                            default='random_affine')
        parser.add_argument('--stc_mode', type=str, default='cos')
        parser.add_argument('--stc_rank', type=str2bool, default=False)
        parser.add_argument('--adaptive_power', type=float, default=1.2)
        parser.add_argument('--attn_layers',
                            type=str2list_int,
                            default=[2, 4, 7])
        parser.add_argument('--guided_eps', type=float, default=1e-2)
        parser.add_argument('--attn_normalize', type=str2bool, default=False)

        return parser

    def forward(self, input_photo) -> torch.Tensor: 
        generator_img = self.generator(input_photo)
        return generator_img

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx, optimizer_idx):
        input_cartoon, input_photo = batch

        if optimizer_idx == 0:  # train generator
            generator_img = self.generator(input_photo)

            if self.hparams.output_guided:
                output = self.guided_filter(input_photo,
                                            generator_img,
                                            r=1,
                                            eps=self.hparams.guided_eps)
            else:
                output = torch.clip(generator_img, -1, 1)
            # 1. blur for Surface Representation
            blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
            blur_fake_logit = self.disc_blur(blur_fake)
            g_loss_blur = self.hparams.g_blur_weight * self.lsgan_loss._g_loss(
                blur_fake_logit)

            # 2. gray for Textural Representation
            gray_fake, = self.colorshift(output)
            gray_fake_logit = self.disc_gray(gray_fake)
            g_loss_gray = self.hparams.g_gray_weight * self.lsgan_loss._g_loss(
                gray_fake_logit)

            # 3. superpixel for Structure Representation
            vgg_output = self.pretrained(output)
            output_superpixel = torch.zeros_like(input_photo)
            _, c, h, w = vgg_output.shape
            superpixel_loss = 0
            if self.hparams.g_superpixel_weight > 0:
                if self.hparams.output_guided:
                    output_superpixel = torch.from_numpy(
                        superpixel(
                            output.detach().permute(
                                (0, 2, 3, 1)).cpu().numpy(),
                            self.superpixel_fn)).to(self.device).permute(
                                (0, 3, 1, 2))
                else:
                    output_superpixel = torch.from_numpy(
                        superpixel(
                            self.guided_filter(
                                input_photo,
                                output,
                                r=1,
                                eps=self.hparams.guided_eps).detach().permute(
                                    (0, 2, 3, 1)).cpu().numpy(),
                            self.superpixel_fn)).to(self.device).permute(
                                (0, 3, 1, 2))

                vgg_superpixel = self.pretrained(output_superpixel)
                superpixel_loss = (self.hparams.g_superpixel_weight *
                                   self.l1_loss(vgg_superpixel, vgg_output) /
                                   (c * h * w))

            # 4. Content loss
            vgg_photo = self.pretrained(input_photo)
            photo_loss = self.hparams.recon_weight * self.l1_loss(
                vgg_photo, vgg_output) / (c * h * w)

            # 5. total variation loss
            tv_loss = self.hparams.tv_weight * self.variation_loss(output)

            # identity loss
            if self.hparams.identity_weight > 0:
                output_cartoon = self.generator(input_cartoon)
                identity_loss = self.hparams.identity_weight * torch.nn.functional.l1_loss(input_cartoon, output_cartoon)
            else:
                identity_loss = 0

            # 6. temporal loss
            temporal_loss = 0
            if self.hparams.temporal_weight > 0:

                if self.hparams.temporal_loss_type == 'STC':
                    if self.hparams.transform_type == 'random_affine':
                        second_frame = self.random_transform(input_photo)
                    elif self.hparams.transform_type == 'random_affine2':
                        second_frame = self.random_transform2(input_photo)
                    elif self.hparams.transform_type == 'flow':
                        second_frame, flow = self.temporal_loss_handler.GenerateFakeData(
                            input_photo)
                    elif self.hparams.transform_type == 'color':
                        second_frame = self.color_transform(input_photo)
                    elif self.hparams.transform_type == 'all':
                        second_frame = self.random_transform(self.color_transform(input_photo))
                    else:
                        raise NotImplementedError

                    styled_second_frame = self.generator(second_frame)
                    if self.hparams.output_guided:
                        styled_second_frame = self.guided_filter(
                            second_frame, styled_second_frame, r=1, eps=1e-2)
                    temporal_loss = self.spatiotemporal_loss(
                        input_photo, second_frame, output, styled_second_frame)
                elif self.hparams.temporal_loss_type == 'flow':
                    second_frame, flow = self.temporal_loss_handler.GenerateFakeData(
                        input_photo)
                    styled_second_frame = self.generator(second_frame)
                    if self.hparams.output_guided:
                        styled_second_frame = self.guided_filter(
                            second_frame, styled_second_frame, r=1, eps=1e-2)
                    temporal_loss = self.temporal_loss_handler.forward(
                        output, styled_second_frame, flow)
                elif self.hparams.temporal_loss_type == 'affine':
                    N = input_photo.shape[0]
                    composed_data = torch.cat([input_photo, output])
                    composed_second_data = self.random_transform(composed_data)
                    second_input, target_second_output = composed_second_data[:N], composed_second_data[N:]
                    second_output = self.generator(second_input)
                    temporal_loss = F.l1_loss(second_output, target_second_output)


            g_loss_total = tv_loss + g_loss_blur + g_loss_gray + superpixel_loss + photo_loss + temporal_loss * self.hparams.temporal_weight + identity_loss
            self.log_dict({
                'gen/g_loss': g_loss_total,
                'gen/tv_loss': tv_loss,
                'gen/g_blur': g_loss_blur,
                'gen/g_gray': g_loss_gray,
                'gen/photo_loss': photo_loss,
                'gen/superpixel_loss': superpixel_loss,
                'gen/temporal_loss': temporal_loss,
                'gen/identity_loss': identity_loss
            })
            if self.global_rank == 0 and self.global_step % 50 == 0:
                log_images(self, {
                    'input/real': input_photo,
                    'generate/superpix': output_superpixel,
                    'generate/anime': generator_img,
                    'generate/filtered': output,
                    'generate/gray': gray_fake,
                    'generate/blur': blur_fake,
                },
                           num=8)

            return g_loss_total
        elif optimizer_idx == 1:  # train discriminator
            generator_img = self.generator(input_photo)
            if self.hparams.output_guided:
                output = self.guided_filter(input_photo, generator_img, r=1)
            else:
                output = torch.clip(generator_img, -1, 1)
            # 1. blur for Surface Representation
            blur_fake = self.guided_filter(output, output, r=5, eps=2e-1)
            blur_cartoon = self.guided_filter(input_cartoon,
                                              input_cartoon,
                                              r=5,
                                              eps=2e-1)
            blur_real_logit = self.disc_blur(blur_cartoon)
            blur_fake_logit = self.disc_blur(blur_fake)
            d_loss_blur = self.lsgan_loss._d_loss(blur_real_logit,
                                                  blur_fake_logit)

            # 2. gray for Textural Representation
            gray_fake, gray_cartoon = self.colorshift(output, input_cartoon)
            gray_real_logit = self.disc_gray(gray_cartoon)
            gray_fake_logit = self.disc_gray(gray_fake)
            d_loss_gray = self.lsgan_loss._d_loss(gray_real_logit,
                                                  gray_fake_logit)

            d_loss_total = d_loss_blur + d_loss_gray
            self.log_dict({
                'dis/d_loss': d_loss_total,
                'dis/d_blur': d_loss_blur,
                'dis/d_gray': d_loss_gray
            })

            return d_loss_total

    def validation_step(self, batch, batch_idx):
        try:
            imgs, paths = batch
            n, c, h, w = imgs.shape
            imgs = F.interpolate(imgs, (h // 4 * 4, w // 4 * 4),
                                 mode='bilinear')
            generated_imgs = self.generator(imgs)
            saved_dir = self.logging_dir / 'val_results' / f'step_{self.global_step}'
            save_images(generated_imgs,
                        saved_dir,
                        paths,
                        inverse_normalize=True)
        except Exception as e:
            # print(e)
            pass

    def test_step(self, batch, batch_idx, dataloader_id=0):
        imgs, paths = batch
        generated_imgs = self(imgs)
        saved_dir = self.logging_dir / 'test_results'
        if dataloader_id == 0:  # photo/test_photo
            save_images(generated_imgs,
                        saved_dir / 'generated_cartoon',
                        paths,
                        inverse_normalize=True,
                        parent_level=0,
                        format='.png')
        elif dataloader_id == 1:  # DAVIS/tram/
            save_images(generated_imgs,
                        saved_dir,
                        paths,
                        inverse_normalize=True,
                        parent_level=2,
                        format='.jpg')

    def configure_optimizers(self):
        lr_g = self.hparams.lrG
        lr_d = self.hparams.lrD
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=lr_g,
                                 betas=(b1, b2))
        opt_d = torch.optim.Adam(itertools.chain(self.disc_blur.parameters(),
                                                 self.disc_gray.parameters()),
                                 lr=lr_d,
                                 betas=(b1, b2))

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 10) / float(11)
            return lr_l

        scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g,
                                                        lr_lambda=lambda_rule)
        scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d,
                                                        lr_lambda=lambda_rule)
        # scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=5, gamma=0.1)
        # scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=5, gamma=0.1)
        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    @staticmethod
    def pretrained_model():
        return WhiteBoxGANPretrain

class WhiteBoxGANPretrain(WhiteBoxGANModel):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx):
        input_cartoon, input_photo = batch
        generator_img = self.generator(input_photo)
        recon_loss = self.l1_loss(input_photo, generator_img)
        self.log('pretrain/recon_loss', recon_loss)
        return recon_loss

    def configure_optimizers(self):
        lr_g = self.hparams.lrG
        b1 = self.hparams.beta1
        b2 = self.hparams.beta2

        optim = torch.optim.Adam(self.generator.parameters(),
                                 lr=lr_g,
                                 betas=(b1, b2))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=5,
                                                    gamma=0.1)
        return {'optimizer': optim, 'lr_scheduler': scheduler}

    def validation_step(self, batch, batch_idx):
        try:
            imgs, paths = batch
            n, c, h, w = imgs.shape
            imgs = F.interpolate(imgs, (h // 4 * 4, w // 4 * 4), mode='bilinear')
            generated_imgs = self.generator(imgs)
            saved_dir = self.logging_dir / 'val_results' / f'init_epoch_{self.current_epoch}'
            save_images(generated_imgs, saved_dir, paths, inverse_normalize=True)
        except Exception as e:
            pass


def superpixel(batch_image: np.ndarray, superpixel_fn: callable) -> np.ndarray:
    """ convert batch image to superpixel

  Args:
      batch_image (np.ndarray): np.ndarry, shape must be [b,h,w,c]
      seg_num (int, optional): . Defaults to 200.

  Returns:
      np.ndarray: superpixel array, shape = [b,h,w,c]
  """
    num_job = batch_image.shape[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(superpixel_fn)(image)
                                         for image in batch_image)
    return np.array(batch_out)


class GuidedFilter(nn.Module):
    def box_filter(self, x: torch.Tensor, r):
        ch = x.shape[1]
        k = 2 * r + 1
        weight = 1 / ((k)**2)  # 1/9
        # [c,1,3,3] * 1/9
        box_kernel = torch.ones((ch, 1, k, k),
                                dtype=torch.float32,
                                device=x.device).fill_(weight)
        # same padding
        return nf.conv2d(x, box_kernel, padding=r, groups=ch)

    def forward(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
        """
        r: band width (like spatial filter in biliteral filter)
        eps:degree of edge preserving(like range filter in biliteral filter)
        """
        b, c, h, w = x.shape
        device = x.device
        N = self.box_filter(
            torch.ones((1, 1, h, w), dtype=x.dtype, device=device), r)

        mean_x = self.box_filter(x, r) / N
        mean_y = self.box_filter(y, r) / N
        cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
        var_x = self.box_filter(x * x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A, r) / N
        mean_b = self.box_filter(b, r) / N

        output = mean_A * x + mean_b
        return output


class ColorShift(nn.Module):
    def __init__(self, mode='uniform'):
        super().__init__()
        self.dist: Distribution = None
        self.mode = mode
        if self.mode == 'normal':
            self.dist = torch.distributions.Normal(
                torch.tensor((0.299, 0.587, 0.114)),
                torch.tensor((0.1, 0.1, 0.1)))
        elif self.mode == 'uniform':
            self.dist = torch.distributions.Uniform(
                torch.tensor((0.199, 0.487, 0.014)),
                torch.tensor((0.399, 0.687, 0.214)))

    def forward(self, *img: torch.Tensor) -> Tuple[torch.Tensor]:
        rgb = self.dist.sample().to(img[0].device)
        return [
            torch.sum(im * rgb.reshape(1, 3, 1, 1), dim=1, keepdim=True) /
            rgb.sum() for im in img
        ]


class VariationLoss(nn.Module):
    def __init__(self, k_size: int) -> None:
        super().__init__()
        self.k_size = k_size

    def forward(self, image: torch.Tensor):
        b, c, h, w = image.shape
        tv_h = torch.mean(
            (image[:, :, self.k_size:, :] - image[:, :, :-self.k_size, :])**2)
        tv_w = torch.mean(
            (image[:, :, :, self.k_size:] - image[:, :, :, :-self.k_size])**2)
        tv_loss = (tv_h + tv_w) / (3 * h * w)
        return tv_loss