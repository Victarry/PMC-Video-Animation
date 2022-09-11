from typing import Dict, List
from PIL.Image import Image
import torchvision
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import torch
from pathlib import Path
import pytorch_lightning as pl

def inverse_transform(tensor):
    """change float tensor of (1, 3, H, W) in [-1, 1] to uint8 numpy array of shape (H, W, C) in [0, 255]
    """
    tensor = (tensor.squeeze() + 1) / 2 * 255
    tensor = tensor.permute(1, 2, 0)
    img = tensor.detach().cpu().numpy().astype(np.uint8)
    return img


def save_images(img_batch, dest_dir: Path, filenames: List[Path], inverse_normalize=False, parent_level=0, format='.jpg'):
    """save batch of tensors into image file, keep relvative path to parent_level
    if filename is /bar/foo/aa/x.jpg and parent level is 2, the output path will be dest_dir/foo/aa/x.jpg
    Args:
        inverse_normalize (bool): flag for wheather to inverse normalize tensors
    """
    N = img_batch.shape[0]
    if inverse_normalize:
        img_batch = (img_batch+1) /2
    for i in range(N):
        img = img_batch[i]
        file_path = Path(filenames[i])
        target_file = dest_dir / file_path.relative_to(file_path.parents[parent_level])
        target_file = target_file.with_suffix(format) # change saved image format
        if not target_file.parent.is_dir():
            target_file.parent.mkdir(parents=True)
        save_image(img, target_file)

def get_grad_norm(model):
    l2_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            l2_norm += param_norm**2
    
    return l2_norm**(0.5)

def mkdir(dir):
    path = Path(dir)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)


def tensor2numpy(tensor: torch.Tensor):
    "Convert tensor of (1, C, H, W) in [0, 1] to numpy array of (H, W, C) in [0, 255]"
    output = tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255 # (H, W, C)
    output = output.astype(np.uint8)
    return output

def pillow2tensor(img: Image):
    "convert PIL.Image to pytorch tensor"
    array = np.array(img, dtype=np.float32)
    return transforms.ToTensor()(array)

def log_images(cls: pl.LightningModule,
               images_dict: Dict[str, torch.Tensor],
               num: int = 4):
    for k, images in images_dict.items():
        image_show = torchvision.utils.make_grid((images[:num] + 1) / 2,
                                                 nrow=4,
                                                 normalize=False)  # to [0~1]
        cls.logger.experiment.add_image(k, image_show, cls.global_step)