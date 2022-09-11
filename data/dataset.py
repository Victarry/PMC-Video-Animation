import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Resize
from data.base import DistributedSamplerWrapper, ImageFolder, MergeDataset, MultiRandomSampler, MultiBatchDataset, MultiBatchSampler
from torchvision import transforms
from pathlib import Path


class Dataset(pl.LightningDataModule):
    def __init__(self,
                 root: str,
                 style: str ,
                 accelerator: str,
                 face_style: str,
                 sample_steps: list,
                 batch_size: int,
                 num_workers: int,
                 evaluate_temp=False, **kargs):
        super().__init__()
        self.root = Path(root)
        self.scene_style = style
        self.face_style = face_style
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_steps = sample_steps
        self.is_distributed = accelerator == 'ddp'
        self.dims = (3, 256, 256)
        self.evaluate_temp = evaluate_temp

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--face_style', default='pa_face')
        parser.add_argument('--sample_steps', default=[4, 1])
        return parser

    def setup(self, stage=None):
        if stage in ['fit', 'validate']:
            scenery_cartoon = ImageFolder((self.root / 'scenery_cartoon').as_posix(), transform=self.transform)
            scenery_photo = ImageFolder((self.root / 'scenery_photo').as_posix(), transform=self.transform)
            photo_val = ImageFolder((self.root / 'photo' / 'val').as_posix(), transform=self.transform, return_paths=True)
            self.scenery_dataset = MergeDataset(scenery_cartoon, scenery_photo)

            face_cartoon = ImageFolder(self.root / 'face_cartoon', transform=self.transform)
            face_photo = ImageFolder(self.root / 'face_photo', transform=self.transform)
            self.face_dataset = MergeDataset(face_cartoon, face_photo)

            self.train_dataset = MultiBatchDataset(self.scenery_dataset, self.face_dataset)
            self.train_sampler = MultiBatchSampler([
                MultiRandomSampler(self.scenery_dataset),
                MultiRandomSampler(self.face_dataset)],
                self.sample_steps, 
                self.batch_size
            )
            self.val_dataset = photo_val
        elif stage == 'test':
            def davis_crop(image):
                from torchvision.transforms.functional import crop
                return crop(image, 0, 0, 480, 848)

            davis_transfrom = transforms.Compose([
                transforms.Lambda(davis_crop),
                # read image and convert to tensor range in [0, 1]
                transforms.ToTensor(),
                # rescale the image value from [0, 1] to [-1, 1]
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            self.davis_dataset = ImageFolder(self.root/ 'DAVIS', return_paths=True, transform=davis_transfrom)
            self.test_dataset = ImageFolder(self.root / 'test_photo', transform=self.transform, return_paths=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False,
                      batch_size=1, num_workers=4)

    def test_dataloader(self):
        test_photo_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        if self.evaluate_temp:
            davis_loader = DataLoader(self.davis_dataset, batch_size=self.batch_size // 4, num_workers=self.num_workers)
            return test_photo_loader, davis_loader
        else:
            return test_photo_loader