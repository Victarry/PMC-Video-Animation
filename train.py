import time
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from options.train_options import TrainOptions
import models
import data

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model_cls = models.find_model_using_name(opt.model)
    data_cls = data.find_dataset_using_name(opt.dataset_mode)
    logger = TensorBoardLogger(opt.logging_dir, name='', version='events')

    datamodule = data_cls(**vars(opt))

    # initial training stage
    if opt.resume_from is None:
        if opt.pretrained_init_model is None:
            checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(
                opt.logging_dir, 'checkpoints'),
                                                save_top_k=-1,
                                                filename='pretrain-{epoch:02d}',
                                                save_weights_only=True)
            pre_trainer = pl.Trainer(logger=logger,
                                    max_epochs=opt.init_epochs,
                                    gpus=opt.gpus,
                                    accelerator=opt.accelerator,
                                    callbacks=[checkpoint_callback])
            pretrain_model = model_cls.pretrained_model()(**vars(opt))
            pre_trainer.fit(pretrain_model, datamodule)
            while True:
                if Path(checkpoint_callback.best_model_path).exists():
                    model = model_cls.load_from_checkpoint(
                        checkpoint_callback.best_model_path)
                    break
                else:
                    time.sleep(10)
        else:
            model = model_cls.load_from_checkpoint(opt.pretrained_init_model,
                                                strict=False,
                                                **vars(opt))
    else:
        model = model_cls.load_from_checkpoint(opt.resume_from, strict=False, **vars(opt))

    # save checkpoint every 5 epoch
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(
        opt.logging_dir, 'checkpoints'),
                                          filename='train-{epoch:02d}-{step}',
                                          every_n_val_epochs=1,
                                          save_top_k=-1)
    profiler = pl.profiler.SimpleProfiler(filename='profile')
    trainer = pl.Trainer(log_every_n_steps=5,
                         logger=logger,
                         max_epochs=opt.epochs,
                         gpus=opt.gpus,
                         val_check_interval= 8000 // opt.batch_size,
                         accelerator=opt.accelerator,
                         callbacks=[checkpoint_callback],
                         replace_sampler_ddp=False,
                         profiler=profiler)
    trainer.fit(model, datamodule=datamodule)
