import argparse
import os
import torch
from utils import util
from rich.console import Console
import models
import data

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.

    NOTE: this will overrides logging dir as subdirectory named exp_name_{style}
    """

    def __init__(self):
        parser = argparse.ArgumentParser()
        # basic parameters
        parser.add_argument('--model', default='whiteboxgan')
        parser.add_argument('--dataset_mode')
        parser.add_argument('--accelerator', default='ddp')
        parser.add_argument('--gpus', default=1, help='number of gpus to train on')
        parser.add_argument('--exp_name', required=True, help='name of the experiments. It decides where to store samples and checkpoints.')
        parser.add_argument('--logging_dir', type=str, default='./logs', help='logging directory, including hparams, val_results, test_results,  ')
        # model related parameters
        parser.add_argument('--netG', default='anime', help='type of generator. resnet_generator | unet_generator | anime_generator')
        parser.add_argument('--res_blocks', type=int, default=8, help='number of resblocks in resnet generator')
        parser.add_argument('--no_sn', action='store_true', help='usage of sepectral normalization in discriminator')
        parser.add_argument('--g_norm', default='layer', help='norm layer type of generator.')
        parser.add_argument('--d_norm', default='layer', help='norm layer type of discriminator.')
        # dataset parameters
        parser.add_argument('--root', default='datasets')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--num_workers', type=int, default=8, help='threads for loading data')
        parser.add_argument('--comment', help='other comment to record')
        self.is_train = False
        self.parser = parser
        self.console = Console()


    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [logging_dir] / [exp_name] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        # check if current stage of exp_name has already been runned
        opt.exp_name = f'{opt.exp_name}'
        hparams_path = os.path.join(opt.logging_dir, opt.exp_name, '{}_opt.txt'.format(opt.phase))
        opt.logging_dir= os.path.join(opt.logging_dir, opt.exp_name)
        if os.environ.get("LOCAL_RANK", None) is None or os.environ["LOCAL_RANK"] == 0:
            print(message)
            if os.path.exists(hparams_path):
                self.console.print('The exp_name have been used.', style='bold red')
                self.console.print('Please press Return continue or press Ctrl-C to exit')
                s = input()
                if s == '':
                    pass
                    self.console.print('Overrides current exp logging dir.', style='bold red')
                else:
                    exit(0) 
            # Setting logging dir and exp_name

            util.mkdir(opt.logging_dir)
            # save to the disk
            with open(hparams_path, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        # assert gather options only called once
        assert getattr(self, 'initialized', None) == None
        self.initialized = True

        parser = self.parser
        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.is_train)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self, args=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)
        self.opt = opt
        return self.opt
