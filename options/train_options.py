from .base_options import BaseOptions

def str2list(s):
    return [float(x) for x in s.split(',')]

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        parser = self.parser
        # set pretrained generator path
        parser.add_argument('--pretrained_init_model', type=str, help='loading pretained init generator model')
        parser.add_argument('--resume_from', type=str, default=None, help='model weight to resume training')
        # network saving and loading parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training hyperparameters 
        parser.add_argument('--use_schedule', action='store_true', help='flag to use learning rate schedule.')
        parser.add_argument('--epochs', type=int, default=101)
        parser.add_argument('--gan_mode', default='lsgan', choices=['vanilla', 'lsgan', 'wgan-gp']) 
        parser.add_argument('--init_epochs', type=int, default=10, help='number of epochs in the init stage')
        parser.add_argument('--init_lr', default=2e-4, help='learning rate for generator in init stage')
        parser.add_argument('--lrG', type=float, default=2e-5, help='learning rate for generator')
        parser.add_argument('--lrD', type=float, default=4e-5, help='learning rate for discriminator')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999)
        # Discriminator loss
        parser.add_argument('--d_adv_weight', type=float, default=300.0, help='Weight of adversarial loss for discriminator')
        parser.add_argument('--gp_weight', type=float, default=10, help='Weight ')
        parser.add_argument('--d_real_weight', type=float, default=1.2, help='Weight ')
        parser.add_argument('--d_fake_weigth', type=float, default=1.2, help='Weight ')
        parser.add_argument('--d_gray_weight', type=float, default=1.2, help='Weight ')
        parser.add_argument('--d_smooth_weight', type=float, default=0.8, help='Weight ')
        # Generator loss
        parser.add_argument('--g_adv_weight', type=float, default=3.0, help='Weight of adversarial loss for generator')
        parser.add_argument('--content_weight', type=float, default=1.2, help='Weight about VGG19 perceptural loss')# 1.5 for Hayao, 2.0 for Paprika, 1.2 for Shinkai
        parser.add_argument('--style_weight', type=float, default=0, help='Weight about style loss of gram matrix in VGG19')# 2.5 for Hayao, 0.6 for Paprika, 2.0 for Shinkai
        parser.add_argument('--color_weight', type=float, default=10., help='Weight about color reconstruction in yuv color space.') # 15. for Hayao, 50. for Paprika, 10. for Shinkai
        parser.add_argument('--tv_weight', type=float, default=1., help='Weight about total variation loss')# 1. for Hayao, 0.1 for Paprika, 1. for Shinkai
        # parser.add_argument('--spatial_weight', type=float, default=0, help='weight about spatial loss.')
        # parser.add_argument('--spatiotemporal_weight', type=float, default=1, help='weight about spatial-temporal loss.')
        parser.add_argument('--identity_loss', action='store_true', help='flag to use identity loss.')
        parser.add_argument('--temporal_weight', type=float, default=0, help='Weight about temporal loss.') # 10
        # Specific settings for Compound Regularization (proposed model: --data_sigma --data_w)
        parser.add_argument('--temporal_loss_type', default='STC', choices=['flow', 'STC', 'affine']) # STC(spatial-temporal correlative loss)
        parser.add_argument('--flow_type', default='random', help='using depth map to generate flow.')
        parser.add_argument('--data_sigma', default=True, type=bool, help='use noise in temporal loss')
        parser.add_argument('--data_w', default=True, type=bool, help='use warp in temporal loss')
        parser.add_argument('--data_noise_level', type=float, default=0.001, help='noise level in temporal loss')
        parser.add_argument('--data_motion_level', type=float, default=8, help='motion level in temporal loss')
        parser.add_argument('--data_shift_level', type=float, default=10, help='shift level in temporal loss')