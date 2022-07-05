
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset type')
    parser.add_argument('--img_size', type=int, default=32, help='image size, 32 for cifar10, 48 for stl10')
    parser.add_argument('--bottom_width', type=int, default=4, help='init resolution, 4 for cifar10, 6 for stl10')
    parser.add_argument('--channels', type=int, default=3, help='image channels')
    parser.add_argument('--data_path', type=str, default='/cache/datasets/cifar10', help='dataset path')
    
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--gpu_ids', type=str, default='0', help='visible GPU ids')
    parser.add_argument('--num_workers', type=int, default=16, help='number of cpu threads to use during batch generation')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    
    # train
    parser.add_argument('--arch', type=str, default='arch_cifar10', help='architecture name')
    # parser.add_argument('--arch_D', type=str, help='architecture name of D')
    parser.add_argument('--max_epoch_G', type=int, default=200, help='max number of epoch for training G')
    parser.add_argument('--max_iter_G', type=int, default=None, help='max number of iteration for training G')
    parser.add_argument('--n_critic_search', type=int, default=1, help='number of training steps for discriminator per iter in search')
    parser.add_argument('--n_critic_search_dis', type=int, default=5, help='number of training steps for discriminator per iter in search')
    parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
    parser.add_argument('--gen_bs', type=int, default=256, help='batche size of G')
    parser.add_argument('--dis_bs', type=int, default=256, help='batche size of D')
    parser.add_argument('--gf_dim', type=int, default=256, help='base channel-dim of G')
    parser.add_argument('--df_dim', type=int, default=128, help='base channel-dim of D')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for D')
    parser.add_argument('--lr_decay', action='store_true', help='learning rate decay or not')
    parser.add_argument('--beta1', type=float, default=0.0, help='decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.9, help='decay of first order momentum of gradient')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='init type')
    parser.add_argument('--d_spectral_norm', type=str2bool, default=True,
                        help='add spectral_norm on discriminator or not')
    parser.add_argument('--g_spectral_norm', type=str2bool, default=False,
                        help='add spectral_norm on generator or not')
    parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
    
    # val
    parser.add_argument('--print_freq', type=int, default=50, help='interval between each verbose')
    parser.add_argument('--val_freq', type=int, default=20, help='interval between each validation')
    # parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument('--num_eval_imgs', type=int, default=5000)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    # search
    parser.add_argument('--derived_start_epoch', type=int, default=0, help='')
    parser.add_argument('--derived_max_epoch', type=int, default=None, help='')
    parser.add_argument('--derived_epoch_interval', type=int, default=None, help='')
    parser.add_argument('--tau_max', type=float, default=5, help='max tau for gumbel softmax')
    parser.add_argument('--tau_min', type=float, default=0.1, help='min tau for gumbel softmax')
    parser.add_argument('--gumbel_softmax', type=str2bool, default=False, help='use gumbel softmax or not')
    parser.add_argument('--amending_coefficient', type=float, default=0.1, help='')
    parser.add_argument('--derive_freq', type=int, default=1, help='')
    parser.add_argument('--derive_per_epoch', type=int, default=0, help='number of derive per epoch')
    parser.add_argument('--draw_arch', type=str2bool, default=True, help='visualize the searched architecture or not')
    parser.add_argument('--early_stop', type=str2bool, default=False, help='use early stop strategy or not')
    parser.add_argument('--genotypes_exp', type=str, help='ues genotypes of the experiment')
    parser.add_argument('--warmup', type=int, default=40, help='epochs before EA')
    parser.add_argument('--num_individual', type=int, default=32, help='numbers of  EA individual')
    parser.add_argument('--num_selected', type=int, default=8, help='numbers of  selected individual')
    parser.add_argument('--ga_interval', type=int, default=10, help='numbers of  EA interval')
    parser.add_argument('--epoch_generator', type=int, default=200, help='numbers of epoch for generator')
    parser.add_argument('--epoch_both', type=int, default=400, help='numbers of epoch for generator')
    parser.add_argument('--epoch_discriminator', type=int, default=200, help='numbers of epoch for discriminator')
    opt = parser.parse_args()
    return opt