# @Date    : 2020-12-30
# @Author  : Guohao YIng


from torch import nn
from archs.basic_blocks_search import Cell, DisCell, OptimizedDisBlock


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        if args.dataset == 'cifar10':
            # for lower resolution (32 * 32) dataset CIFAR-10
            self.base_latent_dim = args.latent_dim // 3
        else:
            # for higher resolution (48 * 48) dataset STL-10
            self.base_latent_dim = args.latent_dim // 2
        self.l1 = nn.Linear(self.base_latent_dim,
                            (self.bottom_width ** 2) * args.gf_dim)
        self.l2 = nn.Linear(self.base_latent_dim, ((
            self.bottom_width * 2) ** 2) * args.gf_dim)
        if args.dataset == 'cifar10':
            self.l3 = nn.Linear(self.base_latent_dim, ((
                self.bottom_width * 4) ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim), nn.ReLU(), nn.Conv2d(
                args.gf_dim, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z, genotypes):
        h = self.l1(z[:, :self.base_latent_dim])\
            .view(-1, self.ch, self.bottom_width, self.bottom_width)

        n1 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2])\
            .view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)
        if self.args.dataset == 'cifar10':
            n2 = self.l3(z[:, self.base_latent_dim * 2:])\
                .view(-1, self.ch, self.bottom_width * 4, self.bottom_width * 4)

        h1_skip_out, h1 = self.cell1(h, genotype=genotypes[0])
        h2_skip_out, h2 = self.cell2(
            h1+n1, (h1_skip_out, ), genotype=genotypes[1])
        _, h3 = self.cell3(h2+n2, (h1_skip_out, h2_skip_out), genotype=genotypes[2])
        output = self.to_rgb(h3)

        return output

class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisCell(args, self.ch, self.ch, activation=activation)
        self.block3 = DisCell(args, self.ch, self.ch, activation=activation)
        self.block4 = DisCell(args, self.ch, self.ch, activation=activation)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x, genotypes):
        h = x
        h = self.block1(h)
        h = self.block2(h, genotypes[0])
        h = self.block3(h, genotypes[1])
        h = self.block4(h, genotypes[2])
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output