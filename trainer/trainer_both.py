import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import random
import heapq
from utils.sort import CARS_NSGA
from utils.utils import count_parameters_in_MB
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from archs.fully_super_network import Generator, Discriminator
logger = logging.getLogger(__name__)


class BothTrainer():
    def __init__(self, args, gen_net, dis_net, gen_optimizer, dis_optimizer, train_loader, gan_alg):
        self.args = args
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.train_loader = train_loader
        self.gan_alg = gan_alg
        # self.dis_genotype = dis_genotype
        if args.warmup == 0:
            self.gen_genotypes = np.stack(
                [gan_alg.search_mutate() for i in range(args.num_individual)], axis=0)
            self.dis_genotypes = np.stack(
                [gan_alg.search_mutate_dis() for i in range(args.num_individual)], axis=0)
        else:
            self.gen_genotypes = np.stack([gan_alg.search()
                                       for i in range(args.num_individual)], axis=0)
            self.dis_genotypes = np.stack([gan_alg.search_dis()
                                       for i in range(args.num_individual)], axis=0)

    def train(self, epoch, writer_dict, schedulers=None):
        writer = writer_dict['writer']
        gen_step = 0
        
        # train mode
        gen_net = self.gen_net.train()
        dis_net = self.dis_net.train()
        
        for iter_idx, (imgs, _) in enumerate(tqdm(self.train_loader)):
            i = np.random.randint(0, self.args.num_individual, 1)[0]
            if epoch <= self.args.warmup:
                genotype_G = self.gan_alg.search()
                genotype_D = self.gan_alg.search_dis()
                    # logits = self.trainer.model.forward_random(input)
            else:
                genotype_G = self.gen_genotypes[i]
                genotype_D = self.dis_genotypes[i]
            global_steps = writer_dict['train_global_steps']

            real_imgs = imgs.type(torch.cuda.FloatTensor)
            
            # sample noise
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (imgs.shape[0], self.args.latent_dim)))
            # train D
            self.dis_optimizer.zero_grad()
            real_validity = dis_net(real_imgs, genotype_D)
            
            fake_imgs = gen_net(z, genotype_G).detach()
            assert fake_imgs.size() == real_imgs.size()
            fake_validity = dis_net(fake_imgs, genotype_D)
            # Hinge loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            self.dis_optimizer.step()
            writer.add_scalar('d_loss', d_loss.item(), global_steps)
            # train G
            if global_steps % self.args.n_critic_search == 0:
                self.gen_optimizer.zero_grad()

                # sample noise
                gen_z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.gen_bs, self.args.latent_dim)))

                gen_imgs = gen_net(gen_z, genotype_G)
                fake_validity = dis_net(gen_imgs, genotype_D)
                # Hinge loss
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                self.gen_optimizer.step()
                # learning rate
                if schedulers:
                    gen_scheduler, dis_scheduler = schedulers
                    g_lr = gen_scheduler.step(global_steps)
                    d_lr = dis_scheduler.step(global_steps)
                    writer.add_scalar('LR/g_lr', g_lr, global_steps)
                    writer.add_scalar('LR/d_lr', d_lr, global_steps)
                writer.add_scalar('g_loss', g_loss.item(), global_steps)
                gen_step += 1
                i = np.random.randint(0, self.args.num_individual, 1)[0]
                if epoch <= self.args.warmup:
                    genotype_G = self.gan_alg.search()
                    genotype_D = self.gan_alg.search_dis()
                    # logits = self.trainer.model.forward_random(input)
                else:
                    genotype_G = self.gen_genotypes[i]
                    genotype_D = self.dis_genotypes[i]
            # verbose
            if gen_step and iter_idx % self.args.print_freq == 0:
                tqdm.write(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                    (epoch, self.args.max_epoch_D, iter_idx % len(self.train_loader), len(self.train_loader), d_loss.item(), g_loss.item()))
            writer_dict['train_global_steps'] = global_steps + 1

    def search_evol_arch(self, epoch, fid_stat):
        offsprings = self.gen_offspring(self.gen_genotypes)
        offsprings_dis = self.gen_offspring_dis(self.dis_genotypes)
        gen_genotypes = np.concatenate((self.gen_genotypes, offsprings), axis=0)
        dis_genotypes = np.concatenate((self.dis_genotypes, offsprings_dis), axis=0)
        is_values, fid_values, params = np.zeros(len(gen_genotypes)), np.zeros(
            len(gen_genotypes)), np.zeros(len(gen_genotypes))
        keep_N, selected_N = len(offsprings), self.args.num_selected
        for idx, genotype_G in enumerate(tqdm(gen_genotypes)):
            is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
            param_szie = count_parameters_in_MB(
                Generator(self.args, genotype_G))
            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_szie
        """
        indexs = heapq.nlargest(self.args.num_individual, range(len(values)), values.__getitem__)
        self.genotypes = genotypes[indexs]
        max_index = values.index(max(values))
        """
        logger.info(f'mean_IS_values: {np.mean(is_values)}, mean_FID_values: {np.mean(fid_values)},@ epoch {epoch}.')
        obj = [fid_values, params]
        keep, selected = CARS_NSGA(is_values, obj, keep_N), CARS_NSGA(
            is_values, obj, selected_N)
        for i in selected:
            logger.info(
                f'genotypes_{i}, IS_values: {is_values[i]}, FID_values: {fid_values[i]}, param_szie: {params[i]}|| @ epoch {epoch}.')
        self.gen_genotypes = gen_genotypes[keep]
        self.dis_genotypes = dis_genotypes[keep]
        return gen_genotypes[selected]

    def validate(self, genotype_G, fid_stat):
        #writer = writer_dict['writer']
        #global_steps = writer_dict['valid_global_steps']
        # eval mode
        gen_net = self.gen_net.eval()
        # get fid and inception score
        fid_buffer_dir = os.path.join(
            self.args.path_helper['sample_path'], 'fid_buffer')
        os.makedirs(fid_buffer_dir, exist_ok=True)
        eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
        img_list = list()
        for iter_idx in tqdm(range(eval_iter), desc='sample images'):
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
            # generate a batch of images
            gen_imgs = gen_net(z, genotype_G).mul_(127.5).add_(127.5).clamp_(
                0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            for img_idx, img in enumerate(gen_imgs):
                file_name = os.path.join(
                    fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
                imsave(file_name, img)
            img_list.extend(list(gen_imgs))
        # get inception score
        logger.info('=> calculate inception score')
        mean, std = get_inception_score(img_list)
        # get fid score
        logger.info('=> calculate fid score')
        fid_score = calculate_fid_given_paths(
            [fid_buffer_dir, fid_stat], inception_path=None)
        return mean, std, fid_score

    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.5:
                alphas_c = self.mutation(
                    alphas[np.random.randint(0, alphas.shape[0])])
            else:
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                while(a == b):
                    a, b = np.random.randint(
                        0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                alphas_c = self.crossover(alphas[a], alphas[b])
            # else:
            #     alphas_c = self.gan_alg.search()
            if not self.gan_alg.judge_repeat(alphas_c):
                offsprings.append(alphas_c)
        # offsprings = torch.cat([offspring.unsqueeze(0) for offspring in offsprings], dim=0)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        """Judge if two individuals are the same.
        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param new_alphas: An individual
        :type new_alphas: nn.Tensor
        :return: True or false
        :rtype: nn.Tensor
        """
        diff = np.reshape(np.absolute(
            alphas - np.expand_dims(new_alphas, axis=0)), (alphas.shape[0], -1))
        diff = np.sum(diff, axis=1)
        return np.sum(diff == 0)

    def crossover(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        # alpha a
        new_alphas = alphas_a.copy()
        # alpha b
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        while(new_alphas[layer][index] == alphas_a[layer][index]):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            new_alphas[layer][index] = alphas_b[layer][index]
            if index >= 2 and index < 4 and new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
            elif index >= 4 and new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
        return new_alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for An individual."""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        if index < 2:
            new_alphas[layer][index] = random.randint(0, 2)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 2)
        elif index >= 2 and index < 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        return new_alphas

    def select_best(self, epoch):
        values = []
        for genotype_G in self.genotypes:
            ssim_value, psnr_value = self.validate(genotype_G)
            #logger.info(f'ssim_value: {ssim_value}, psnr_value: {psnr_value}|| @ epoch {epoch}.')
            values.append(ssim_value)
        max_index = values.index(max(values))
        return self.genotypes[max_index]

    def gen_offspring_dis(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.5:
                alphas_c = self.mutation_dis(
                    alphas[np.random.randint(0, alphas.shape[0])])
            # elif rand < 0.5:
            else:
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                while(a == b):
                    a, b = np.random.randint(
                        0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                alphas_c = self.crossover_dis(alphas[a], alphas[b])
            # else:
            #     alphas_c = self.gan_alg.search_dis()
            if not self.gan_alg.judge_repeat_dis(alphas_c):
                offsprings.append(alphas_c)
        # offsprings = torch.cat([offspring.unsqueeze(0) for offspring in offsprings], dim=0)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def gen_offspring_dis_epoch(self, alphas, epoch, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.25 + (0.25*epoch/self.args.max_epoch_D):
                alphas_c = self.mutation_dis(
                    alphas[np.random.randint(0, alphas.shape[0])])
            elif rand < 0.5 + (0.5*epoch/self.args.max_epoch_D):
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                while(a == b):
                    a, b = np.random.randint(
                        0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                alphas_c = self.crossover_dis(alphas[a], alphas[b])
            else:
                alphas_c = self.gan_alg.search_dis()
            if not self.gan_alg.judge_repeat_dis(alphas_c):
                offsprings.append(alphas_c)
        # offsprings = torch.cat([offspring.unsqueeze(0) for offspring in offsprings], dim=0)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def crossover_dis(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        # alpha a
        new_alphas = alphas_a.copy()
        # alpha b
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        while(new_alphas[layer][index] == alphas_a[layer][index]):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            new_alphas[layer][index] = alphas_b[layer][index]
            if index >= 1 and index < 3 and new_alphas[layer][1] == 0 and new_alphas[layer][2] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
            elif index >= 3 and new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
        return new_alphas

    def mutation_dis(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        if index == 0:
            new_alphas[layer][index] = random.randint(1, 6)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(1, 6)
        elif index >= 1 and index < 3:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][1] == 0 and new_alphas[layer][2] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 3 and index < 5:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        if index == 5:
            new_alphas[layer][index] = random.randint(-1, 5)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(-1, 5)
        if index == 6:
            new_alphas[layer][index] = random.randint(0, 5)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 5)
        return new_alphas


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * \
                (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
