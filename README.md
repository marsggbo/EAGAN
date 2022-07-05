<div align="center">

<h1>(ECCV2022) EAGAN: Efficient Two-stage Evolutionary Architecture Search for GANs</h1>

<div>
    Guohao Ying, Xin He, Bin Gao, Bo Han, and Xiaowen Chu.
</div>

</div>

## Introduction
This is the official implementation of *EAGAN: Efficient Two-stage Evolutionary Architecture Search for GANs*. We introduce a novel NAS framework, namely **EAGAN**, to alleviate the instability when searching GANs. Our EAGAN decouples the search into two stages, where stage-1 searches G with a fixed D and adopts the many-to-one training strategy, and stage-2 searches D with the optimal G found in stage-1 and adopts the one-to-one training strategy and the weight-resetting strategy to enhance the stability of GAN training.


The framework of the proposed method:

<div align="center">
<img src='./images/eagan_pipeline.png' width='100%'>
</div>



## Installation

### Download
```
git clone https://github.com/marsggbo/EAGAN
cd EAGAN
```


### Environment
We recommend using Anaconda to manage the python environment:
```
conda create -n EAGAN python=3.8
conda activate EAGAN
pip install -r requirements.txt
```


### Datasets

In our work, we use CIFAR-10 and STL-10 datasets. The default path for both datasets is `~/datasets/cifar10` and `~/datasets/stl10`. Therefore, you should move or download both datasets in this path.

## Searching

### 1. Search generator

```bash
bash ./scripts/train_search_gen.sh
```

or run the following commands

```bash
python search_gen_arch.py \
--gpu_ids 0 \
--num_workers 16 \
--gen_bs 80 \
--dis_bs 40 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_epoch_G 120 \
--n_critic 1 \
--arch arch_cifar10 \
--draw_arch False \
--genotypes_exp arch_train_cifar10_2021_11_04_16_56_26 \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0002 \
--d_lr 0.00004 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 5000 \
--exp_name arch_train_cifar10 \
--data_path ~/datasets/cifar10
```

- `genotypes_exp`: specify the genotype path of initial discriminator. 
  -  `exps/cifar10_D1.npy` is used by default.
  -  if you want to specify a different initial discriminator, you can modify `dis_genotype` variable in `search_gen_arch.py`, e.g., `dis_genotype=np.load('./exps/cifar10_D2.npy')`

### 2. Search discriminator

```bash
bash ./scripts/train_search_dis.sh
```


## Fully train

### 1. Fully train GAN on cifar10

```bash
bash ./scripts/train_arch_cifar10.sh
```

or run the following commands

```bash
python fully_train_arch.py \
--gpu_ids 0 \
--num_workers 16 \
--gen_bs 80 \
--dis_bs 40 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_epoch_G 120 \
--n_critic 5 \
--arch arch_cifar10 \
--draw_arch False \
--genotypes_exp cifar10_D1.npy \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 50000 \
--exp_name arch_train_cifar10 \
--data_path ~/datasets/cifar10
```


### 2. Fully train GAN on stl10

```bash
bash ./scripts/train_arch_stl10.sh
```

## 3. Test

```bash
bash ./scripts/test_arch_cifar10.sh
bash ./scripts/test_arch_stl10.sh
```


## Citation
If you find this work useful for your research, please kindly cite our paper:
```
@inproceedings{ECCV2022EAGAN,
  title={EAGAN: Efficient Two-stage Evolutionary Architecture Search for GANs},
  author={Guohao Ying, Xin He, Bin Gao, Bo Han, and Xiaowen Chu.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Acknowledgements

Thanks NVIDIA AI TECHNOLOGY CENTER (NVAITC) for providing GPU clusters to support our work.

[//]: <## Acknowledgements>
