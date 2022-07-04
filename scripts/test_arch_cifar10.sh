
srun -n 1 --cpus-per-task 20 python fully_test_arch.py \
--gpu_ids 0,1 \
--num_workers 16 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--arch arch_cifar10 \
--draw_arch False \
--checkpoint arch_train_cifar10 \
--genotypes_exp arch_cifar10 \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--num_eval_imgs 50000 \
--eval_batch_size 100 \
--exp_name arch_test_cifar10 \
--data_path /cache/datasets/cifar10 \
--checkpoint /home/datasets/Flythings3D/MEGAN/exps/cifar10_best