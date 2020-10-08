# Experiments

This document specifies how to replicate our results on MNIST. 
Results on other datasets, e.g., CIFAR-10 and ImageNet, can be reproduced in a similar way.

### Training

Please specify GPU number(s) before run each line by modifying `CUDA_VISIBLE_DEVICES=[num]`.

```
# Baseline: Gaussian (Cohen et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50
CUDA_VISIBLE_DEVICES=[num] python code/train_cohen.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00

# Baseline: SmoothAdv (Salman et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--attack PGD --epsilon 255 --num-steps 10 --warmup 10 --num-noise-vec 4
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--attack PGD --epsilon 255 --num-steps 10 --warmup 10 --num-noise-vec 4
CUDA_VISIBLE_DEVICES=[num] python code/train_salman.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--attack PGD --epsilon 255 --num-steps 10 --warmup 10 --num-noise-vec 4

# Baseline: Stability training (Li et al., 2019)
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.25 \
--lbd 2
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 0.50 \
--lbd 2
CUDA_VISIBLE_DEVICES=[num] python code/train_stab.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90 --noise 1.00 \
--lbd 2

# Baseline: MACER (Zhai et al., 2020)
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 16.0 --deferred
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 16.0 --deferred
CUDA_VISIBLE_DEVICES=[num] python code/train_macer.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 16 --beta 16.0 --margin 8.0 --lbd 6.0 --deferred

# Ours: Gaussian + Consistency (lbd=20) 
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--num-noise-vec 2 --lbd 20
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--num-noise-vec 2 --lbd 20
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 2 --lbd 20

# Ours: SmoothAdv + Consistency (lbd=5)
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.25 \
--num-noise-vec 4 --lbd 5 --adv-training --epsilon 255 --num-steps 10 --warmup 10
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 0.50 \
--num-noise-vec 4 --lbd 5 --adv-training --epsilon 255 --num-steps 10 --warmup 10
CUDA_VISIBLE_DEVICES=[num] python code/train.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 4 --lbd 5 --adv-training --epsilon 255 --num-steps 10 --warmup 10

```

### Certification

```
# Baseline: Gaussian (Cohen et al., 2019)
python code/certify.py mnist logs/mnist/cohen/noise_0.25/checkpoint.pth.tar 0.25 \
test/certify/mnist/cohen/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/cohen/noise_0.5/checkpoint.pth.tar 0.50 \
test/certify/mnist/cohen/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/cohen/noise_1.0/checkpoint.pth.tar 1.00 \
test/certify/mnist/cohen/noise_1.00 --N=100000 --skip=1

# Baseline: SmoothAdv (Salman et al., 2019)
python code/certify.py mnist logs/mnist/salman/pgd_255.0_10_10/num_4/noise_0.25/checkpoint.pth.tar 0.25 \
test/certify/mnist/salman/pgd_255.0_10_10/num_4/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/salman/pgd_255.0_10_10/num_4/noise_0.5/checkpoint.pth.tar 0.5 \
test/certify/mnist/salman/pgd_255.0_10_10/num_4/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/salman/pgd_255.0_10_10/num_4/noise_1.0/checkpoint.pth.tar 1.0 \
test/certify/mnist/salman/pgd_255.0_10_10/num_4/noise_1.00 --N=100000 --skip=1

# Baseline: Stability training (Li et al., 2019)
python code/certify.py mnist logs/mnist/stab/lbd_2.0/noise_0.25/checkpoint.pth.tar 0.25 \
test/mnist/stab/lbd_2/noise_0.25 --N 100000 --skip 1
python code/certify.py mnist logs/mnist/stab/lbd_2.0/noise_0.5/checkpoint.pth.tar 0.5 \
test/mnist/stab/lbd_2/noise_0.5 --N 100000 --skip 1
python code/certify.py mnist logs/mnist/stab/lbd_2.0/noise_1.0/checkpoint.pth.tar 1.0 \
test/mnist/stab/lbd_2/noise_1.0 --N 100000 --skip 1

# Baseline: MACER (Zhai et al., 2020)
python code/certify.py mnist logs/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/noise_0.25/checkpoint.pth.tar 0.25 \
test/certify/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/noise_0.5/checkpoint.pth.tar 0.5 \
test/certify/mnist/macer_deferred30/num_16/lbd_16.0/gamma_8.0/beta_16.0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/macer_deferred30/num_16/lbd_6.0/gamma_8.0/beta_16.0/noise_1.0/checkpoint.pth.tar 1.0 \
test/certify/mnist/macer_deferred30/num_16/lbd_6.0/gamma_8.0/beta_16.0/noise_1.00 --N=100000 --skip=1

# Ours: Gaussian + Consistency (lbd=20)
python code/certify.py mnist logs/mnist/consistency/cohen/num_2/lbd_20.0/noise_0.25/checkpoint.pth.tar 0.25 \
test/certify/mnist/consistency/cohen/num_2/lbd_20.0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/consistency/cohen/num_2/lbd_20.0/noise_0.5/checkpoint.pth.tar 0.5 \
test/certify/mnist/consistency/cohen/num_2/lbd_20.0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/consistency/cohen/num_2/lbd_20.0/noise_1.0/checkpoint.pth.tar 1.0 \
test/certify/mnist/consistency/cohen/num_2/lbd_20.0/noise_1.00 --N=100000 --skip=1

# Ours: SmoothAdv + Consistency (lbd=5)
python code/certify.py mnist logs/mnist/consistency/salman_255.0_10_10/num_4/lbd_5.0/noise_0.25/checkpoint.pth.tar 0.25 \
test/certify/mnist/consistency/salman_255.0_10_10/num_4/lbd_5.0/noise_0.25 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/consistency/salman_255.0_10_10/num_4/lbd_5.0/noise_0.5/checkpoint.pth.tar 0.5 \
test/certify/mnist/consistency/salman_255.0_10_10/num_4/lbd_5.0/noise_0.50 --N=100000 --skip=1
python code/certify.py mnist logs/mnist/consistency/salman_255.0_10_10/num_4/lbd_5.0/noise_1.0/checkpoint.pth.tar 1.0 \
test/certify/mnist/consistency/salman_255.0_10_10/num_4/lbd_5.0/noise_1.00 --N=100000 --skip=1
```
