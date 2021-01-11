# Consistency Regularization for Certified Robustness of Smoothed Classifiers (NeurIPS2020)

This repository contains code for the paper
**"Consistency Regularization for Certified Robustness of Smoothed Classifiers"** 
by [Jongheon Jeong](https://sites.google.com/view/jongheonj) and [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html). 


## Dependencies
```
conda create -n smoothing-consistency python=3
conda activate smoothing-consistency

# IMPORTANT: Please make sure `pytorch != 1.4.0`
#   Currently, our code is not compatible to `pytorch == 1.4.0`;
#   See more details at `https://github.com/pytorch/pytorch/issues/32395`.
# Below is for linux, with CUDA 10; see https://pytorch.org/ for the correct command for your system
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch 

conda install scipy pandas statsmodels matplotlib seaborn
pip install setGPU tensorboardX
```

## Scripts

### Training Scripts

The main script is `train_consistency.py`; We also provide training scripts 
to reproduce other baseline methods in `train_*.py`, as listed in what follows:

| File | Description |
| ------ | ------ |
| [train_consistency.py](code/train_consistency.py) |  The main script; Consistency regularization |
| [train_cohen.py](code/train_cohen.py) | Gaussian augmentation (Cohen et al., 2019) |
| [train_salman.py](code/train_salman.py) | SmoothAdv (Salman et al., 2019) |
| [train_stab.py](code/train_stab.py) | Stability training (Li et al., 2019) |
| [train_macer.py](code/train_macer.py) | MACER (Zhai et al., 2020) |

The sample scripts below demonstrate how to run `train.py` with Gaussian training and SmoothAdv.
Notice that SmoothAdv training is enabled by simply passing `--adv-training` option to the script. 
One can modify `CUDA_VISIBLE_DEVICES` to further specify GPU number(s) to work on.

```
# Consistency regularization (lbd=5) with Gaussian augmentation (Cohen et al., 2019)
CUDA_VISIBLE_DEVICES=0 python code/train_consistency.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 2 --lbd 5

# Consistency regularization (lbd=1) with SmoothAdv (Salman et al., 2019)
CUDA_VISIBLE_DEVICES=0 python code/train_consistency.py mnist lenet --lr 0.01 --lr_step_size 30 --epochs 90  --noise 1.00 \
--num-noise-vec 2 --lbd 1 --adv-training --epsilon 255 --num-steps 2 --warmup 10
```

For a more detailed instruction to reproduce our experiments, see [`EXPERIMENTS.MD`](EXPERIMENTS.MD).

### Testing Scripts

All the testing scripts is originally from https://github.com/locuslab/smoothing:

* The script [certify.py](code/certify.py) certifies the robustness of a smoothed classifier.  For example,

```python code/certify.py mnist model_output_dir/checkpoint.pth.tar 0.50 certification_output --alpha 0.001 --N0 100 --N 100000```

will load the base classifier saved at `model_output_dir/checkpoint.pth.tar`, smooth it using noise level &sigma;=0.50,
and certify the MNIST test set with parameters `N0=100`, `N=100000`, and `alpha=0.001`.

* The script [predict.py](code/predict.py) makes predictions using a smoothed classifier.  For example,

```python code/predict.py mnist model_output_dir/checkpoint.pth.tar 0.50 prediction_outupt --alpha 0.001 --N 1000```

will load the base classifier saved at `model_output_dir/checkpoint.pth.tar`, smooth it using noise level &sigma;=0.50,
and classify the MNIST test set with parameters `N=1000` and `alpha=0.001`.

* The script [analyze.py](code/analyze.py) contains some useful classes and functions to analyze the result data 
from [certify.py](code/certify.py) or [predict.py](code/predict.py).
