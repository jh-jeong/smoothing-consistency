# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn

from archs.cifar_resnet import resnet as resnet_cifar
from archs.lenet import LeNet
from archs.densenet import densenet40

from datasets import get_normalize_layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lenet - the classic LeNet for MNIST
# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
# cifar_densenet40 - a 40-layer densely-connected network for CIFAR
ARCHITECTURES = ["lenet", "resnet50", "cifar_resnet20", "cifar_resnet110", "cifar_densenet40"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False))
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10)
    elif arch == "cifar_densenet40":
        model = densenet40(num_classes=10)
    elif arch == "lenet":
        model = LeNet(num_classes=10)
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model).to(device)
