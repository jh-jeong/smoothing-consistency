# this file is based on code publicly available at
#   https://github.com/locuslab/smoothing
# written by Jeremy Cohen.

import argparse
import time

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from architectures import ARCHITECTURES
from datasets import DATASETS, get_num_classes
from train_utils import log, test
from train_utils import prologue
from third_party.macer import macer_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--id', default=None, type=int,
                    help='experiment id, `randint(10000)` if None')

#####################
# Options added by Salman et al. (2019)
parser.add_argument('--resume', action='store_true',
                    help='if true, tries to resume training from existing checkpoint')
parser.add_argument('--pretrained-model', type=str, default='',
                    help='Path to a pretrained model')

#####################
# MACER-specific
parser.add_argument('--num-noise-vec', default=16, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
parser.add_argument('--beta', default=16.0, type=float)
parser.add_argument('--lbd', default=16.0, type=float)
parser.add_argument('--margin', default=8.0, type=float)
parser.add_argument('--deferred', action='store_true',
                    help='if true, MACER is applied after the first learning rate drop')


args = parser.parse_args()
if args.deferred:
    mode = f"macer_deferred{args.lr_step_size}"
else:
    mode = f"macer"
args.outdir = f"logs/{args.dataset}/{mode}/num_{args.num_noise_vec}/lbd_{args.lbd}/gamma_{args.margin}/beta_{args.beta}/noise_{args.noise_sd}"


def main():
    train_loader, test_loader, criterion, model, optimizer, scheduler, \
    starting_epoch, logfilename, model_path, device, writer = prologue(args)

    for epoch in range(starting_epoch, args.epochs):
        before = time.time()
        train_loss = train(train_loader, model, optimizer, epoch, args.noise_sd, device, writer)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, args.noise_sd, device, writer, args.print_freq)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, 0.0, test_loss, test_acc))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step(epoch)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)


def train(loader: DataLoader, model: torch.nn.Module, optimizer: Optimizer,
          epoch: int, noise_sd: float, device: torch.device, writer=None):
    # switch to train mode
    model.train()

    lbd = args.lbd
    if args.deferred and epoch <= args.lr_step_size:
        lbd = 0

    cl, rl = macer_train(sigma=noise_sd, lbd=lbd, gauss_num=args.num_noise_vec,
                         beta=args.beta, gamma=args.margin,
                         num_classes=get_num_classes(args.dataset),
                         model=model, trainloader=loader, optimizer=optimizer, device=device)

    writer.add_scalar('loss/train', cl, epoch)
    writer.add_scalar('loss/robust', rl, epoch)

    return cl


if __name__ == "__main__":
    main()
