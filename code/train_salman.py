# this file is based on code publicly available at
#   https://github.com/Hadisalman/smoothing-adversarial
# written by Hadi Salman.

import argparse
import time

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from architectures import ARCHITECTURES
from datasets import DATASETS
from third_party.smoothadv import Attacker, PGD_L2, DDN
from train_utils import AverageMeter, accuracy, log, test, requires_grad_
from train_utils import prologue

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
# Attack params
parser.add_argument('--attack', default='DDN', type=str, choices=['DDN', 'PGD'])
parser.add_argument('--epsilon', default=64.0, type=float)
parser.add_argument('--num-steps', default=10, type=int)
parser.add_argument('--warmup', default=1, type=int, help="Number of epochs over which "
                                                          "the maximum allowed perturbation increases linearly "
                                                          "from zero to args.epsilon.")
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors to use for finding adversarial examples. `m_train` in the paper.")
parser.add_argument('--no-grad-attack', action='store_true',
                    help="Choice of whether to use gradients during attack or do the cheap trick")

# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=256.0, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)

args = parser.parse_args()
if args.attack == 'PGD':
    mode = f"pgd_{args.epsilon}_{args.num_steps}_{args.warmup}"
elif args.attack == 'DDN':
    mode = f"ddn_{args.epsilon}_{args.num_steps}_{args.warmup}_{args.init_norm_DDN}_{args.gamma_DDN}"
else:
    raise Exception('Unknown attack')
args.outdir = f"logs/{args.dataset}/salman/{mode}/num_{args.num_noise_vec}/noise_{args.noise_sd}"

args.epsilon /= 256.0
args.init_norm_DDN /= 256.0


def main():
    train_loader, test_loader, criterion, model, optimizer, scheduler, \
    starting_epoch, logfilename, model_path, device, writer = prologue(args)

    if args.attack == 'PGD':
        print('Attacker is PGD')
        attacker = PGD_L2(steps=args.num_steps, device=device, max_norm=args.epsilon)
    elif args.attack == 'DDN':
        print('Attacker is DDN')
        attacker = DDN(steps=args.num_steps, device=device, max_norm=args.epsilon,
                       init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
    else:
        raise Exception('Unknown attack')

    for epoch in range(starting_epoch, args.epochs):
        attacker.max_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon / args.warmup])
        attacker.init_norm = np.min([args.epsilon, (epoch + 1) * args.epsilon / args.warmup])

        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd,
                                      attacker, device, writer)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, args.noise_sd, device, writer, args.print_freq)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, after - before,
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        # In PyTorch 1.1.0 and later, you should call `optimizer.step()` before `lr_scheduler.step()`.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step(epoch)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_path)


def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float,
          attacker: Attacker, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    requires_grad_(model, True)

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.repeat((1, args.num_noise_vec, 1, 1)).reshape(-1, *batch[0].shape[1:])
            batch_size = inputs.size(0)

            # augment inputs with noise
            noise = torch.randn_like(inputs, device=device) * noise_sd

            requires_grad_(model, False)
            model.eval()
            inputs = attacker.attack(model, inputs, targets,
                                     noise=noise, num_noise_vectors=args.num_noise_vec,
                                     no_grad=args.no_grad_attack)
            model.train()
            requires_grad_(model, True)

            noisy_inputs = inputs + noise

            targets = targets.unsqueeze(1).repeat(1, args.num_noise_vec).reshape(-1, 1).squeeze()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    if writer:
        writer.add_scalar('loss/train', losses.avg, epoch)
        writer.add_scalar('batch_time', batch_time.avg, epoch)
        writer.add_scalar('accuracy/train@1', top1.avg, epoch)
        writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
