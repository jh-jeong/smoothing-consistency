# this file is copied from
#   https://github.com/RuntianZ/macer
# originally written by Runtian Zhai.

'''
MACER Algorithm
MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius
ICLR 2020 Submission
'''

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


def macer_train(sigma, lbd, gauss_num, beta, gamma, num_classes,
                    model, trainloader, optimizer, device):
  m = Normal(torch.tensor([0.0]).to(device),
             torch.tensor([1.0]).to(device))

  cl_total = 0.0
  rl_total = 0.0
  input_total = 0

  for _, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    input_size = len(inputs)
    input_total += input_size

    new_shape = [input_size * gauss_num]
    new_shape.extend(inputs[0].shape)
    inputs = inputs.repeat((1, gauss_num, 1, 1)).view(new_shape)
    noise = torch.randn_like(inputs, device=device) * sigma
    noisy_inputs = inputs + noise

    outputs = model(noisy_inputs)
    outputs = outputs.reshape((input_size, gauss_num, num_classes))

    # Classification loss
    outputs_softmax = F.softmax(outputs, dim=2).mean(1)
    outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
    classification_loss = F.nll_loss(
        outputs_logsoftmax, targets, reduction='sum')
    cl_total += classification_loss.item()

    if lbd == 0:
      robustness_loss = classification_loss * 0
    else:
      # Robustness loss
      beta_outputs = outputs * beta  # only apply beta to the robustness loss
      beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
      top2 = torch.topk(beta_outputs_softmax, 2)
      top2_score = top2[0]
      top2_idx = top2[1]
      indices_correct = (top2_idx[:, 0] == targets)  # G_theta

      out0, out1 = top2_score[indices_correct,
                              0], top2_score[indices_correct, 1]
      robustness_loss = m.icdf(out1) - m.icdf(out0)
      indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
          robustness_loss) & (torch.abs(robustness_loss) <= gamma)  # hinge
      out0, out1 = out0[indices], out1[indices]
      robustness_loss = m.icdf(out1) - m.icdf(out0) + gamma
      robustness_loss = robustness_loss.sum() * sigma / 2
      rl_total += robustness_loss.item()

    # Final objective function
    loss = classification_loss + lbd * robustness_loss
    loss /= input_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  cl_total /= input_total
  rl_total /= input_total
  print('Classification Loss: {}  Robustness Loss: {}'.format(cl_total, rl_total))

  return cl_total, rl_total