import torch
import torch.nn.functional as F


def consistency_loss(logits, lbd, loss='default', lbd2=0.5):
    """
    Consistency regularization for certified robustness.

    **IMPORTANT**: The loss design mainly evaluated in the paper is
    slightly different to the cross-entropy (KL + entropy), due to an
    incorrect implementation of the entropy term. We keep this form
    by default here to maintain the reproducibility (loss='default'),
    but also provide the corrected version (loss='xent') as an option
    like other variants, e.g., KL or MSE. All these forms (including
    'default') are valid losses to obtain consistency, and we have
    confirmed they all improve ACRs compared to the baselines.

    Parameters
    ----------
    logits : List[torch.Tensor]
        A list of logit batches of the same shape, where each
        is sampled from f(x + noise) with i.i.d. noises.
        len(logits) determines the number of noises, i.e., m > 1.
    lbd : float
        Hyperparameter that controls the strength of the regularization.
    loss : {'default', 'xent', 'kl_ent', 'kl', 'mse'} (optional)
        Which loss to minimize to achieve consistency.
        - 'default': The default implementation. All the reported values
            in the paper are reproducible with this option.
        - 'xent': The "correct" implementation of cross-entropy.
            Compared to 'default', one should use a lower lbd (e.g., lbd = 3)
            for better results. It usually achieves higher clean accuracy
            than 'default', and higher ACR particularly with SmoothAdv.
        - 'kl_ent': A hybrid form ``lbd * KL + lbd2 * entropy``.
            The entropy term has an extra hyperparameter lbd2.
            It is equivalent to 'xent' when lbd2 = lbd, but allows
            a larger lbd (e.g., lbd = 20) when lbd2 is smaller (e.g., lbd2 < 1).
            This roughly fills the gap between 'default' and 'xent'.
        - 'kl': The KL-divergence between each predictions and their average.
        - 'mse': The mean-squared error between the first two logits.
    lbd2 : float (optional)
        Only used when loss='kl_ent'.

    """

    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m

    loss_kl = [kl_div(logit, avg_softmax) for logit in logits]
    loss_kl = sum(loss_kl) / m

    if loss == 'default':
        loss_ent = __entropy(avg_softmax)
        consistency = lbd * (loss_kl + loss_ent)
    elif loss == 'xent':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * (loss_kl + loss_ent)
    elif loss == 'kl_ent':
        loss_ent = entropy(avg_softmax)
        consistency = lbd * loss_kl + lbd2 * loss_ent
    elif loss == 'kl':
        consistency = lbd * loss_kl
    elif loss == 'mse':
        sm1, sm2 = softmax[0], softmax[1]
        loss_mse = ((sm2 - sm1) ** 2).sum(1)
        consistency = lbd * loss_mse
    else:
        raise NotImplementedError()

    return consistency.mean()


def kl_div(input, targets):
    return F.kl_div(F.log_softmax(input, dim=1), targets, reduction='none').sum(1)


def entropy(input):
    logsoftmax = torch.log(input.clamp(min=1e-20))
    xent = (-input * logsoftmax).sum(1)
    return xent


def __entropy(input):
    targets_prob = F.softmax(input, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    return xent