import logging

import torch
from torch import nn

class SmoothLabelCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1, log_prefix='', ignore_index=None):
        super().__init__()
        self.eps = eps
        self.log_soft = nn.LogSoftmax(dim=1)
        #self.kl = nn.KLDivLoss(reduction='batchmean')
        self.kl = nn.KLDivLoss(reduction='none')

        # for verbose printing only
        #self.register_buffer('iter', torch.tensor(0))
        self.iter = 0
        self.max_loss = 0
        self.min_loss = 0
        self.log_prefix = log_prefix
        self.ignore_index = ignore_index

    def forward(self, feature, target):
        # if it is fp16, convert it to fp32 explicitly as some trainer will not
        # do automatically
        feature = feature.float()
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target[valid_mask]
            feature = feature[valid_mask]
        assert target.numel() > 0
        debug_print = (self.iter % 100) == 0
        self.iter += 1
        eps = self.eps
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(feature)
        if debug_print:
            with torch.no_grad():
                prob = torch.nn.functional.softmax(feature.detach(), dim=1)
                num = feature.size(0)
                avg_prob = prob[torch.arange(num), target].mean()
                logging.info('{}: iter={}, avg pos = {}, max loss = {}, min loss = {}'.format(
                    self.log_prefix,
                    self.iter,
                    avg_prob,
                    self.max_loss,
                    self.min_loss,
                ))
                self.max_loss = 0
                self.min_loss = 10000000
        loss = self.kl(log_prb, one_hot)
        with torch.no_grad():
            if len(loss) > 0:
                self.max_loss = max(self.max_loss, loss.max().cpu())
                self.min_loss = min(self.min_loss, loss.min().cpu())
        return loss.sum(dim=1).mean()
