from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class WTTMLoss(nn.Module):
    def __init__(self):
        super(WTTMLoss, self).__init__()

    def forward(self, y_s, y_t, gamma, beta):
        p_s = F.log_softmax(y_s, dim=1)
        p_t = torch.pow(torch.softmax(y_t, dim=1), gamma)
        norm = torch.sum(p_t, dim=1)
        p_t = p_t / norm.unsqueeze(1)
        KL = torch.sum(F.kl_div(p_s, p_t, reduction='none'), dim=1)
        loss = torch.mean(norm*KL)

        return beta * loss
