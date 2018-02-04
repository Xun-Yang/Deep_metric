from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.autograd import Variable


def orth_reg(net, loss, cof=1e-3):
    orth_loss = 0
    for m in net.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            dimension = w.size()[0]
            diff = torch.matmul(w, w.t()) - Variable(torch.eye(dimension),  requires_grad=False).cuda()
            _loss = (1.0/w.size()[0])*torch.pow(torch.norm(diff), 2)
            orth_loss += cof*_loss
            loss = loss + orth_loss
    return loss
