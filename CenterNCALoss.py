from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


def euclidean_dist(inputs_):
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


class CenterNCALoss(nn.Module):
    def __init__(self, alpha=30):
        super(CenterNCALoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        n = inputs.size(0)
        num_dim = inputs.size(1)
        targets_ = list(set(targets.data))
        num_class = len(targets_)
        num_instance = n//num_class

        # targets_ = Variable(torch.LongTensor(targets_)).cuda()
        targets_ = Variable(torch.IntTensor(targets_))
        pos_mask = (targets.repeat(num_class, 1).t()).eq(targets_.repeat(n, 1))
        # _mask = Variable(torch.ByteTensor(num_class, n).fill_(1)).cuda() - mask_
        neg_mask = Variable(torch.ByteTensor(n, num_class).fill_(1)) - pos_mask

        centers = []
        inputs_list = []
        prec = 0

        temp = targets.data.numpy()
        for i, target in enumerate(targets_):
            idx = np.where(temp == target.data[0])
            input_ = inputs[idx[0], :]
            centers.append(torch.mean(input_, 0))
            inputs_list.append(input_)

        centers = [center.resize(1, num_dim) for center in centers]
        centers = torch.cat(centers, 0)

        # no gradient on centers
        centers = centers.detach()

        centers_dist = pair_euclidean_dist(inputs, centers)
        pos_dist = centers_dist[pos_mask]
        neg_dist = centers_dist[neg_mask].resize(n, num_class-1)

        # for computation stability
        base = torch.mean(centers_dist).data[0]
        loss = torch.mean(torch.exp(-self.alpha*(pos_dist - base)))

        dist_an = torch.mean(neg_dist).data[0]
        dist_ap = float(num_instance)/(num_instance-1) * torch.mean(pos_dist).data[0]

        return loss, prec, dist_ap, dist_an


def main():
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    x = Variable(torch.rand(data_size, input_dim), requires_grad=False)
    w = Variable(torch.rand(input_dim, output_dim), requires_grad=True)
    inputs = x.mm(w)
    y_ = 8*list(range(num_class))
    targets = Variable(torch.IntTensor(y_))

    print(CenterNCALoss()(inputs, targets))


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
