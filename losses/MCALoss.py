from __future__ import print_function, absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from utils import BatchGenerator, cluster_


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = torch.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = torch.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist


# def normalize(x):
#     norm = x.norm(dim=1, p=2, keepdim=True)
#     x = x.div(norm.expand_as(x))
#     return x


class MCALoss(nn.Module):
    def __init__(self, alpha=16, centers=None, center_labels=None):
        super(MCALoss, self).__init__()
        self.alpha = alpha
        self.centers = centers
        self.center_labels = center_labels

    def forward(self, inputs, targets):
        print('center is same or not \n?', self.centers[0][0])
        centers_dist = pair_euclidean_dist(inputs, (self.centers))
        loss = []
        dist_ap = []
        # dist_an = []
        num_match = 0
        for i, target in enumerate(targets):
            # for computation stability
            dist = centers_dist[i]
            pos_pair_mask = (self.center_labels == target)
            pos_pair = torch.masked_select(dist, pos_pair_mask)
            dist_ap.extend(pos_pair)

            base = (torch.max(dist) + torch.min(dist)).data[0]/2
            pos_exp = torch.sum(torch.exp(-self.alpha*(pos_pair - base)))
            a_exp = torch.sum(torch.exp(-self.alpha*(dist - base)))
            loss_ = - torch.log(pos_exp/a_exp)
            loss.append(loss_)
            if loss_.data[0] < 0.3:
                num_match += 1
        loss = torch.mean(torch.cat(loss))
        # print(dist_an, dist_ap)
        dist_an = torch.mean(centers_dist).data[0]
        dist_ap = torch.mean(torch.cat(dist_ap)).data[0]

        accuracy = float(num_match)/len(targets)
        return loss, accuracy, dist_ap, dist_an


def main():
    features = np.load('0_feat.npy')
    labels = np.load('0_label.npy')

    centers, center_labels = cluster_(features, labels, n_clusters=3)
    centers = Variable(torch.FloatTensor(centers).cuda(),  requires_grad=True)
    center_labels = Variable(torch.LongTensor(center_labels)).cuda()

    num_instances = 3
    batch_size = 120
    Batch = BatchGenerator(labels, num_instances=num_instances, batch_size=batch_size)
    batch = Batch.batch()

    inputs = Variable(torch.FloatTensor(features[batch, :])).cuda()
    targets = Variable(torch.LongTensor(labels[batch])).cuda()
    print(torch.mean(inputs))
    mca = MCALoss(alpha=16, centers=centers, center_labels=center_labels)
    for i in range(10):
        centers.grad.zero_()
        # loss, accuracy, dist_ap, dist_an =
            # MCALoss(alpha=16, centers=centers, center_labels=center_labels)(inputs, targets)
        loss, accuracy, dist_ap, dist_an = \
            mca(inputs, targets)
        print(loss.data[0])
        loss.backward()
        print(centers.grad.data)
        centers.data -= centers.grad.data
        # print(centers.grad)

if __name__ == '__main__':
    main()
    print('Congratulations to you!')

