# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import DataSet
import numpy as np

from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture

def Euclidean_(X, Y):
    XX = np.sum(np.power(X, 2), 1)
    YY = np.sum(np.power(Y, 2), 1)
    X_ = np.repeat(np.expand_dims(XX, axis=1), len(YY), axis=1)
    Y_ = np.repeat(np.expand_dims(YY, axis=1), len(XX), axis=1).transpose()
    dist = X_ + Y_ - 2 * np.matmul(X, Y.transpose())
    return dist


def display(X):
    for l in X:
        # for i in range(len(l)):
        temp = ['%.2f' % k for k in l]
        print(' '.join(temp))
        # print('\n')


def normalize(X):
    norm_inverse = np.diag(1/np.sqrt(np.sum(np.power(X, 2), 1)))
    X_norm = np.matmul(norm_inverse, X)
    return X_norm

torch.cuda.set_device(7)
cudnn.benchmark = True
r = '/opt/intern/users/xunwang/checkpoints/nca/cub/48/model.pkl'
data = 'cub'
dim = 48
n_clusters = 3

# model = inception_v3(dropout=0.5)
model = torch.load(r)
model = model.cuda()

data = DataSet.create(data, train=True)
data_loader = torch.utils.data.DataLoader(
    data.train, batch_size=8, shuffle=False, drop_last=False)

features, labels = extract_features(model, data_loader, print_freq=32, metric=None)
features = [feature.resize_(1, dim) for feature in features]
features = torch.cat(features)
features = features.numpy()
labels = np.array(labels)

# clustering test
# weight = n_clusters * [0.2]
centers = []
center_labels = []
for label in set(labels):
    X = features[labels == label]
    kmeans = KMeans(n_clusters=n_clusters, random_state=None).fit(X)
    center_ = kmeans.cluster_centers_
    centers.extend(center_)
    center_labels.extend(n_clusters*[label])
centers = np.conjugate(centers)




















