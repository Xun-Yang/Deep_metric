# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from evaluations import extract_features
import DataSet
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

torch.cuda.set_device(7)

cudnn.benchmark = True
r = '/opt/intern/users/xunwang/checkpoints/nca/cub/64/model.pkl'
data = 'cub'
dim = 64

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

for k in range(1):
    # clustering test
    n_clusters = 3
    weight = n_clusters * [0.2]
    X = features[labels == 11]
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical').fit(X)
    cluster_assignment = gmm.predict(X)
    print(gmm.covariances_)
    print(gmm.means_)
    # print(cluster_assignment)
    for i in range(n_clusters):
        print(np.sum(cluster_assignment == i))

















