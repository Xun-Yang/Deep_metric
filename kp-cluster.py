from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from utils import BatchGenerator

import numpy as np


def pair_euclidean_dist(inputs_x, inputs_y):
    n = inputs_x.size(0)
    m = inputs_y.size(0)
    xx = np.pow(inputs_x, 2).sum(dim=1, keepdim=True).expand(n, m)
    yy = np.pow(inputs_y, 2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy
    dist.addmm_(1, -2, inputs_x, inputs_y.t())
    # dist = dist.clamp(min=1e-12).sqrt()
    return dist

features = np.load('feat.npy')
labels = np.load('labels.npy')

# features = np.load('0_feat.npy')
# labels = np.load('0_label.npy')

num_instances = 8
batch_size = 128

idx = []

Batch = BatchGenerator(labels, num_instances=num_instances, batch_size=batch_size)


for i in range(2):
    batch = Batch.batch()
    X_ = features[batch, :]
    y = labels[batch]
    y = [[k] for k in y]
    enc = OneHotEncoder()
    enc.fit(y)

    one_hot_y = enc.transform(y).toarray()
    X = np.concatenate([X_, 0.1*one_hot_y], 1)

    kmeans = KMeans(n_clusters=36, random_state=0).fit(X)
    pred_cluster = kmeans.labels_
    result = dict()
    for i in range(len(y)):
        k = str(y[i])+' ' + str(pred_cluster[i])
        if k in result:
            result[k].append(i)
        else:
            result[k] = [i]
    split_ = result.values()

centroid_labels = []
centroids = []
















