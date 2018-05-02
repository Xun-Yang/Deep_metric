#!/usr/bin/env python

import numpy as np
from kmodes.kprototypes import KPrototypes

X = np.random.randint(1, 1000, (128, 32))

# # stocks with their market caps, sectors and countries
# syms = np.genfromtxt('stocks.csv', dtype=str, delimiter=',')[:, 0]
# print(syms)
# X = np.genfromtxt('stocks.csv', dtype=object, delimiter=',')[:, 1:]
# print(X)
# X[:, 0] = X[:, 0].astype(float)

kproto = KPrototypes(n_clusters=4, init='huang', gamma=0.1, verbose=2)
clusters = kproto.fit_predict(X, categorical=[1, 2])

# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

print(clusters)
# for s, c in zip(syms, clusters):
#     print("Symbol: {}, cluster:{}".format(s, c))