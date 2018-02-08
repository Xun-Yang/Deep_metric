from __future__ import print_function, absolute_import

from .SoftmaxNeigLoss import SoftmaxNeigLoss
from .KNNSoftmax import KNNSoftmax

__factory = {
    'softneig': SoftmaxNeigLoss,
    'knnsoftmax': KNNSoftmax,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)
