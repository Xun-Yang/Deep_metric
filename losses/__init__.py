from .triplet import TripletLoss
from .CenterTriplet import CenterTripletLoss
from .GaussianMetric import GaussianMetricLoss
from .HistogramLoss import HistogramLoss
from .BatchAll import BatchAllLoss
from .NeighbourLoss import NeighbourLoss
from .DistanceMatchLoss import DistanceMatchLoss
from .NeighbourHardLoss import NeighbourHardLoss
from .DistWeightLoss import DistWeightLoss
from .BinDevianceLoss import BinDevianceLoss
from .BinBranchLoss import BinBranchLoss
from .MarginDevianceLoss import MarginDevianceLoss
from .MarginPositiveLoss import MarginPositiveLoss
from .ContrastiveLoss import ContrastiveLoss
from .DistWeightContrastiveLoss import DistWeightContrastiveLoss
from .DistWeightDevianceLoss import DistWeightBinDevianceLoss
from .DistWeightDevBranchLoss import DistWeightDevBranchLoss

__factory = {
    'triplet': TripletLoss,
    'histogram': HistogramLoss,
    'gaussian': GaussianMetricLoss,
    'batchall': BatchAllLoss,
    'neighbour': NeighbourLoss,
    'distance_match': DistanceMatchLoss,
    'neighard': NeighbourHardLoss,
    'distweight': DistWeightLoss,
    'bin': BinDevianceLoss,
    'binbranch': BinBranchLoss,
    'margin': MarginDevianceLoss,
    'positive': MarginPositiveLoss,
    'con': ContrastiveLoss,
    'dwcon': DistWeightContrastiveLoss,
    'dwdev': DistWeightBinDevianceLoss,
    'dwdevbranch': DistWeightDevBranchLoss,
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
    return __factory[name]( *args, **kwargs)
