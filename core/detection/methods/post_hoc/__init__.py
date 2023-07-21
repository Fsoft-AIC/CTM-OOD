from .msp_energy import EnergyBasedDetector, MSPDetector, MLSDetector
from .odin import ODINDetector
from .mahalanobis import MahalanobisDetector
# from .knn import KNNPostprocessor
from .ctm import CTMDetector

Name2Class = {
    'msp': MSPDetector,
    'mls': MLSDetector,
    'energy': EnergyBasedDetector,
    'odin': ODINDetector,
    'mahalanobis': MahalanobisDetector,
    # 'knn': KNNPostprocessor,
    'ctm': CTMDetector,
}

def get_detector_from_name(name): 
    return Name2Class[name]
