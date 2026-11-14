from .rainfall import RainfallPlugin
from .elevation import ElevationPlugin
from .slope import SlopePlugin
from .twi import TWIPlugin
from .proximity import ProximityPlugin
from .land_use import LandUsePlugin
from .soil import SoilPlugin
from .water_exclusion import WaterExclusionPlugin
from .reporting import VectorExplainabilityPlugin, LandslideExplainabilityPlugin
from .curvature import CurvaturePlugin, SlopeAspectPlugin
try:
    from .hotspot import LandslideHotspotPlugin, extract_landslide_hotspots
except ImportError:
    pass
from .calculators import WeightedSumCalculator, BNPBCalculator
