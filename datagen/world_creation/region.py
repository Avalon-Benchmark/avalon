import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import attr
import numpy as np
from numpy.random import Generator

import openturns as ot
from contrib.serialization import Serializable
from datagen.godot_base_types import NewRange
from datagen.world_creation.utils import get_random_seed_for_line


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloatRange(NewRange):
    min_ge: float
    max_lt: float

    def overlap(self, other: "FloatRange") -> Optional["FloatRange"]:
        min_ge = max(self.min_ge, other.min_ge)
        max_lt = min(self.max_lt, other.max_lt)
        if min_ge >= max_lt:
            return None
        return FloatRange(min_ge, max_lt)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Region(Serializable):
    x: FloatRange
    z: FloatRange

    def contains_point_2d(self, point: Union[np.ndarray, Tuple[float, float]], epsilon: float = 0.01) -> bool:
        if point[0] >= self.x.min_ge - epsilon and point[0] < self.x.max_lt + epsilon:
            if point[1] >= self.z.min_ge - epsilon and point[1] < self.z.max_lt + epsilon:
                return True
        return False

    def contains_point_2d_exact(self, point: Union[np.ndarray, Tuple[float, float]]) -> bool:
        return (
            point[0] >= self.x.min_ge
            and point[0] < self.x.max_lt
            and point[1] >= self.z.min_ge
            and point[1] < self.z.max_lt
        )

    def overlaps_region(self, other: "Region") -> bool:
        for point in other.points:
            if self.contains_point_2d(point):
                return True
        if other.contains_point_2d((self.x.min_ge, self.z.min_ge)):
            return True
        return False

    # TODO: possibly better to think about these regions as having heights as well
    def contains_point_3d(self, point: Union[np.ndarray, Tuple[float, float, float]], epsilon: float = 0.01) -> bool:
        if point[0] > self.x.min_ge - epsilon and point[0] < self.x.max_lt + epsilon:
            if point[2] > self.z.min_ge - epsilon and point[2] < self.z.max_lt + epsilon:
                return True
        return False

    def contains_region(self, other: "Region", epsilon: float = 0.01) -> bool:
        for point in other.points:
            if not self.contains_point_2d(point, epsilon=epsilon):
                return False
        return True

    @property
    def points(self) -> List[Tuple[float, float]]:
        return [
            (float(self.x.min_ge), float(self.z.min_ge)),
            (float(self.x.max_lt), float(self.z.min_ge)),
            (float(self.x.max_lt), float(self.z.max_lt)),
            (float(self.x.min_ge), float(self.z.max_lt)),
        ]

    def get_linspace(self, cells_per_meter: float) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.x.min_ge, self.x.max_lt, round(self.x.size * cells_per_meter) + 1)
        y = np.linspace(self.z.min_ge, self.z.max_lt, round(self.z.size * cells_per_meter) + 1)
        return x, y

    def get_random_generator(self, seed: int) -> Generator:
        seed_for_region = get_random_seed_for_line(
            seed, (self.x.min_ge, self.z.min_ge), (self.x.max_lt, self.z.max_lt)
        )
        return np.random.default_rng(seed_for_region)

    def epsilon_expand(self, x_min, x_max, z_min, z_max, epsilon=0.01):
        x_min_epsilon = epsilon if x_min else 0
        x_max_epsilon = epsilon if x_max else 0
        z_min_epsilon = epsilon if z_min else 0
        z_max_epsilon = epsilon if z_max else 0
        return Region(
            FloatRange(self.x.min_ge - x_min_epsilon, self.x.max_lt + x_max_epsilon),
            FloatRange(self.z.min_ge - z_min_epsilon, self.z.max_lt + z_max_epsilon),
        )

    def get_randomish_points(self, count: int) -> np.ndarray:
        points = create_low_discrepancy_random_points(count)
        # shift to 0 - 1 range
        points *= 0.5
        points += 0.5
        # scale to region size
        sizes = np.array([self.x.size, self.z.size])
        points *= sizes
        # add the min
        min_point = np.array([self.x.min_ge, self.z.min_ge])
        points += min_point
        return points


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class EdgedRegion(Region):
    edge_vertices: np.ndarray


def create_low_discrepancy_random_points(min_point_count: int) -> np.ndarray:
    dim = 2
    distribution = ot.ComposedDistribution([ot.Uniform()] * dim)
    sequence = ot.SobolSequence(dim)
    samplesize = 2 ** (int(math.log2(min_point_count)) + 1)  # Sobol' sequences are in base 2
    experiment = ot.LowDiscrepancyExperiment(sequence, distribution, samplesize, False)
    sample = experiment.generate()
    return np.array(sample)
