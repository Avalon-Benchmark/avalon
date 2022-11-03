from typing import Tuple

import attr
import numpy as np

from avalon.datagen.world_creation.types import MapFloatNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HarmonicsConfig:
    harmonics: Tuple[int, ...]
    weights: Tuple[float, ...]
    # just used for debugging
    is_deterministic: bool = False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class EdgeConfig:
    # edges are scaled based on their radius so that the size of the noise is constant
    # this factor changes that automatic scaling
    # a higher number will effectively scale up a smaller circle by that much
    scale: float = 1.0
    # noise of 0.0 means "no noise, smooth line", noise of 1.0 means "very noisy"
    # going much beyond 1.0 is probably a bad idea
    noise: float = 0.5
    # 1.0 means a perfect circle, 0.0 means weirdly blobby
    circularity: float = 0.5

    def to_harmonics(self, radius: float) -> HarmonicsConfig:
        noise = self.noise**0.5
        base_radius = 20.6
        base_harmonic_count = 20 * noise
        scale_factor = (radius / base_radius) * (1.0 / self.scale)
        noise_harmonics = list(range(1, round(base_harmonic_count * scale_factor**0.5)))
        scaled_harmonics = tuple([round(2 * scale_factor) + i for i in noise_harmonics])
        noise_weights = tuple([noise * 0.75 / x for x in scaled_harmonics])

        shape_factor = (1.0 - self.circularity) * 1.75
        shape_weights = tuple(x * shape_factor for x in [0.1, 0.3, 0.2, 0.1, 0.05])
        shape_harmonics = (1, 2, 3, 4, 5)

        return HarmonicsConfig(
            harmonics=scaled_harmonics + shape_harmonics,
            weights=noise_weights + shape_weights,
        )


def create_harmonics(
    rand: np.random.Generator, theta: MapFloatNP, config: HarmonicsConfig, is_normalized: bool
) -> MapFloatNP:
    variation = np.zeros_like(theta)
    for harmonic, weight in zip(config.harmonics, config.weights):
        # noise = rand.normal()
        noise: float = rand.uniform(-1, 1)
        if config.is_deterministic:
            noise = 1.0
        variation += np.sin(harmonic * theta) * ((noise) * weight)
        # noise = rand.normal()
        noise = rand.uniform(-1, 1)
        if config.is_deterministic:
            noise = 1.0
        variation += np.cos(harmonic * theta) * ((noise) * weight)
    if is_normalized:
        variation_min = np.min(variation)
        variation = (variation - variation_min) / (np.max(variation) - variation_min)
    return variation
