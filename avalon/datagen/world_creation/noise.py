from typing import Tuple
from typing import cast

import numpy as np

from avalon.datagen.world_creation.types import MapFloatNP


# from: https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
def perlin(
    shape: Tuple[int, ...],
    scale: float,
    rand: np.random.Generator,
    is_normalized: bool = False,
    noise_min: float = 0.0,
) -> MapFloatNP:
    assert len(shape) == 2  # Ensures type-checking passes when passing `some_array.shape` directly
    cache_key = (scale,) + shape
    # create the grid
    y_lin = np.linspace(0, scale * shape[0], shape[0], endpoint=False)
    x_lin = np.linspace(0, scale * shape[1], shape[1], endpoint=False)
    x, y = np.meshgrid(x_lin, y_lin)

    # permutation table
    p = np.arange(max([round(scale * shape[0]), round(scale * shape[1])]) + 1, dtype=int)
    rand.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    result = lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

    if is_normalized:
        result -= np.min(result)
        result /= np.max(result)

        if noise_min > 0.0:
            result /= 1.0 - noise_min
            result += noise_min

    return cast(MapFloatNP, result)


def lerp(a, b, x):  # type: ignore
    "linear interpolation"
    return a + x * (b - a)


def fade(t):  # type: ignore
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def gradient(h, x, y):  # type: ignore
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y
