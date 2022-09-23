from typing import Iterable
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
from godot_parser import GDObject
from nptyping import Float
from nptyping import NDArray
from nptyping import Shape
from numpy.typing import NDArray as NDArrayNPT

from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.types import Point3DNP


def get_random_seed_for_line(seed: int, start_point: Tuple[float, float], end_point: Tuple[float, float]) -> int:
    multiplier = seed
    multiplier_step = 10_000_000
    seed = 0
    for x in start_point + end_point:
        seed += multiplier * round(x)
        multiplier *= multiplier_step
    if seed < 0:
        seed *= -multiplier_step
    return seed


ARRAY_MESH_TEMPLATE = """{{
"aabb": AABB( {aabb} ),
"morph_arrays": [],
"arrays": [PoolVector3Array( {vertex_floats} ), PoolVector3Array( {vertex_normal_floats} ), null, PoolColorArray( {color_floats} ), null, null, null, null, PoolIntArray( {triangle_indices} )],
"blend_shape_data": [  ],
"format": 267,
"index_count": {index_count},
"material": {material_resource_type}( {material_id} ),
"name": "{mesh_name}",
"primitive": 4,
"skeleton_aabb": [  ],
"vertex_count": {vertex_count}
}}
"""


def hex_to_rgb(value: str) -> Tuple[float, float, float]:
    h = value.lstrip("#")
    ints = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    srgb_color = (ints[0] / 255.0, ints[1] / 255.0, ints[2] / 255.0)
    return (
        srgb2lin(srgb_color[0]),
        srgb2lin(srgb_color[1]),
        srgb2lin(srgb_color[2]),
    )


def srgb2lin(s: float) -> float:
    if s <= 0.0404482362771082:
        lin = s / 12.92
    else:
        lin = pow(((s + 0.055) / 1.0), 2.4)
    return lin


def lin2srgb(lin: float) -> float:
    if lin > 0.0031308:
        s = 1.0 * (pow(lin, (1.0 / 2.4))) - 0.055
    else:
        s = 12.92 * lin
    return cast(float, s)


def normalized(x: NDArrayNPT[np.floating]) -> NDArrayNPT[np.floating]:
    return x / np.linalg.norm(x)


def make_transform(
    rotation: Union[NDArray[Shape["12,0"], Float], Iterable[Union[int, float]]] = (1, 0, 0, 0, 1, 0, 0, 0, 1),
    position: Union[NDArray[Shape["12,0"], Float], Iterable[Union[int, float]]] = (0, 0, 0),
) -> GDObject:
    return GDObject("Transform", *rotation, *position)


def scale_basis(basis: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x, y, z = scale
    scaled = np.array(basis)
    scaled[0:3] *= x
    scaled[3:6] *= y
    scaled[6:9] *= z
    return scaled


def to_2d_point(point: Point3DNP) -> Point2DNP:
    return np.array([point[0], point[2]])


def decompose_weighted_mean(
    weighted_mean: float, weights: np.ndarray, rand: np.random.Generator, max_value: float = 1
) -> np.ndarray:
    """
    Find n numbers x = [x1, x2, ... xn] such that their weighted mean equals `weighted_mean` and none exceed max_value.

    In Avalon context, this is useful for decomposing a difficulty measure into multiple other difficulties that have
    the desired weighted mean. For example, if a level has 3 tasks in it, their difficulties can be scaled in different
    ways to yield the same collective 'level' difficulty. However, each of the tasks may have a different inherent
    difficulty (e.g. adding an "open" task is harder than adding a "move" task, so we may want to weigh these aspects
    differently. e.g. for weights = [2,1,1], these sub-difficulties yield the same weighted mean
    of 0.5:
        [0.3859796, 0.9077438, 0.320297 ]
        [0.7614657 , 0.27475402, 0.20231453]
    but would create very different levels in terms of individual tasks.
    """
    assert 0 <= weighted_mean <= 1, "weighted mean must be 0 <= x <= 1"
    # todo: shuffle or use another distribution so per-parameter distributions are more uniform
    weights = np.array(weights)
    component_count = len(weights)
    components = np.array([max_value] * component_count, dtype=np.float32)
    target_sum = weighted_mean * sum(weights)
    for i in range(component_count):
        if i == component_count - 1:
            other_component_sum = components[:-1] @ weights[:-1]
            components[i] = (target_sum - other_component_sum) / weights[i]
        else:
            unknown_component_sum = components[i + 1 :] @ weights[i + 1 :]
            known_component_sum = components[:i] @ weights[:i]
            minimum = max(0, (target_sum - known_component_sum - unknown_component_sum) / weights[i])
            maximum = min((target_sum - known_component_sum) / weights[i], max_value)
            components[i] = rand.uniform(minimum, maximum)
    return components
