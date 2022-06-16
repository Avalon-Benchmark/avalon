from typing import Any
from typing import List
from typing import Tuple
from typing import Union

# TODO: think a little more critically here...
import numpy as np
import scipy.ndimage
from IPython.core.display import HTML
from IPython.core.display import display
from attr import Attribute
from matplotlib import pyplot as plt
from trimesh import Trimesh

from datagen.world_creation.constants import AvalonTask

WORLD_RAISE_AMOUNT = 10_000


class ImpossibleWorldError(Exception):
    pass


class WorldTooSmall(ImpossibleWorldError):
    def __init__(self, task: AvalonTask, min_dist: float, available_dist: float, *args: object) -> None:
        super().__init__(f"Small {task} world: needed {min_dist} but only have {available_dist}", *args)


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


IS_DEBUG_VIS = False
IS_RAYCAST_DEBUG = False


# TERRAIN_WIDTH = 30.0
# TERRAIN_DEPTH = 30.0


def plot_points(points, index1, index2):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    plt.plot(points[:, index1], points[:, index2], "o")
    # plt.xlim((0.0, TERRAIN_WIDTH))
    # plt.ylim((0.0, TERRAIN_DEPTH))
    plt.show()


def plot_triangulation(triangulation):
    plt.triplot(triangulation.points[:, 0], triangulation.points[:, 1], triangulation.simplices)
    plt.plot(triangulation.points[:, 0], triangulation.points[:, 1], "o")
    plt.show()


def plot_terrain(data, *args, **kwargs):
    data = data.copy()
    island = data > WORLD_RAISE_AMOUNT - 1_000
    min_island_height = data[island].min()
    data[island] = data[island] - min_island_height
    data[data < 0.0] = data.max()
    plot_value_grid(data, *args, **kwargs)


def plot_value_grid(data, title="", markers=tuple()):
    data = np.array(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    if title:
        ax.set_title(title)
    y = list(range(0, data.shape[0]))
    x = list(range(0, data.shape[1]))
    ax.pcolormesh(x, y, data, shading="nearest", vmin=data.min(initial=0.0), vmax=data.max(initial=1.0))

    if markers:
        x = [x[1] for x in markers]
        y = [x[0] for x in markers]
        first_line = plt.plot(x[:1], y[:1], "o", markerfacecolor="orange", markersize=12)[0]
        line = plt.plot(x[1:], y[1:], "o", markerfacecolor="red", markersize=12)[0]
        line.set_clip_on(False)

    plt.show()
    fig.clf()


def plot_value_grid_multi_marker(data, title: str, marker_lists):
    data = np.array(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    if title:
        ax.set_title(title)
    y = list(range(0, data.shape[0]))
    x = list(range(0, data.shape[1]))
    ax.pcolormesh(x, y, data, shading="nearest", vmin=data.min(initial=0.0), vmax=data.max(initial=1.0))

    for markers, color in marker_lists:
        x = [x[1] for x in markers]
        y = [x[0] for x in markers]
        line = plt.plot(x, y, "o", markerfacecolor=color, markersize=12)[0]
        line.set_clip_on(False)

    plt.show()
    fig.clf()


def print_biome_legend(config):
    display(HTML(f"<h1>Biomes:</h1>"))

    for biome in config.biomes:
        display(HTML(f'<h2 style="background: {biome.color}">{biome.name}</h2>'))


def at_least_n_items(n):
    def check(instance: Any, attribute: Attribute, value: Union[List, Tuple]):
        assert len(value) >= n, f"Must pass at least {n} {attribute.name} to {instance.__class__.__name__}"

    return check


def decompose_weighted_mean(
    weighted_mean: float, weights: np.array, max_value: float = 1, rand: np.random.Generator = np.random.default_rng()
):
    """
    find n numbers x = [x1, x2, ... xn] such that their weighted mean equals `weighted_mean` and none exceed max_value
    """
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


def inset_borders(array):
    # requires "solids" to be 1 and "voids" to be 0
    inset_array = array.copy()
    convolution = scipy.ndimage.convolve(array.astype(np.int), np.ones((3, 3)), mode="constant")
    inset_array[convolution != 9] = 0
    return inset_array


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


def unnormalize(mesh: Trimesh):
    """Re-create mesh such that none of its faces share vertices"""
    points_per_face = 3
    coords_per_point = 3
    face_count = len(mesh.faces)
    new_vertices = np.empty((face_count * points_per_face, coords_per_point), dtype=np.float32)
    for i, face in enumerate(mesh.faces):
        offset = i * points_per_face
        new_vertices[offset : offset + points_per_face, :] = mesh.vertices[face]
    new_faces = np.array(range(len(new_vertices))).reshape(-1, coords_per_point)
    return Trimesh(vertices=new_vertices, faces=new_faces, process=False)
