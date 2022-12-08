from typing import Any
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from nptyping import NDArray
from nptyping import Shape
from scipy.spatial import Delaunay

from avalon.datagen.world_creation.constants import WORLD_RAISE_AMOUNT

# TODO: remove this
IS_DEBUG_VIS = False


def plot_points(points: Union[np.ndarray, List[np.ndarray]], index1: int, index2: int) -> None:
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    plt.plot(points[:, index1], points[:, index2], "o")
    # plt.xlim((0.0, TERRAIN_WIDTH))
    # plt.ylim((0.0, TERRAIN_DEPTH))
    plt.show()


def plot_triangulation(triangulation: Delaunay) -> None:
    plt.triplot(triangulation.points[:, 0], triangulation.points[:, 1], triangulation.simplices)
    plt.plot(triangulation.points[:, 0], triangulation.points[:, 1], "o")
    plt.show()


GridPlotDataType = NDArray[Shape["MapY, MapX"], Any]


def plot_terrain(
    data: GridPlotDataType, title: str = "", markers: Iterable[Union[np.ndarray, Tuple[float, float]]] = tuple()
) -> None:
    data = data.copy()
    island = data > WORLD_RAISE_AMOUNT - 1_000
    min_island_height = data[island].min()
    data[island] = data[island] - min_island_height
    data[data < 0.0] = data.max()
    plot_value_grid(data, title=title, markers=markers)


def plot_value_grid(
    data: GridPlotDataType, title: str = "", markers: Iterable[Union[np.ndarray, Tuple[float, float]]] = tuple()
) -> None:
    data = np.array(data)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    if title:
        ax.set_title(title)
    y = list(range(0, data.shape[0]))
    x = list(range(0, data.shape[1]))
    ax.pcolormesh(x, y, data, shading="nearest", vmin=data.min(initial=0.0), vmax=data.max(initial=1.0))

    if markers:
        marker_x = [x[1] for x in markers]
        marker_y = [x[0] for x in markers]
        first_line = plt.plot(marker_x[:1], marker_y[:1], "o", markerfacecolor="orange", markersize=12)[0]
        line = plt.plot(marker_x[1:], marker_y[1:], "o", markerfacecolor="red", markersize=12)[0]
        line.set_clip_on(False)

    plt.show()
    fig.clf()


def plot_value_grid_multi_marker(
    data: GridPlotDataType,
    title: str,
    marker_lists: Sequence[Tuple[Iterable[Union[np.ndarray, Tuple[int, int]]], str]],
) -> None:
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
