import math
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
from loguru import logger
from scipy import stats
from scipy.ndimage import convolve

from avalon.common.errors import SwitchError
from avalon.datagen.godot_base_types import Vector2
from avalon.datagen.world_creation.geometry import Axis
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import euclidean_distance
from avalon.datagen.world_creation.indoor.constants import TILE_SIZE
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.constants import CornerType
from avalon.datagen.world_creation.indoor.constants import WallType
from avalon.datagen.world_creation.indoor.wall_footprint import WallFootprint
from avalon.datagen.world_creation.types import BuildingBoolNP
from avalon.datagen.world_creation.types import BuildingFloatNP
from avalon.datagen.world_creation.types import BuildingIntNP
from avalon.datagen.world_creation.types import Point3DNP

Corner = Tuple[Tuple[int, int], Optional[CornerType], bool]


def find_corners(tiles: BuildingIntNP, solid_tile_value: int = 1) -> List[Corner]:
    """
    Iterates through tiles to find all interior/exterior corners of a wall in order.
    Rough algorithm:
    1. Iterate through tiles east-ward until it finds a wall (denoted by `solid_tile_value`)
    2. Trace along the wall starting from the first point, attempting to make a turn
    3. If a turn is possible, a corner is found and direction changes; if not, keep going the same direction
    4. Repeat steps 2-3 until we reach the initial corner again - then all corners have been found.
    """
    length, width = tiles.shape
    current_direction = Azimuth.EAST
    z, x = 0, 0
    done = False
    corners: List[Corner] = []
    while not done:
        if not corners:
            if tiles[z, x] == solid_tile_value:
                corners.append(((z, x), None, True))
            else:
                if x < width - 1:
                    x += 1
                    continue
                elif z < length - 1:
                    x = 0
                    z += 1
                    continue
                else:
                    break  # no corners

        # Always try turning left first
        directions = [Azimuth.NORTH, Azimuth.EAST, Azimuth.SOUTH, Azimuth.WEST] * 2
        current_idx = directions.index(current_direction)
        sorted_dirs = [directions[current_idx - 1], *directions[current_idx : current_idx + 3]]
        for i, new_direction in enumerate(sorted_dirs):
            new_z, new_x = z, x
            if new_direction == Azimuth.NORTH:
                new_z -= 1
            elif new_direction == Azimuth.EAST:
                new_x += 1
            elif new_direction == Azimuth.SOUTH:
                new_z += 1
            elif new_direction == Azimuth.WEST:
                new_x -= 1
            else:
                raise SwitchError(new_direction)

            if new_z < 0 or new_x < 0:
                continue  # negative = out of bounds

            try:
                tile = tiles[new_z, new_x]
            except IndexError:
                continue  # out of bounds

            if tile != solid_tile_value:
                continue  # outside footprint

            last_coordinates = (z, x)
            z, x = new_z, new_x
            if new_direction == current_direction:
                break  # keep going along the border, no corner yet

            is_outside_corner = i != 0
            type_mapping = {
                (Azimuth.NORTH, Azimuth.EAST): (CornerType.NW,),
                (Azimuth.NORTH, Azimuth.WEST): (CornerType.NE,),
                (Azimuth.NORTH, Azimuth.SOUTH): (CornerType.NW, CornerType.NE),
                (Azimuth.EAST, Azimuth.NORTH): (CornerType.SE,),
                (Azimuth.EAST, Azimuth.SOUTH): (CornerType.NE,),
                (Azimuth.EAST, Azimuth.WEST): (CornerType.NE, CornerType.SE),
                (Azimuth.SOUTH, Azimuth.EAST): (CornerType.SW,),
                (Azimuth.SOUTH, Azimuth.WEST): (CornerType.SE,),
                (Azimuth.SOUTH, Azimuth.NORTH): (CornerType.SE, CornerType.SW),
                (Azimuth.WEST, Azimuth.SOUTH): (CornerType.NW,),
                (Azimuth.WEST, Azimuth.NORTH): (CornerType.SW,),
                (Azimuth.WEST, Azimuth.EAST): (CornerType.SW, CornerType.NW),
            }
            new_corners: List[Corner] = []
            for corner_type in type_mapping[(current_direction, new_direction)]:
                new_corners.append((last_coordinates, corner_type, is_outside_corner))
            if last_coordinates == corners[0][0]:
                corners.pop(0)
                corners = new_corners + corners
                done = True
            else:
                corners.extend(new_corners)
                current_direction = new_direction
            break
    return corners


def find_exterior_wall_footprints(
    tiles: BuildingIntNP, no_build_mask: Optional[BuildingBoolNP] = None
) -> List[WallFootprint]:
    walls = []
    for wall_type in WallType:
        wall_kernel = wall_type.convolution_kernel
        out_of_bounds = tiles == False
        wall_or_corner_mask = convolve(tiles.astype(np.int8), wall_kernel, mode="constant") == 0
        wall_mask = ~out_of_bounds & wall_or_corner_mask
        for corner_type in wall_type.corner_types:
            corner_kernel = corner_type.convolution_kernel
            corner_mask = convolve(tiles.astype(np.int8), corner_kernel, mode="constant") == 0
            wall_mask &= ~corner_mask

        # Hallways and story links (ramps) intersecting with exterior walls are entrances/exits; if mask is passed,
        # we exclude them; e.g. to avoid building windows there.
        if no_build_mask is not None:
            wall_mask &= ~no_build_mask

        if wall_type.is_vertical:
            wall_mask = wall_mask.T
        wall_tiles = list(zip(*np.where(wall_mask)))

        wall = None
        for current_tile, next_tile in zip(wall_tiles[:-1], wall_tiles[1:]):
            if not wall:
                z, x = current_tile
                wall = [(z, x) if wall_type.is_vertical else (x, z)]

            end_tile = None
            if current_tile[1] + 1 != next_tile[1]:
                end_tile = current_tile
            elif wall is not None and next_tile == wall_tiles[-1]:
                end_tile = next_tile

            if end_tile:
                z, x = end_tile
                z += 1
                x += 1
                wall.append((z, x) if wall_type.is_vertical else (x, z))
                walls.append(WallFootprint(wall[0], wall[1], wall_type, wall_type.is_vertical))
                wall = None
    return walls


def visualize_tiles(tiles: BuildingIntNP) -> None:
    char_by_int = {i + 1: chr(ord("A") + i) for i in range(0, 52)}
    char_by_int[0] = "."
    for line in tiles:
        chars = [char_by_int[-x if x < 0 else x] for x in line]
        logger.debug("".join(chars))


def draw_line_in_grid(
    points: Tuple[BuildingTile, ...],
    grid: Union[BuildingIntNP, BuildingFloatNP],
    line_tile_value: Union[int, float],
    drawable_grid_value: Optional[Union[int, float]] = 0,
    include_ends: bool = True,
) -> None:
    for i, (from_point, to_point) in enumerate(zip(points[:-1], points[1:])):
        is_first_connection = i == 0
        is_last_connection = i == len(points) - 2
        is_connection_vertical = from_point.x == to_point.x
        if is_connection_vertical:
            axis = Axis.Z.value
        else:
            axis = Axis.X.value

        # We call max(..., 0) here for negative coordinates (exterior points) to work correctly
        start_coord = max(min(getattr(from_point, axis), getattr(to_point, axis)), 0)
        end_coord = max(getattr(from_point, axis), getattr(to_point, axis)) + 1
        if not include_ends:
            if is_first_connection and start_coord != 0:
                start_coord += 1
            if is_last_connection:
                end_coord -= 1

        if is_connection_vertical:
            local_grid = grid[start_coord:end_coord, from_point.x]
        else:
            local_grid = grid[from_point.z, start_coord:end_coord]

        if drawable_grid_value is not None:
            local_grid[local_grid == drawable_grid_value] = line_tile_value
            if include_ends:
                # todo: this is sorta leaky
                local_grid[0] = line_tile_value
                local_grid[-1] = line_tile_value
        else:
            local_grid[True] = line_tile_value


def tile_centroid(tile_position: Tuple[int, int], tile_size: float = 1) -> Vector2:
    half_size = tile_size / 2
    return Vector2(tile_position[0] + half_size, tile_position[1] + half_size)


def decide_tiles_by_distance(
    free_tiles: List[Tuple[int, int]],
    target_tile: Tuple[int, int],
    difficulty: float,
    rand: np.random.Generator,
    tile_count: int = 1,
) -> List[Tuple[int, int]]:
    target_tile_position = BuildingTile(*target_tile)
    tile_distances = [
        euclidean_distance(BuildingTile(*tile_position), target_tile_position) for tile_position in free_tiles
    ]
    desired_distance = difficulty * max(tile_distances)
    std = np.std(tile_distances)
    std = std if std != 0 else 1  # std=0 yields invalid distribution
    distance_distribution = stats.norm(desired_distance, math.sqrt(std))
    weights = np.array([distance_distribution.pdf(distance) for distance in tile_distances])
    weights /= weights.sum()
    return [tuple(tile) for tile in rand.choice(free_tiles, tile_count, p=weights)]  # type: ignore


def get_neighbor_tiles(
    tiles: np.ndarray, point: Tuple[int, int], only_if_equals: Optional[Any] = None
) -> Tuple[Tuple[int, int], ...]:
    z, x = point
    length, width = tiles.shape
    neighbor_tiles = []
    if z + 1 < length:
        neighbor_tiles.append((z + 1, x))
    if x + 1 < width:
        neighbor_tiles.append((z, x + 1))
    if z - 1 >= 0:
        neighbor_tiles.append((z - 1, x))
    if x - 1 >= 0:
        neighbor_tiles.append((z, x - 1))
    return tuple(
        [tile for tile in neighbor_tiles if only_if_equals is None or tiles[tile[0], tile[1]] == only_if_equals]
    )


def is_point_in_tile(point: Point3DNP, tile: BuildingTile) -> bool:
    x, _y, z = point
    return cast(bool, tile.x < x < tile.x + TILE_SIZE and tile.z < z < tile.z + TILE_SIZE)
