import math
from collections import defaultdict
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import attr
import godot_parser
import networkx as nx
import numpy as np
import scipy.ndimage
import seaborn as sns
from godot_parser import GDSubResourceSection
from godot_parser import Node as GDNode
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from networkx import Graph
from scipy.ndimage import convolve
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from trimesh import Trimesh
from trimesh import creation
from trimesh.visual import ColorVisuals

from common.errors import SwitchError
from contrib.serialization import Serializable
from datagen.godot_base_types import NewRange
from datagen.godot_base_types import Vector2
from datagen.godot_base_types import Vector3
from datagen.world_creation.geometry import Axis
from datagen.world_creation.geometry import Box
from datagen.world_creation.geometry import Geometry
from datagen.world_creation.geometry import Position
from datagen.world_creation.geometry import euclidean_distance
from datagen.world_creation.geometry import global_to_local_coords
from datagen.world_creation.geometry import local_to_global_coords
from datagen.world_creation.geometry import midpoint
from datagen.world_creation.godot_utils import make_transform
from datagen.world_creation.indoor.constants import CEILING_THICKNESS
from datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from datagen.world_creation.indoor.constants import SLOPE_THICKNESS
from datagen.world_creation.indoor.helpers import add_box_to_scene
from datagen.world_creation.indoor.helpers import draw_line_in_grid
from datagen.world_creation.indoor.helpers import rotate_position
from datagen.world_creation.new_godot_scene import ImprovedGodotScene
from datagen.world_creation.region import FloatRange
from datagen.world_creation.region import Region
from datagen.world_creation.types import RawGDObject
from datagen.world_creation.utils import ARRAY_MESH_TEMPLATE
from datagen.world_creation.utils import unnormalize


class TileIdentity(Enum):
    FULL = 0
    ROOM = -1
    HALLWAY = -2
    RAMP = -3
    RAMP_BOTTOM_LANDING = -4
    RAMP_TOP_LANDING = -5
    VOID = -6

    @property
    def pretty_name(self):
        return " ".join([section.lower() for section in self.name.split("_")])


class Orientation(Enum):
    HORIZONTAL = 0
    VERTICAL = 1

    def other(self):
        return Orientation.HORIZONTAL if self == Orientation.VERTICAL else Orientation.VERTICAL


class Azimuth(Enum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    @property
    def angle_from_positive_x(self):
        if self == Azimuth.NORTH:
            return -90
        elif self == Azimuth.EAST:
            return 0
        elif self == Azimuth.SOUTH:
            return 90
        elif self == Azimuth.WEST:
            return 180


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Obstacle:
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Wall(Obstacle):
    story_id: int
    room_id: int
    points: List[Position]
    thickness: int
    height: float


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloorChasm(Obstacle):
    story_id: int
    room_id: int
    points: List[Position]
    thickness: int


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WorkingBlock:
    """Used during 2D block creation when we don't know the full extent of the block"""

    x_min: int
    z_min: int
    x_max: Optional[int] = None
    z_max: Optional[int] = None

    def to_block(self):
        assert self.x_max is not None and self.z_max is not None
        return Block(
            x=NewRange(self.x_min, self.x_max + 1),
            z=NewRange(self.z_min, self.z_max + 1),
            y=NewRange(DEFAULT_FLOOR_THICKNESS, DEFAULT_STORY_HEIGHT + 1),
        )


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Block:
    x: Union[NewRange, FloatRange]
    y: Union[NewRange, FloatRange]
    z: Union[NewRange, FloatRange]


class BlockKind(Enum):
    FLOOR = "floor"
    CEILING = "ceiling"
    WINDOW = "window"
    INTERIOR_WALL = "interior_wall"
    EXTERIOR_WALL = "exterior_wall"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LevelBlock(Block):
    kind: BlockKind
    is_visual: bool = True
    is_collider: bool = True

    @property
    def centroid(self):
        return self.x.midpoint, self.y.midpoint, self.z.midpoint

    @property
    def size(self):
        return self.x.size, self.y.size, self.z.size


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Room:
    id: int
    position: Position
    width: int
    length: int
    outer_height: int
    floor_heightmap: np.ndarray = attr.Factory(
        lambda self: np.ones((self.length, self.width), dtype=np.float), takes_self=True
    )

    def with_heightmap(self, new_heightmap: np.ndarray) -> "Room":
        return attr.evolve(self, floor_heightmap=new_heightmap)

    @property
    def tiles(self):
        return np.full((self.length, self.width), TileIdentity.ROOM.value, dtype=np.int_)

    def with_position(self, new_position: Position) -> "Room":
        return attr.evolve(self, position=new_position)

    def with_id(self, new_id: int) -> "Room":
        return attr.evolve(self, id=new_id)

    @property
    def floor_space(self):
        return self.length * self.width

    @property
    def floor_negative_depth(self):
        return abs(min(0, self.floor_heightmap.min()))

    @property
    def center(self):
        return Position(x=self.position.x + (self.width // 2), z=self.position.z + (self.length // 2))

    @property
    def z_range(self):
        return NewRange(self.position.z, self.position.z + self.length)

    @property
    def x_range(self):
        return NewRange(self.position.x, self.position.x + self.width)


class RoomPlacementError(Exception):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Hallway(Serializable):
    # Hallway points should include joint points with rooms, so we can easily tell if the joint is horizontal / vertical
    points: List[Position]
    from_room_id: int
    to_room_id: int
    width: int

    @property
    def total_length(self) -> int:
        length = 0
        for point_a, point_b in zip(self.points[:-1], self.points[1:]):
            if point_a.x == point_b.x:
                segment_length = abs(point_a.z - point_b.z)
            else:
                segment_length = abs(point_a.x - point_b.x)
            length += segment_length
        return length

    @property
    def from_room_azimuth(self) -> Azimuth:
        """What direction is the entryway into the hallway pointing to, when going from the 'from' room?"""
        return self._azimuth(self.points[0], self.points[1])

    @property
    def to_room_azimuth(self) -> Azimuth:
        """What direction is the entryway into the hallway pointing to, when going from the 'to' room?"""
        return self._azimuth(self.points[-1], self.points[-2])

    def _azimuth(self, point_a: Position, point_b: Position) -> Azimuth:
        if point_a.x == point_b.x:
            return Azimuth.SOUTH if point_a.z < point_b.z else Azimuth.NORTH
        elif point_a.z == point_b.z:
            return Azimuth.EAST if point_a.x < point_b.x else Azimuth.WEST
        else:
            raise SwitchError((point_a, point_b))


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StoryLink(Serializable):
    # todo: multipoint links?
    bottom_story_id: int
    bottom_room_id: int
    bottom_position: Position
    top_story_id: int
    top_room_id: int
    top_position: Position

    def get_geometry(self, building: "Building") -> List[Geometry]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Hole(StoryLink):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Ladder(StoryLink):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Ramp(StoryLink):
    width: int  # width will be expanded right+down from specified positions

    @property
    def slope_run(self):
        # start/end positions are the landings - the actual slope floor length does not include these
        if self.bottom_position.x == self.top_position.x:
            return abs(self.bottom_position.z - self.top_position.z) - 1
        else:
            return abs(self.bottom_position.x - self.top_position.x) - 1

    def get_geometry(self, building: "Building") -> List[Geometry]:
        # todo: width>1
        assert (
            self.bottom_story_id == 0
        ), "todo: if we have 3+ stories, this needs to be offset by bottom stories as well"
        bottom_story: Story = building.stories[self.bottom_story_id]
        bottom_room = bottom_story.rooms[self.bottom_room_id]
        bottom_position_in_room_space = Position(
            self.bottom_position.x - bottom_room.position.x, self.bottom_position.z - bottom_room.position.z
        )
        bottom_story_floor_positive_depth = (
            bottom_room.floor_heightmap[bottom_position_in_room_space.z, bottom_position_in_room_space.x]
            - DEFAULT_FLOOR_THICKNESS
        )
        bottom_story_floor_depth = bottom_story.floor_negative_depth + bottom_story_floor_positive_depth

        top_story: Story = building.stories[self.top_story_id]
        top_room = top_story.rooms[self.top_room_id]
        top_position_in_room_space = Position(
            self.top_position.x - top_room.position.x, self.top_position.z - top_room.position.z
        )
        top_story_floor_depth = (
            top_story.floor_negative_depth
            + top_room.floor_heightmap[top_position_in_room_space.z, top_position_in_room_space.x]
            - DEFAULT_FLOOR_THICKNESS
        )

        slope_width = self.width
        slope_run = self.slope_run
        slope_rise = (bottom_story.outer_height - bottom_story_floor_positive_depth) + top_story_floor_depth
        slope_length = math.sqrt(slope_run ** 2 + slope_rise ** 2)
        slope_angle = math.atan(slope_rise / slope_run)
        slope_thickness = SLOPE_THICKNESS

        complementary_angle = math.radians(90) - slope_angle
        y_offset = (slope_thickness / 2) * math.sin(complementary_angle)
        xz_offset = (slope_thickness / 2) * math.cos(complementary_angle)

        is_horizontal = self.bottom_position.z == self.top_position.z
        # +1s are to offset the slope from its bottom landing
        if is_horizontal:
            plane_width = slope_length
            plane_length = slope_width
            angle_multiplier = 1 if self.top_position.x > self.bottom_position.x else -1
            pitch, roll = 0, angle_multiplier * math.degrees(slope_angle)
            x = min(self.bottom_position.x, self.top_position.x) + (slope_run / 2) + 1 + xz_offset
            z = min(self.bottom_position.z, self.top_position.z) + (slope_width / 2)
        else:
            plane_width = slope_width
            plane_length = slope_length
            angle_multiplier = -1 if self.top_position.z > self.bottom_position.z else 1
            pitch, roll = angle_multiplier * math.degrees(slope_angle), 0
            x = min(self.bottom_position.x, self.top_position.x) + (slope_width / 2)
            z = min(self.bottom_position.z, self.top_position.z) + (slope_run / 2) + 1 + xz_offset

        y = math.tan(slope_angle) * (slope_run / 2) - y_offset + bottom_story_floor_depth

        return [Box(x, y, z, plane_width, plane_length, slope_thickness, pitch, 0, roll)]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Window(Serializable):
    position: np.ndarray  # centroid
    size: np.ndarray
    is_passable: bool = False  # should the cutout have a collision shape


class CornerType(Enum):
    NE = 0
    SE = 1
    SW = 2
    NW = 3


class WallType(Enum):
    N = 0
    E = 1
    S = 2
    W = 3


kernels_by_corner = {
    (CornerType.SW, True): np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
    (CornerType.SE, True): np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
    (CornerType.NW, True): np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]),
    (CornerType.NE, True): np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]),
    # todo: inside corners?
}

kernels_by_wall = {
    (WallType.N, False): np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]]),
    (WallType.E, True): np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
    (WallType.S, False): np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]]),
    (WallType.W, True): np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]]),
}

corner_types_by_wall_type = {
    WallType.N: (CornerType.NW, CornerType.NE),
    WallType.E: (CornerType.NE, CornerType.SE),
    WallType.S: (CornerType.SE, CornerType.SW),
    WallType.W: (CornerType.NW, CornerType.SW),
}


def find_exterior_corners(tiles):
    """more elegant than below, but unordered; remove later"""
    corners = []
    for (corner_type, is_outside), kernel in kernels_by_corner.items():
        out_of_bounds = set(zip(*np.where(tiles == False)))
        convolution = convolve(tiles.astype(np.int8), kernel, mode="constant")
        possible_corner_positions = set(zip(*np.where(convolution == 0)))
        corner_positions = possible_corner_positions - out_of_bounds
        corners.extend([(position, corner_type, is_outside) for position in corner_positions])
    return corners


def find_corners(tiles: np.ndarray, SOLID: int = 1) -> List[Tuple[Tuple[int, int], CornerType, bool]]:
    length, width = tiles.shape
    current_direction = Azimuth.EAST
    z, x = 0, 0
    done = False
    corners = []
    while not done:
        if not corners:
            if tiles[z, x] == SOLID:
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

            if tile != SOLID:
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
            new_corners = []
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


def flipped(tile):
    return tile[1], tile[0]


class WallFootprint(NamedTuple):
    top_left: Tuple[int, int]  # x, z
    bottom_right: Tuple[int, int]  # x, z
    wall_type: WallType
    is_vertical: bool

    @property
    def wall_thickness(self):
        return self.footprint_width if self.is_vertical else self.footprint_length

    @property
    def wall_length(self):
        return self.footprint_length if self.is_vertical else self.footprint_width

    @property
    def footprint_width(self):
        return self.bottom_right[0] - self.top_left[0]

    @property
    def footprint_length(self):
        return self.bottom_right[1] - self.top_left[1]

    @property
    def centroid(self):
        return np.array(self.top_left) + (np.array(self.bottom_right) - np.array(self.top_left)) / 2


def find_exterior_wall_footprints(
    tiles: np.ndarray, no_build_mask: Optional[np.ndarray] = None
) -> List[WallFootprint]:
    walls = []
    for (wall_type, is_vertical), wall_kernel in kernels_by_wall.items():
        out_of_bounds = tiles == False
        wall_or_corner_mask = convolve(tiles.astype(np.int8), wall_kernel, mode="constant") == 0
        wall_mask = ~out_of_bounds & wall_or_corner_mask
        for corner_type in corner_types_by_wall_type[wall_type]:
            corner_kernel = kernels_by_corner[(corner_type, True)]
            corner_mask = convolve(tiles.astype(np.int8), corner_kernel, mode="constant") == 0
            wall_mask &= ~corner_mask

        # Hallways and story links (ramps) intersecting with exterior walls are entrances/exits; if mask is passed,
        # we exclude them; e.g. to avoid building windows there.
        if no_build_mask is not None:
            wall_mask &= ~no_build_mask

        if is_vertical:
            wall_mask = wall_mask.T
        wall_tiles = list(zip(*np.where(wall_mask)))

        wall = None
        for current_tile, next_tile in zip(wall_tiles[:-1], wall_tiles[1:]):
            if not wall:
                z, x = current_tile
                wall = [(z, x) if is_vertical else (x, z)]

            end_tile = None
            if current_tile[1] + 1 != next_tile[1]:
                end_tile = current_tile
            elif wall is not None and next_tile == wall_tiles[-1]:
                end_tile = next_tile

            if end_tile:
                z, x = end_tile
                z += 1
                x += 1
                wall.append((z, x) if is_vertical else (x, z))
                walls.append(WallFootprint(wall[0], wall[1], wall_type, is_vertical))
                wall = None
    return walls


class FootprintType(Enum):
    RECTANGLE = "rectangle"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    IRREGULAR = "irregular"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Entrance(Serializable):
    story_num: int
    connected_room_id: int
    azimuth: Azimuth
    points: List[Position]
    width: int = 1

    exterior_id_by_azimuth = {
        Azimuth.NORTH: -1,
        Azimuth.SOUTH: -2,
        Azimuth.WEST: -3,
        Azimuth.EAST: -4,
    }

    @property
    def hallway(self):
        return Hallway(self.points, Entrance.exterior_id_by_azimuth[self.azimuth], self.connected_room_id, self.width)

    def get_points_in_story_outline(self, story: "Story") -> List[Tuple]:
        outline_mask = story.get_outline_mask()
        entrance_mask = np.zeros_like(outline_mask)
        draw_line_in_grid(self.points, entrance_mask, TileIdentity.HALLWAY.value, include_ends=False)
        entrance_mask_points_in_outline = np.where(outline_mask & entrance_mask)
        return list(zip(*entrance_mask_points_in_outline))

    def get_connected_room_and_landing_position(self, building: "Building") -> Tuple[Room, Position]:
        story = building.stories[self.story_num]
        room = story.rooms[self.connected_room_id]
        entrance_grid = np.zeros((story.length, story.width))
        draw_line_in_grid(self.points, entrance_grid, TileIdentity.HALLWAY.value, include_ends=True)
        room_grid = np.zeros_like(entrance_grid)
        set_room_tiles_in_grid(room, room.tiles, room_grid)
        entrance_hallway_overlaps_with_room = entrance_grid.astype(np.bool_) & room_grid.astype(np.bool_)
        room_landing_coords = list(zip(*np.where(entrance_hallway_overlaps_with_room)))
        # translate to room space
        room_landing = list(room_landing_coords[0])
        room_landing[0] -= room.position.z
        room_landing[1] -= room.position.x
        return room, Position(x=room_landing[1], z=room_landing[0])


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Story(Serializable):
    """
     ___________
    |__________| \  } ceiling thickness (1), inset into room height
    |          |  \                     \
    |          |   } outer height (5);   } inner height = 5-1-1=3
    |____  ____|  |                     /
    |___| |____| /   } positive floor depth (1), offset INTO room height; bottom line is 0-level
    |   | |    | \
    |   | |    |  } negative floor depth (3)
    |___|_|____| /
    """

    num: int
    width: int
    length: int
    footprint: np.ndarray = attr.Factory(
        lambda self: np.ones((self.length, self.width), dtype=np.bool_), takes_self=True
    )
    rooms: List[Room] = attr.Factory(list)
    hallways: List[Hallway] = attr.Factory(list)
    story_links: List[StoryLink] = attr.Factory(list)
    windows: List[Window] = attr.Factory(list)
    entrances: List[Entrance] = attr.Factory(list)
    has_ceiling: bool = False

    @property
    def outer_height(self):
        return max([r.outer_height for r in self.rooms])

    @property
    def inner_height(self):
        # The EXACT inner height can vary tile by tile depending on terrain; this is the "base" inner height assuming
        # the floor uses DEFAULT_FLOOR_THICKNESS.
        return self.outer_height - DEFAULT_FLOOR_THICKNESS - (CEILING_THICKNESS if self.has_ceiling else 0)

    @property
    def floor_negative_depth(self):
        # Note that all positive story terrain is projected INTO the room (rather than offsetting the room & walls),
        # see diagram above
        return max([room.floor_negative_depth for room in self.rooms])

    @property
    def floor_heightmap(self):
        """returns positive floor thickness for all walkable tiles, np.nan elsewhere"""
        tiles = np.full((self.length, self.width), np.nan, dtype=np.float32)
        for room in self.rooms:
            set_room_tiles_in_grid(room, room.floor_heightmap, tiles)
        for hallway in self.hallways:
            draw_line_in_grid(hallway.points, tiles, DEFAULT_FLOOR_THICKNESS, include_ends=False, filter_values=None)
        return tiles

    def get_room_at_point(self, point: Vector2) -> Optional[Room]:
        # todo: use kdtree or something more efficient?
        for room in self.rooms:
            if (
                room.position.x <= point.x <= room.position.x + room.width - 1
                and room.position.z <= point.y <= room.position.z + room.length - 1
            ):
                return room
        return None

    def get_outline_mask(self):
        is_building = self.footprint.astype(np.int) != 0
        convolution = scipy.ndimage.convolve(self.footprint.astype(np.int), np.ones((3, 3)), mode="constant")
        return is_building & (convolution != 9)

    def get_exterior_wall_azimuths_by_room_id(self) -> Dict[int, List[Azimuth]]:
        outline_mask = self.get_outline_mask()
        exterior_wall_azimuths_by_room_id = {}
        for room in self.rooms:
            room_has_exterior_wall_by_azimuth = {
                Azimuth.NORTH: outline_mask[room.position.z - 1, room.position.x],
                Azimuth.SOUTH: outline_mask[room.position.z + room.length, room.position.x],
                Azimuth.EAST: outline_mask[room.position.z, room.position.x + room.width],
                Azimuth.WEST: outline_mask[room.position.z, room.position.x - 1],
            }
            exterior_wall_azimuths_by_room_id[room.id] = [
                azimuth
                for azimuth, has_exterior_wall in room_has_exterior_wall_by_azimuth.items()
                if has_exterior_wall
            ]
        return exterior_wall_azimuths_by_room_id


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BuildingAestheticsConfig:
    desired_story_count: int = 2
    rail_height: float = 0.75
    rail_thickness: float = 0.35
    rail_overhang_factor: float = 0.8  # 0-1 scalar of how much the building-top rail should hang over the edge
    window_height: float = 2
    window_width: float = 1.5
    window_min_gap: float = 1
    window_y_proportion_of_height = 0.55  # where to place window centroid, expressed as % of story height
    max_windows_per_wall: int = 8
    crossbeam_size: float = 0.35
    crossbeam_protrusion: float = 0.75
    crossbeam_min_gap: float = 2.5
    crossbeam_y_proportion_of_height: float = 0.9  # where to place beam centroid, expressed as % of story height
    # colors are rgba, [0, 255]
    # these are here to make the colors programmatic for nicole to tweak
    # crossbeam_color: Tuple[int, int, int, int] = (255, 0, 0, 255)
    # exterior_color: Tuple[int, int, int, int] = (0, 255, 0, 255)
    # trim_color: Tuple[int, int, int, int] = (0, 0, 255, 255)
    crossbeam_color: Tuple[int, int, int, int] = (0.19462, 0.13843, 0.02624, 1.0)  # (0.12477, 0.13014, 0.13287, 1.0)
    exterior_color: Tuple[int, int, int, int] = (0.66539, 0.56471, 0.28744, 1.0)  # (0.46778, 0.48515, 0.50289, 1.0)
    trim_color: Tuple[int, int, int, int] = (0.88792, 0.78354, 0.50289, 1.0)  # (0.57647, 0.57647, 0.57647, 1.0)
    interior_wall_color: Tuple[int, int, int, int] = (
        0.66539,
        0.56471,
        0.28744,
        1.0,
    )  # (0.57647, 0.57647, 0.57647, 1.0)
    floor_color: Tuple[int, int, int, int] = (0.88792, 0.78354, 0.50289, 1.0)  # (0.46778, 0.48515, 0.50289, 1.0)
    ceiling_color: Tuple[int, int, int, int] = (0.88792, 0.78354, 0.50289, 1.0)  # (0.46778, 0.48515, 0.50289, 1.0)
    block_downsize_epsilon: float = 1e-5  # By how much to reduce block extents to avoid coincident faces


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Building(Serializable):
    id: int
    stories: List[Story]
    story_links: List[StoryLink]
    position: Vector3
    yaw_degrees: float = 0
    is_climbable: bool = False
    aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig()

    @property
    def region(self):
        length = max([story.length for story in self.stories])
        width = max([story.width for story in self.stories])
        return Region(
            x=FloatRange(self.position.x, self.position.x + width),
            z=FloatRange(self.position.z, self.position.z + length),
        )

    @property
    def height(self):
        return FloatRange(self.position.y, self.position.y + sum([s.outer_height for s in self.stories]))

    @property
    def width(self):
        return self.stories[0].width

    @property
    def length(self):
        return self.stories[0].length

    @property
    def offset_point(self):
        # Where to offset any building space coordinates from to get the global ones
        return Vector3(self.position.x - self.width / 2, self.position.y, self.position.z - self.length / 2)

    def with_transform(
        self, new_position: Optional[Vector3] = None, new_yaw_degrees: Optional[float] = None
    ) -> "Building":
        kwargs = {}
        if new_position is not None:
            kwargs["position"] = new_position
        if new_yaw_degrees is not None:
            kwargs["yaw_degrees"] = new_yaw_degrees
        return attr.evolve(self, **kwargs)

    def get_footprint_outline(self, story_num=0) -> Polygon:
        """Returns a 2D outline (y is always 0) of first story footprint outline. Z is mapped to Y in the polygon."""
        tile_size = 1
        story = self.stories[story_num]
        corners = find_corners(story.footprint)
        points = []
        for local_position, corner_type, is_outside in corners:
            position = local_to_global_coords(
                np.array([local_position[1] + tile_size / 2, 0, local_position[0] + tile_size / 2]), self.offset_point
            )
            # NB! Shapely fecking ignores the Z coordinate, so we must make sure to only pass two coordinates here
            # We pass them as X -> X and Z -> Y
            points.append(Point(position[0], position[2]))
        outline = Polygon(points)

        # Scale polygon to match OUTER edge of building, not tile midline
        min_x, min_y, max_x, max_y = outline.bounds
        width, length = max_x - min_x, max_y - min_y
        tile_size = 1
        x_scaling_factor = (width + tile_size) / width
        z_scaling_factor = (length + tile_size) / length
        outline = affinity.scale(outline, x_scaling_factor, z_scaling_factor)

        building_centroid_2d = self.position.x, self.position.z
        outline = affinity.rotate(outline, -self.yaw_degrees, origin=building_centroid_2d)
        return outline

    def get_positive_height_at_point(self, point: Point) -> float:
        """
        Positive height refers to the exterior building height from it's y origin.
        Note: param point must be in world space.
        Returns np.nan if point lies outside map.
        """
        building_centroid_2d = (
            self.position.x + self.stories[0].width / 2,
            self.position.z + self.stories[0].length / 2,
        )
        unrotated_point = affinity.rotate(point, -self.yaw_degrees, origin=building_centroid_2d)
        point_in_building_space = global_to_local_coords(
            np.array([unrotated_point.x, 0, unrotated_point.y]), self.position
        )
        tile = math.floor(point_in_building_space[0]), math.floor(point_in_building_space[2])
        positive_height_at_point = 0
        for story in self.stories:
            out_of_bounds = tile[0] < 0 or tile[0] > story.width - 1 or tile[1] < 0 or tile[1] > story.length - 1
            if out_of_bounds:
                if story.num == 0:
                    return np.nan
                else:
                    continue
            elif story.footprint[tile[1], tile[0]]:
                positive_height_at_point += story.outer_height
                if story.num != 0:
                    # Negative terrain cannot be underground for non-ground floors.
                    positive_height_at_point += story.floor_negative_depth
        return positive_height_at_point

    def translate_block_to_building_space(self, block: LevelBlock, story: Story, epsilon: float = 0) -> LevelBlock:
        x_offset = -story.width / 2
        y_offset = self.get_story_y_offset(story.num)
        z_offset = -story.length / 2

        width, height, length = block.size
        adjusted_size = width - epsilon, height - epsilon, length - epsilon
        x = x_offset + block.x.min_ge + width / 2
        y = y_offset + block.y.min_ge + height / 2
        z = z_offset + block.z.min_ge + length / 2
        centroid = x, y, z
        return make_block(block.kind, centroid, adjusted_size, block.is_visual, block.is_collider)

    def export(self, scene: ImprovedGodotScene, parent_node: GDNode, building_name: str, scaling_factor: float = 1):
        blocks_by_story_by_kind = {}
        for story in self.stories:
            blocks_by_story_by_kind[story.num] = self._generate_blocks(story.num)

        building_material = scene.add_sub_resource("SpatialMaterial", vertex_color_use_as_albedo=True, roughness=0.5)
        building_script = scene.add_ext_resource("res://entities/building.gd", "Script")

        rotation = Rotation.from_euler("y", self.yaw_degrees, degrees=True).as_matrix().flatten()
        building_node = GDNode(
            building_name,
            type="Spatial",
            properties={
                "transform": make_transform(position=self.position, rotation=rotation),
                "script": building_script.reference,
                "is_climbable": self.is_climbable,
            },
        )
        parent_node.add_child(building_node)

        mesh_group = GDNode("meshes", type="Spatial")
        building_node.add_child(mesh_group)

        DEFAULT_MESH = HIGH_POLY
        meshes = make_pretty_building_meshes(self, blocks_by_story_by_kind)
        for mesh_name, mesh in meshes.items():

            def _print_array(array, precision):
                return ", ".join([f"{x:.{precision}f}" for x in array.flatten()])

            mesh_kwargs = {}
            colors = mesh.visual.vertex_colors / 255
            mesh_kwargs[f"surfaces/0"] = RawGDObject(
                ARRAY_MESH_TEMPLATE.format(
                    aabb=_print_array(mesh.bounds, 3),
                    vertex_floats=_print_array(mesh.vertices, 3),
                    vertex_normal_floats=_print_array(mesh.vertex_normals, 3),
                    color_floats=_print_array(colors, 5),
                    triangle_indices=_print_array(mesh.faces, 0),
                    index_count=str(len(mesh.faces) * 3),
                    vertex_count=str(len(mesh.vertices) * 3),
                    material_resource_type="SubResource"
                    if isinstance(building_material, GDSubResourceSection)
                    else "ExtResource",
                    material_id=building_material.id,
                    mesh_name=building_name,
                )
            )
            building_mesh = scene.add_sub_resource("ArrayMesh", resource_name="terrain", **mesh_kwargs)
            is_visible = mesh_name == DEFAULT_MESH
            mesh_node = GDNode(
                mesh_name,
                type="MeshInstance",
                properties={"mesh": building_mesh.reference, "material/0": "null", "visible": is_visible},
            )
            mesh_group.add_child(mesh_node)

        light_group = GDNode("lights", type="Spatial")
        building_node.add_child(light_group)

        static_body_group = GDNode("static_bodies", type="Spatial")
        building_node.add_child(static_body_group)

        for story_num, blocks_by_kind in blocks_by_story_by_kind.items():
            story = self.stories[story_num]

            # Uncomment block below to add an omnilight to the center of each story
            # light_centroid_y = self.get_story_y_offset(story_num) + DEFAULT_FLOOR_THICKNESS + story.inner_height / 2
            # light_centroid = 0, light_centroid_y, 0
            # indoor_light = GDNode(
            #     f"OmniLight{story.num}",
            #     type="OmniLight",
            #     properties={
            #         "transform": make_transform(position=light_centroid),
            #         "omni_range": math.sqrt(story.width ** 2 + story.length ** 2),
            #     },
            # )
            # light_group.add_child(indoor_light)

            for kind, blocks in blocks_by_kind.items():
                for block in blocks:
                    if not block.is_collider:
                        # When generating meshes, we only care for collider blocks here
                        continue

                    block_in_building_space = self.translate_block_to_building_space(block, story)
                    width, height, length = block_in_building_space.size
                    box_shape = scene.add_sub_resource(
                        "BoxShape",
                        extents=godot_parser.Vector3(width / 2, height / 2, length / 2),
                    )

                    static_body = GDNode(
                        f"StaticBody{box_shape.reference.id}",
                        type="StaticBody",
                        properties={
                            "transform": make_transform(position=block_in_building_space.centroid),
                            "shape": box_shape.reference,
                        },
                    )
                    collision_shape = GDNode(
                        f"CollisionShape{box_shape.reference.id}",
                        type="CollisionShape",
                        properties={
                            "transform": make_transform(),
                            "shape": box_shape.reference,
                        },
                    )
                    static_body.add_child(collision_shape)
                    static_body_group.add_child(static_body)

        # todo: clean up repetition here if we keep export_old
        material = scene.add_ext_resource("res://shaders/BasicColor.material", "Material")
        story_links_node = GDNode(f"StoryLinks", type="Spatial")
        building_node.add_child(story_links_node)
        for story_link in self.story_links:
            bottom_story = self.stories[story_link.bottom_story_id]
            story_link_geometry = story_link.get_geometry(self)
            for entity in story_link_geometry:
                # todo: make add function generic
                add_box_to_scene(
                    entity,
                    Vector3(
                        -bottom_story.width / 2,
                        0,
                        -bottom_story.length / 2,
                    ),
                    scene,
                    story_links_node,
                    material,
                    scaling_factor,
                    include_mesh=False,
                )

    def export_old(self, building_name: str, output_file_path: Path, scaling_factor: float = 1) -> Path:
        scene = ImprovedGodotScene()
        material = scene.add_ext_resource("res://shaders/BasicColor.material", "Material")
        blocks_by_story = self._generate_blocks_by_story()
        centroids_and_sizes_by_story = self.get_block_details_by_story(blocks_by_story)

        with scene.use_tree() as tree:
            rotation = Rotation.from_euler("y", self.yaw_degrees, degrees=True).as_matrix().flatten()
            tree.root = GDNode(
                building_name,
                type="Spatial",
                properties={"transform": make_transform(position=self.position, rotation=rotation)},
            )

            for story_num, centroids_and_sizes in centroids_and_sizes_by_story.items():
                story_node = GDNode(f"Story{story_num}", type="Spatial")
                tree.root.add_child(story_node)

                story = self.stories[story_num]
                centroid = (
                    story.width / 2,
                    self.get_story_y_offset(story_num) + DEFAULT_FLOOR_THICKNESS + story.inner_height / 2,
                    story.length / 2,
                )
                indoor_light = GDNode(
                    f"OmniLight{story_num}",
                    type="OmniLight",
                    properties={
                        "transform": make_transform(position=centroid),
                        "omni_range": math.sqrt(story.width ** 2 + story.length ** 2),
                    },
                )
                story_node.add_child(indoor_light)

                for centroid, size in centroids_and_sizes:
                    width, height, length = size
                    cube_mesh = scene.add_sub_resource("CubeMesh", size=godot_parser.Vector3(width, height, length))
                    box_shape = scene.add_sub_resource(
                        "BoxShape",
                        extents=godot_parser.Vector3(width / 2, height / 2, length / 2),
                    )

                    static_body = GDNode(
                        f"StaticBody{cube_mesh.reference.id}",
                        type="StaticBody",
                        properties={"transform": make_transform(position=centroid), "shape": box_shape.reference},
                    )
                    mesh_instance = GDNode(
                        f"Block{cube_mesh.reference.id}",
                        type="MeshInstance",
                        properties={
                            "transform": make_transform(),
                            "mesh": cube_mesh.reference,
                            "material/0": material.reference,
                        },
                    )
                    collision_shape = GDNode(
                        f"CollisionShape{cube_mesh.reference.id}",
                        type="CollisionShape",
                        properties={
                            "transform": make_transform(),
                            "shape": box_shape.reference,
                        },
                    )
                    static_body.add_child(mesh_instance)
                    static_body.add_child(collision_shape)
                    story_node.add_child(static_body)

            story_links_node = GDNode(f"StoryLinks", type="Spatial")
            tree.root.add_child(story_links_node)
            for story_link in self.story_links:
                story_link_geometry = story_link.get_geometry(self)
                for entity in story_link_geometry:
                    # todo: make add function generic
                    add_box_to_scene(entity, Vector3(0, 0, 0), scene, story_links_node, material, scaling_factor)

        scene.write(str(output_file_path.absolute()))
        return output_file_path

    def generate_tiles(self, story_num: int, include_floor_bottom_links: bool = True) -> np.ndarray:
        story = self.stories[story_num]
        tiles = np.invert(story.footprint).astype(np.float32) * TileIdentity.VOID.value
        for room in story.rooms:
            set_room_tiles_in_grid(room, room.tiles, tiles)
        for hallway in story.hallways:
            # `Hallway` points include the joint points with rooms, which we don't want to draw them, so we trim off
            # the first point of the first connection and the last point of the last connection
            draw_line_in_grid(hallway.points, tiles, TileIdentity.HALLWAY.value, include_ends=False)
        for link in self.story_links:
            if link.bottom_story_id == story_num and include_floor_bottom_links:
                set_link_in_grid(link, tiles, bottom=True)
            elif link.top_story_id == story_num:
                set_link_in_grid(link, tiles, bottom=False)
        return tiles

    def _generate_blocks(
        self, story_num: int, kinds: Tuple[BlockKind, ...] = tuple(BlockKind)
    ) -> Dict[BlockKind, List[LevelBlock]]:
        blocks_by_kind = {}
        for kind in kinds:
            if kind == BlockKind.FLOOR:
                blocks = [*self._generate_terrain_blocks(story_num), *self._generate_hallway_floor_blocks(story_num)]
            elif kind == BlockKind.EXTERIOR_WALL:
                blocks = self._generate_exterior_wall_blocks(story_num)
            elif kind == BlockKind.INTERIOR_WALL:
                blocks = self._generate_interior_wall_blocks(story_num)
            elif kind == BlockKind.CEILING:
                story = self.stories[story_num]
                if story.has_ceiling:
                    blocks = self._generate_ceiling_blocks(story_num)
                else:
                    blocks = []
            elif kind == BlockKind.WINDOW:
                blocks = self._generate_window_blocks(story_num)
            else:
                raise SwitchError(kind)
            blocks_by_kind[kind] = blocks
        return blocks_by_kind

    def get_story_y_offset(self, requested_story_num: int) -> float:
        # Returns the y position for where it's positive floor should begin
        assert requested_story_num <= len(self.stories)

        y = -DEFAULT_FLOOR_THICKNESS  # so that y=0 usually equates to top of ground story floor
        for i, story in enumerate(self.stories):
            if story.num > 0:
                y += story.floor_negative_depth
            if story.num == requested_story_num:
                return y
            y += story.outer_height

    def _generate_interior_wall_blocks(self, story_num: int) -> List[LevelBlock]:
        story = self.stories[story_num]
        wall_blocks = []
        story_tiles = self.generate_tiles(story_num)
        is_outline = story.get_outline_mask()
        is_wall = np.invert(story_tiles.astype(np.bool_))
        interior_wall_tiles = is_wall & ~is_outline
        interior_wall_blocks = generate_blocks(interior_wall_tiles, SOLID=True)

        for block in interior_wall_blocks:
            stretched_block = LevelBlock(
                block.x,
                NewRange(-story.floor_negative_depth, story.outer_height),
                block.z,
                kind=BlockKind.INTERIOR_WALL,
            )
            cut_blocks = self._cut_windows(stretched_block, story.windows)
            wall_blocks.extend(cut_blocks)
        return wall_blocks

    def _generate_exterior_wall_blocks(self, story_num: int) -> List[LevelBlock]:
        story = self.stories[story_num]
        wall_blocks = []
        story_tiles = self.generate_tiles(story_num)
        is_outline = story.get_outline_mask()

        exterior_wall_tiles = np.full_like(story_tiles, fill_value=TileIdentity.ROOM.value)
        exterior_wall_tiles[is_outline] = story_tiles[is_outline]
        exterior_wall_blocks = generate_blocks(exterior_wall_tiles)

        for block in exterior_wall_blocks:
            offset_block = LevelBlock(
                block.x,
                NewRange(-story.floor_negative_depth, story.outer_height),
                block.z,
                kind=BlockKind.EXTERIOR_WALL,
            )
            cut_blocks = self._cut_windows(offset_block, story.windows)
            wall_blocks.extend(cut_blocks)
        return wall_blocks

    def _generate_hallway_floor_blocks(self, story_num: int) -> List[LevelBlock]:
        # TODO: To cover all cases, we actually want to build each hallway separately, making staircases between rooms
        #  of different heights. For now we assume that hallways only connect height=1(default floor thickness) terrain.
        story = self.stories[story_num]
        hallway_tiles = np.zeros((story.length, story.width))
        for hallway in story.hallways:
            draw_line_in_grid(hallway.points, hallway_tiles, TileIdentity.HALLWAY.value, include_ends=False)

        room_tiles = np.zeros((story.length, story.width))
        for room in story.rooms:
            set_room_tiles_in_grid(room, room.tiles, room_tiles)
        no_room_tiles = ~room_tiles.astype(np.bool_)

        # Ensure no hallway blocks are placed outside footprint (e.g. for entrance hallways leading to exterior
        # Also ensure we're only building hallway blocks outside rooms to avoid z-fighting
        hallway_tiles = (hallway_tiles.astype(np.bool_) & story.footprint & no_room_tiles).astype(np.float64)
        hallway_tiles[hallway_tiles == 1] = TileIdentity.HALLWAY.value

        return [
            LevelBlock(
                block.x, NewRange(-story.floor_negative_depth, DEFAULT_FLOOR_THICKNESS), block.z, kind=BlockKind.FLOOR
            )
            for block in generate_blocks(hallway_tiles, SOLID=TileIdentity.HALLWAY.value)
        ]

    def _generate_terrain_blocks(self, story_num: int) -> List[LevelBlock]:
        # "Terrain" for rooms and buildings works as follows:
        # 0 is ground level - a floor heightmap of all 0s equates to no floor blocks. Walls are built atop this level.
        # >=1 is positive terrain - blocks built inside the height of the room. A floor heightmap equating to wall
        #   height would end up in a non-existent room (floor touches ceiling).
        # <=-1 is chasms (gaps) downward into the floor, and only really make sense if you have some non-negative
        #   terrain. Walls blocks and positive terrain are all extended downwards to enclose the chasms.

        all_blocks = []
        story = self.stories[story_num]
        min_height = -story.floor_negative_depth

        ramp_cut_tiles = self.generate_tiles(story_num, include_floor_bottom_links=False)
        ramp_cut_tiles[ramp_cut_tiles != TileIdentity.RAMP.value] = 0

        for room in story.rooms:
            # The floor heightmap gets overridden by cuts necessary in the floor to accommodate ramps leading up here
            tiles = room.floor_heightmap.copy()
            room_cut_tiles = ramp_cut_tiles[
                room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width
            ]
            room_cut_tiles = list(zip(*[list(a) for a in room_cut_tiles.nonzero()]))
            for z, x in room_cut_tiles:
                tiles[z, x] = min_height

            unique_heights = np.unique(tiles)
            for height in unique_heights:
                if height == min_height:
                    continue
                tiles_for_height = tiles.copy()
                tiles_for_height[tiles_for_height == height] = np.inf
                blocks = generate_blocks(tiles_for_height, SOLID=np.inf)
                # todo: consolidate range types; make block type all floatranges?
                for block in blocks:
                    offset_block = LevelBlock(
                        x=NewRange(block.x.min_ge + room.position.x, block.x.max_lt + room.position.x),
                        y=FloatRange(min_height, height),
                        z=NewRange(block.z.min_ge + room.position.z, block.z.max_lt + room.position.z),
                        kind=BlockKind.INTERIOR_WALL,
                    )
                    cut_blocks = self._cut_windows(offset_block, story.windows)
                    all_blocks.extend(cut_blocks)
        return all_blocks

    def _cut_windows(self, parent_block: LevelBlock, windows: List[Window]) -> List[LevelBlock]:
        window_blocks = []
        # TODO: improve efficiency - we shouldn't need to do a O(N^2) loop with these
        for window in windows:
            window_block_in_story_space = make_block(BlockKind.WINDOW, window.position, window.size)
            if (
                window_block_in_story_space.x.overlap(parent_block.x)
                and window_block_in_story_space.y.overlap(parent_block.y)
                and window_block_in_story_space.z.overlap(parent_block.z)
            ):
                local_coords = global_to_local_coords(
                    window.position,
                    np.array([parent_block.x.min_ge, parent_block.y.min_ge, parent_block.z.min_ge]),
                )
                window_block_in_parent_block_space = make_block(BlockKind.WINDOW, local_coords, window.size)
                window_blocks.append(window_block_in_parent_block_space)
        if len(window_blocks) > 0:
            return self._cut_blocks(parent_block, window_blocks)
        else:
            return [parent_block]

    def _cut_blocks(self, parent_block: LevelBlock, child_blocks: List[LevelBlock]) -> List[LevelBlock]:
        """
        Cut child_blocks as holes in parent_block in 2D
        is_visual / is_collider properties in parent are maintained in cut blocks.
        NOTE: child block coordinates must be localized to start at parent block origin
        """
        cut_axis = None
        for child_block in child_blocks:
            if child_block.x.size >= parent_block.x.size:
                block_cut_axis = Axis.X
            elif child_block.y.size >= parent_block.y.size:
                block_cut_axis = Axis.Y
            elif child_block.z.size >= parent_block.z.size:
                block_cut_axis = Axis.Z
            else:
                raise ValueError("Child blocks need to exceed parent block size in at least one dimension")
            if cut_axis is None:
                cut_axis = block_cut_axis
            elif cut_axis != block_cut_axis:
                raise NotImplementedError(
                    "3d cutting is not supported yet; all child blocks must cut on the same axis"
                )

        if cut_axis == Axis.X:
            first_axis, second_axis = Axis.Y, Axis.Z
        elif cut_axis == Axis.Y:
            first_axis, second_axis = Axis.X, Axis.Z
        else:
            first_axis, second_axis = Axis.X, Axis.Y

        # "tail" represents the leftover of the original block that doesn't fit into the tile_size=1 grid
        # we add it back to the blocks that touch the right or bottom of the grid when building the blocks below
        first_axis_size = getattr(parent_block, first_axis.value).size
        first_axis_tail = math.ceil(first_axis_size) - first_axis_size
        second_axis_size = getattr(parent_block, second_axis.value).size
        second_axis_tail = math.ceil(second_axis_size) - second_axis_size

        mult = 3  # resolution_multiplier - by what factor are we scaling up the grid's resolution
        grid = np.zeros((math.ceil(first_axis_size) * mult, math.ceil(second_axis_size) * mult))
        for child_block in child_blocks:
            first_range = getattr(child_block, first_axis.value)
            second_range = getattr(child_block, second_axis.value)
            first_range_min = round(first_range.min_ge * mult)
            if first_range_min < 0:
                first_range_min = 0
            second_range_min = round(second_range.min_ge * mult)
            if second_range_min < 0:
                second_range_min = 0

            grid[
                first_range_min : round(first_range.max_lt * mult),
                second_range_min : round(second_range.max_lt * mult),
            ] = 1

        cut_blocks = generate_blocks(grid)
        # NOTE: the axis naming here is tricky:
        # `generate_blocks` uses its own internal axis naming: first axis is Z, second axis is X.
        # The meaning of those axes is different (since our cut axis arbitrary), so we need to map these back to our
        # meaning of first/second axis.
        # e.g. if we're cutting on X, the primary axis is Y and secondary is Z
        # the blocks we get back from generate_blocks were tiled as (z, x), so the mapping back is
        # z -> Y and x -> Z

        resulting_blocks = []
        for block in cut_blocks:
            # cut axis is untouched - we just set to whatever the parent had
            resulting_block = LevelBlock(
                kind=parent_block.kind,
                x=NewRange(0, 0),
                y=NewRange(0, 0),
                z=NewRange(0, 0),
                is_collider=parent_block.is_collider,
                is_visual=parent_block.is_visual,
            )
            cut_axis_range: NewRange = getattr(parent_block, cut_axis.value)
            setattr(resulting_block, cut_axis.value, cut_axis_range)

            # Note that we need to scale all child block axis ranges down by dividing by mult
            first_axis_range = block.z
            parent_first_axis_offset = getattr(parent_block, first_axis.value).min_ge
            end_offset = parent_first_axis_offset
            if (
                first_axis_range.max_lt / mult + end_offset
                == getattr(parent_block, first_axis.value).max_lt - first_axis_tail
            ):
                end_offset += first_axis_tail
            offset_block_range = NewRange(
                first_axis_range.min_ge / mult + parent_first_axis_offset,
                first_axis_range.max_lt / mult + end_offset,
            )
            setattr(resulting_block, first_axis.value, offset_block_range)

            second_axis_range = block.x
            parent_second_axis_offset = getattr(parent_block, second_axis.value).min_ge
            end_offset = parent_second_axis_offset
            if (
                second_axis_range.max_lt / mult + end_offset
                == getattr(parent_block, second_axis.value).max_lt - second_axis_tail
            ):
                end_offset += second_axis_tail
            offset_block_range = NewRange(
                second_axis_range.min_ge / mult + parent_second_axis_offset,
                second_axis_range.max_lt / mult + end_offset,
            )
            setattr(resulting_block, second_axis.value, offset_block_range)
            resulting_blocks.append(resulting_block)
        return resulting_blocks

    def _generate_ceiling_blocks(self, story_num: int) -> List[LevelBlock]:
        # todo: smarter ceiling cuts using ramp angle and player height
        tiles = self.generate_tiles(story_num)
        tiles[(tiles != TileIdentity.ROOM.value) & (tiles != TileIdentity.HALLWAY.value)] = np.nan
        tiles[(tiles == TileIdentity.ROOM.value) | (tiles == TileIdentity.HALLWAY.value)] = 0
        blocks = generate_blocks(tiles)
        story = self.stories[story_num]
        level_blocks = []
        for block in blocks:
            level_block = LevelBlock(
                x=block.x,
                y=NewRange(story.outer_height - CEILING_THICKNESS, story.outer_height),
                z=block.z,
                kind=BlockKind.CEILING,
            )
            level_blocks.append(level_block)
        return level_blocks

    def _generate_window_blocks(self, story_num: int) -> List[LevelBlock]:
        level_blocks = []
        for window in self.stories[story_num].windows:
            level_blocks.append(
                make_block(
                    BlockKind.WINDOW, window.position, window.size, is_visual=False, is_collider=not window.is_passable
                )
            )
        return level_blocks

    def plot(self):
        vmin = min(tile_symbol.value for tile_symbol in TileIdentity)
        colors = {
            "gray": TileIdentity.FULL.value,
            "lightgreen": TileIdentity.ROOM.value,
            "blue": TileIdentity.RAMP.value,
            "darkblue": TileIdentity.RAMP_BOTTOM_LANDING.value,
            "lightblue": TileIdentity.RAMP_TOP_LANDING.value,
            "lightgray": TileIdentity.HALLWAY.value,
        }
        cmap = sorted(colors, key=colors.get)

        fig, axes = plt.subplots(len(self.stories), 1, figsize=(5, 5 * len(self.stories)))
        for i, story in enumerate(self.stories):
            tiles = self.generate_tiles(story.num)
            ax = axes[i] if len(self.stories) > 1 else axes

            # Rooms with their heightmap
            continuous_tiles = tiles.copy()
            continuous_tiles[tiles != TileIdentity.ROOM.value] = np.nan
            max_height = 0
            for room in story.rooms:
                continuous_tiles[
                    room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width
                ] = room.floor_heightmap
                max_height = max(max_height, room.floor_heightmap.max())
            continuous_tiles[continuous_tiles > 0] /= max_height
            sns.heatmap(
                continuous_tiles,
                square=True,
                linewidths=0.25,
                linecolor="white",
                cmap="YlGn_r",
                vmin=-1,
                vmax=1,
                ax=ax,
            )

            # unset tiles, hallways, ramps
            discrete_tiles = tiles.copy()
            discrete_tiles[tiles == TileIdentity.ROOM.value] = np.nan
            sns.heatmap(
                discrete_tiles,
                square=True,
                linewidths=0.25,
                linecolor="white",
                cmap=cmap,
                cbar=False,
                vmin=vmin,
                vmax=0,
                ax=ax,
            )

            for r, room in enumerate(story.rooms):
                ax.annotate(str(r), (room.center.x + 0.25, room.center.z + 0.75))
            ax.title.set_text(f"Floor {story.num}")

        ax = axes[0] if len(self.stories) > 1 else axes
        handles, labels = ax.get_legend_handles_labels()
        manual_patches = [
            mpatches.Patch(color=color, label=TileIdentity(tile_value).pretty_name)
            for color, tile_value in colors.items()
        ]
        handles.extend(manual_patches)
        plt.legend(handles=handles, bbox_to_anchor=(1, 0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
        plt.show()

    def rebuild_rotated(self, yaw_degrees: int):
        if yaw_degrees == 0:
            # todo: not a copy, should rebuild anyway for consistency
            return self

        if yaw_degrees == -180:
            yaw_degrees = 180
        assert yaw_degrees in {90, -90, 180}
        numpy_rotation = int(-yaw_degrees / 90)

        new_story_links = []
        for story_link in self.story_links:
            bottom_story = self.stories[story_link.bottom_story_id]
            new_bottom_position = rotate_position(
                story_link.bottom_position, bottom_story.width, bottom_story.length, yaw_degrees
            )
            top_story = self.stories[story_link.top_story_id]
            new_top_position = rotate_position(story_link.top_position, top_story.width, top_story.length, yaw_degrees)
            new_story_links.append(
                attr.evolve(story_link, bottom_position=new_bottom_position, top_position=new_top_position)
            )

        rotated_stories = []
        for story in self.stories:
            if yaw_degrees in {90, -90}:
                new_story_width = story.length
                new_story_length = story.width
            else:
                new_story_width = story.width
                new_story_length = story.length

            new_footprint = np.rot90(story.footprint, numpy_rotation)

            new_rooms = []
            for room in story.rooms:
                if yaw_degrees in {90, -90}:
                    new_room_width = room.length
                    new_room_length = room.width
                else:
                    new_room_width = room.width
                    new_room_length = room.length

                new_position = rotate_position(room.position, story.width, story.length, yaw_degrees)
                # Switch from rotated point to new top-left point
                if yaw_degrees == 90:
                    new_position.x -= room.length - 1
                elif yaw_degrees == -90:
                    new_position.z -= room.width - 1
                else:
                    new_position.x -= room.width - 1
                    new_position.z -= room.length - 1
                assert new_position.x >= 0
                assert new_position.z >= 0

                new_heightmap = np.rot90(room.floor_heightmap, numpy_rotation)
                new_rooms.append(
                    attr.evolve(
                        room,
                        position=new_position,
                        width=new_room_width,
                        length=new_room_length,
                        floor_heightmap=new_heightmap,
                    )
                )

            new_hallways = []
            for hallway in story.hallways:
                new_points = [
                    rotate_position(point, story.width, story.length, yaw_degrees) for point in hallway.points
                ]
                new_hallways.append(attr.evolve(hallway, points=new_points))

            new_story_links_for_story = [
                story_link
                for story_link in new_story_links
                if story.num in {story_link.bottom_story_id, story_link.top_story_id}
            ]

            new_windows = []
            for window in story.windows:
                position_2d = Position(window.position[0], window.position[2])
                new_position_2d = rotate_position(position_2d, story.width, story.length, yaw_degrees, tile_like=False)
                new_position = np.array([new_position_2d.x, window.position[1], new_position_2d.z])
                if yaw_degrees in {90, -90}:
                    new_size = np.array([window.size[2], window.size[1], window.size[0]])
                else:
                    new_size = window.size.copy()
                new_windows.append(attr.evolve(window, position=new_position, size=new_size))

            rotated_stories.append(
                attr.evolve(
                    story,
                    width=new_story_width,
                    length=new_story_length,
                    footprint=new_footprint,
                    rooms=new_rooms,
                    hallways=new_hallways,
                    story_links=new_story_links_for_story,
                    windows=new_windows,
                )
            )
        return attr.evolve(self, stories=rotated_stories, story_links=new_story_links)


class BuildingNavGraph(Graph):
    def __init__(self, building: Building):
        super(BuildingNavGraph, self).__init__()
        self._add_story_edges(building)
        self._add_story_link_edges(building)

    def _add_story_edges(self, building: Building):
        y_offset = 0
        for story in building.stories:
            y_offset += story.floor_negative_depth
            for room in story.rooms:
                room_y = room.floor_heightmap[room.length // 2, room.width // 2]
                self.add_node(
                    self.get_room_node(story, room), position=(room.center.x, y_offset + room_y, room.center.z)
                )
            for i, hallway in enumerate(story.hallways):
                if hallway.from_room_id < 0 or hallway.to_room_id < 0:
                    # Exterior links
                    continue
                from_room = story.rooms[hallway.from_room_id]
                to_room = story.rooms[hallway.to_room_id]
                from_room_node = self.get_room_node(story, from_room)
                to_room_node = self.get_room_node(story, to_room)
                distance_to_hallway = euclidean_distance(from_room.center, hallway.points[0])
                distance_from_hallway = euclidean_distance(hallway.points[-1], to_room.center)
                total_length = hallway.total_length + distance_to_hallway + distance_from_hallway
                self.add_edge(from_room_node, to_room_node, distance=total_length)
            y_offset += story.outer_height

    def _add_story_link_edges(self, building: Building):
        for link in building.story_links:
            assert isinstance(link, Ramp)
            from_story = building.stories[link.bottom_story_id]
            from_room = from_story.rooms[link.bottom_room_id]
            to_story = building.stories[link.top_story_id]
            to_room = to_story.rooms[link.top_room_id]
            from_room_node = self.get_room_node(from_story, from_room)
            to_room_node = self.get_room_node(to_story, to_room)
            distance_to_ramp = euclidean_distance(from_room.center, link.bottom_position)
            distance_from_ramp = euclidean_distance(link.top_position, to_room.center)
            ramp_run = link.slope_run
            ramp_rise = from_story.outer_height + to_story.floor_negative_depth
            ramp_length = math.sqrt(ramp_run ** 2 + ramp_rise ** 2)
            total_length = ramp_length + distance_to_ramp + distance_from_ramp
            self.add_edge(from_room_node, to_room_node, distance=total_length)

    def get_nearest_node(self, point: Vector3) -> str:
        positions = nx.get_node_attributes(self, "position")
        kd_tree = KDTree(list(positions.values()))
        distance, position_idx = kd_tree.query(np.array([point.x, point.y, point.z]))
        return list(positions.keys())[position_idx]

    def get_room_node(self, story: Story, room: Room) -> str:
        return f"{story.num}_R{room.id}"

    def plot(self, origin=Vector3(0, 0, 0), ax=None):
        plt.figure(figsize=(25, 25))
        positions = nx.get_node_attributes(self, "position")

        def translate_and_swap_axes(p):
            # This swap ensures the plot matches the building tile plot
            # to avoid axis swap, we need mpl 3.6.0+ to change up-axis: https://stackoverflow.com/a/56457693/5814943
            x, y, z = p
            x += origin.x
            y += origin.y
            z += origin.z
            return z, x, y

        node_xyz = np.array([translate_and_swap_axes(positions[v]) for v in self.nodes])
        edge_xyz = np.array(
            [(translate_and_swap_axes(positions[u]), translate_and_swap_axes(positions[v])) for u, v in self.edges()]
        )

        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure()
            fig.tight_layout()
            ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*node_xyz.T, s=100, ec="w")
        for i, node in enumerate(self.nodes):
            ax.text(*node_xyz[i], node, zdir=None, size=8)

        distance_by_edge = nx.get_edge_attributes(self, "distance")
        for edge, edge_coordinates in zip(self.edges, edge_xyz):
            middle = midpoint(Vector3(*edge_coordinates[0]), Vector3(*edge_coordinates[1]))
            ax.plot(*edge_coordinates.T, color="tab:gray")
            ax.text(middle.x, middle.y, middle.z, f"{distance_by_edge[edge]:.1f}", color="darkgray", size=6)

        if not custom_ax:
            zs = [z for x, y, z in node_xyz]
            ax.set_zlim(min(zs), max(zs) + 0.1)
            ax.set_xlabel("z")
            ax.set_ylabel("x")
            ax.set_zlabel("y")
        return ax


def generate_room(room_id: int, width: int, height: int, length: int) -> Room:
    assert room_id != 0
    return Room(id=room_id, position=Position(0, 0), width=width, length=length, outer_height=height)


def add_room_tiles_to_grid(room: Room, tiles: np.ndarray, grid: np.ndarray) -> None:
    local_grid = grid[room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width]
    local_grid[local_grid == 0] += tiles[local_grid == 0]


def set_room_tiles_in_grid(room: Room, tiles: np.ndarray, grid: np.ndarray) -> None:
    grid[room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width] = tiles


def set_link_in_grid(link: StoryLink, grid: np.ndarray, bottom: bool) -> None:
    if bottom:
        grid[link.bottom_position.z, link.bottom_position.x] = TileIdentity.RAMP_BOTTOM_LANDING.value
    else:
        grid[link.top_position.z, link.top_position.x] = TileIdentity.RAMP_TOP_LANDING.value
    start = min(link.bottom_position.z, link.top_position.z), min(link.bottom_position.x, link.top_position.x)
    end = max(link.bottom_position.z, link.top_position.z), max(link.bottom_position.x, link.top_position.x)
    if start[0] == end[0]:
        start = start[0], start[1] + 1
        end = end[0], end[1] - 1
    else:
        start = start[0] + 1, start[1]
        end = end[0] - 1, end[1]
    grid[start[0] : end[0] + 1, start[1] : end[1] + 1] = TileIdentity.RAMP.value


def make_block(
    kind: BlockKind,
    centroid: np.array,
    size: np.array,
    is_visual: bool = True,
    is_collider: bool = True,
) -> LevelBlock:
    return LevelBlock(
        x=FloatRange(centroid[0] - size[0] / 2, centroid[0] + size[0] / 2),
        y=FloatRange(centroid[1] - size[1] / 2, centroid[1] + size[1] / 2),
        z=FloatRange(centroid[2] - size[2] / 2, centroid[2] + size[2] / 2),
        is_visual=is_visual,
        is_collider=is_collider,
        kind=kind,
    )


def generate_blocks(tiles: np.ndarray, SOLID=0) -> List[Block]:
    """
    Divide a grid of `tiles` into as few `Block`s as possible.
    The algorithm iterates tile-by-tile to find a solid tile that is not already part of a block. It then starts a new
    block. The block formation goes greedily left-to-right to find a row of solid tiles and then extends the row
    downward as far as possible.
    """
    length, width = tiles.shape

    block = None
    blocks = []
    covered_tiles = set()
    for (z, x), tile in np.ndenumerate(tiles):
        if (z, x) in covered_tiles:
            continue

        zi, xi = z, x  # iterators over z (down) and x (right) axes
        while True:
            if tiles[zi][xi] == SOLID:
                if not block:
                    if (zi, xi) in covered_tiles:
                        break
                    # We've got an unused solid tile, start a working block
                    block = WorkingBlock(z_min=zi, x_min=xi)

                covered_tiles.add((zi, xi))

                if block.x_max is None:
                    # We're still determining row width
                    if xi + 1 < width and tiles[zi][xi + 1] == SOLID and (zi, xi + 1) not in covered_tiles:
                        # Tile to right is solid and unused
                        xi += 1
                        continue
                    else:
                        # We've reached the end of the row or a used/non-solid block, so row is complete
                        block.x_max = xi

                free_tile_below = zi + 1 < length
                if free_tile_below:
                    candidate_tiles = tiles[zi + 1 : zi + 2, block.x_min : block.x_max + 1]
                    if (candidate_tiles == SOLID).all():
                        candidate_coords = set(product(range(zi + 1, zi + 2), range(block.x_min, block.x_max + 1)))
                        no_dupe_tiles = len(candidate_coords.intersection(covered_tiles)) == 0
                        if no_dupe_tiles:
                            covered_tiles = covered_tiles.union(candidate_coords)
                            zi += 1
                            continue

                # We can't extend the block downward, so we finish here
                block.z_max = zi
                blocks.append(block.to_block())
                block = None
                break
            else:
                # We've struck a non-solid tile, so we finalize our block if we have one
                if block:
                    if block.x_max is None:
                        block.x_max = xi
                    if block.z_max is None:
                        block.z_max = zi
                    blocks.append(block)
                    block = None
                    break

            # Keep iterating row-wise, wrapping around to the next column
            if xi + 1 < width:
                xi += 1
            elif zi + 1 < length:
                xi = 0
                zi += 1
            else:
                break
    return blocks


def make_solid_block_building(region: Region, height: FloatRange = FloatRange(-15.0, 15.0)):
    dimensions = (int(region.z.size), int(region.x.size))
    room = Room(
        0,
        Position(1, 1),
        dimensions[1] - 2,
        dimensions[0] - 2,
        int(height.size - DEFAULT_FLOOR_THICKNESS - CEILING_THICKNESS),
    )
    stories = [
        Story(
            0, length=dimensions[0], width=dimensions[1], rooms=[room], hallways=[], story_links=[], has_ceiling=True
        )
    ]
    return Building(0, stories, story_links=[], position=Vector3(region.x.min_ge, height.min_ge, region.z.min_ge))


def get_evenly_spaced_centroids(
    wall_footprint: WallFootprint, thing_width: float, min_gap: float, centroid_y: float, max_things: int = np.inf
) -> np.ndarray:
    # like CSS flexbox space-evenly. Horizontal only for now. Depth-wise placed in middle of wall.
    thing_count, leftover_space = divmod(wall_footprint.wall_length, thing_width + min_gap)
    if thing_count > max_things:
        thing_count = max_things
        leftover_space = wall_footprint.wall_length - thing_count * (thing_width + min_gap)

    centroids = []
    wall_origin_x, wall_origin_z = wall_footprint.top_left
    for i in range(int(thing_count)):
        extra_gap_per_thing = leftover_space / thing_count
        x = (
            wall_origin_x + (i + 0.5) * (thing_width + min_gap + extra_gap_per_thing)
            if not wall_footprint.is_vertical
            else wall_origin_x + wall_footprint.wall_thickness / 2
        )
        z = (
            wall_origin_z + (i + 0.5) * (thing_width + min_gap + extra_gap_per_thing)
            if wall_footprint.is_vertical
            else wall_origin_z + wall_footprint.wall_thickness / 2
        )
        centroids.append((x, centroid_y, z))
    return np.array(centroids)


def homogeneous_transform_matrix(position: np.ndarray = np.array([0, 0, 0]), rotation: Optional[Rotation] = None):
    if rotation is None:
        rotation = np.eye(3)
    else:
        rotation = rotation.as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def make_color_visuals(mesh: Trimesh, rgba: Tuple[int, int, int, int]):
    return ColorVisuals(mesh, face_colors=np.repeat([rgba], len(mesh.faces), axis=0))


class MeshData:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.face_normals = []
        self.face_colors = []
        self.index_offset = 0


HIGH_POLY = "high_poly"
LOW_POLY = "low_poly"


def make_pretty_building_meshes(
    building: Building, blocks_by_story_by_kind: Dict[int, Dict[BlockKind, List[LevelBlock]]]
) -> Dict[str, Trimesh]:
    corners_by_story = defaultdict(list)
    wall_footprints_by_story = defaultdict(list)
    exterior_wall_blocks_by_story = defaultdict(list)
    original_outlines_by_story = {}
    for story in building.stories:
        outline = building.get_footprint_outline(story.num)
        # The outline we get is already rotated since it's returned in global space. However, we want it unrotated
        # and untranslated, since we translate and rotate the entire mesh later.
        building_centroid_2d = building.position.x, building.position.z
        outline = affinity.rotate(outline, building.yaw_degrees, origin=building_centroid_2d)
        outline = affinity.translate(outline, -building.position.x, -building.position.z, 0)
        original_outlines_by_story[story.num] = outline

        with story.mutable_clone() as adjusted_story:
            new_footprint = adjusted_story.footprint.copy()
            wall_footprints_by_story[story.num] = find_exterior_wall_footprints(new_footprint)
            corners = find_corners(new_footprint)
            corners_by_story[story.num] = corners
            for position, corner_type, is_outside in corners:
                if is_outside:
                    new_footprint[position[0], position[1]] = False
            adjusted_story.footprint = new_footprint
            building.stories[story.num] = adjusted_story
            exterior_wall_blocks_by_story[story.num].extend(building._generate_exterior_wall_blocks(story.num))

    mesh_datasets = defaultdict(lambda: MeshData())

    x_offset = -building.stories[0].width / 2
    z_offset = -building.stories[0].length / 2
    tile_size = 1

    def add_mesh_data(parent_name, child_mesh):
        nonlocal mesh_datasets
        parent_mesh_data = mesh_datasets[parent_name]
        parent_mesh_data.vertices.extend(child_mesh.vertices)
        parent_mesh_data.faces.extend(child_mesh.faces + parent_mesh_data.index_offset)
        parent_mesh_data.face_normals.extend(child_mesh.face_normals)
        if child_mesh.visual:
            parent_mesh_data.face_colors.extend(child_mesh.visual.face_colors)
        else:
            parent_mesh_data.face_colors.extend([0, 0, 0] * len(child_mesh.faces))
        parent_mesh_data.index_offset += len(child_mesh.vertices)

    for story_num, blocks_by_kind in blocks_by_story_by_kind.items():
        story = building.stories[story_num]
        y_offset = building.get_story_y_offset(story_num)
        for kind, blocks in blocks_by_kind.items():
            if kind == BlockKind.EXTERIOR_WALL:
                blocks = exterior_wall_blocks_by_story[story.num]

            for block in blocks:
                if not block.is_visual:
                    continue

                translated_block = building.translate_block_to_building_space(
                    block, story, building.aesthetics.block_downsize_epsilon
                )

                box = creation.box(translated_block.size)
                box = unnormalize(box)
                position = np.array([*translated_block.centroid])
                transform = homogeneous_transform_matrix(position)
                box.apply_transform(transform)
                colors_by_kind = {
                    BlockKind.EXTERIOR_WALL: building.aesthetics.exterior_color,
                    BlockKind.INTERIOR_WALL: building.aesthetics.interior_wall_color,
                    BlockKind.FLOOR: building.aesthetics.floor_color,
                    BlockKind.CEILING: building.aesthetics.ceiling_color,
                }
                box.visual = make_color_visuals(box, colors_by_kind.get(kind))
                add_mesh_data(HIGH_POLY, box)
                if kind in {BlockKind.EXTERIOR_WALL, BlockKind.CEILING}:
                    add_mesh_data(LOW_POLY, box)

        for story_link in story.story_links:
            if story_link.bottom_story_id != story.num:
                continue
            story_link_geometry = story_link.get_geometry(building)
            for box in story_link_geometry:
                assert isinstance(box, Box)
                trimesh_box = creation.box((box.width, box.height, box.length))
                position = np.array(
                    [
                        x_offset + box.x,
                        building.get_story_y_offset(story_link.bottom_story_id) + DEFAULT_FLOOR_THICKNESS + box.y,
                        z_offset + box.z,
                    ]
                )
                rotation = Rotation.from_euler("xyz", [box.pitch, box.yaw, box.roll], degrees=True)
                rotation_transform = homogeneous_transform_matrix(rotation=rotation)
                trimesh_box.apply_transform(rotation_transform)
                translation_transform = homogeneous_transform_matrix(position)
                trimesh_box.apply_transform(translation_transform)
                trimesh_box = unnormalize(trimesh_box)
                trimesh_box.visual = make_color_visuals(trimesh_box, building.aesthetics.exterior_color)
                add_mesh_data(HIGH_POLY, trimesh_box)

        outline = original_outlines_by_story[story_num]

        rail_height = building.aesthetics.rail_height
        rail_width = building.aesthetics.rail_thickness

        # constant: 1/2 of tile size
        buffer_width = 0.5

        min_x, min_y, max_x, max_y = outline.bounds
        width, length = max_x - min_x, max_y - min_y
        x_scaling_factor = width / (width + buffer_width * 2 + rail_width * building.aesthetics.rail_overhang_factor)
        z_scaling_factor = length / (length + buffer_width * 2 + rail_width * building.aesthetics.rail_overhang_factor)
        outline = outline.buffer(buffer_width, join_style=1, single_sided=True)
        outline = affinity.scale(outline, x_scaling_factor, z_scaling_factor)

        path = np.array([(c[0], y_offset + story.outer_height + rail_height, c[1]) for c in outline.exterior.coords])
        square = shapely_box(-rail_width / 2, 0, rail_width / 2, rail_height)
        border = creation.sweep_polygon(square, path)
        border = unnormalize(border)
        border.visual = make_color_visuals(border, building.aesthetics.trim_color)
        add_mesh_data(HIGH_POLY, border)

        offset = np.array([x_offset, 0, z_offset])

        crossbeam_size = building.aesthetics.crossbeam_size
        crossbeam_protrusion = building.aesthetics.crossbeam_protrusion
        crossbeam_centroid_y = y_offset + story.outer_height * building.aesthetics.crossbeam_y_proportion_of_height
        for wall_footprint in wall_footprints_by_story[story_num]:
            centroids = get_evenly_spaced_centroids(
                wall_footprint, crossbeam_size, building.aesthetics.crossbeam_min_gap, crossbeam_centroid_y
            )
            for centroid in centroids:
                centroid = local_to_global_coords(centroid, offset)
                transform = homogeneous_transform_matrix(position=centroid)
                if wall_footprint.is_vertical:
                    # Centroids are in wall middle, so adding +1 (wall thickness) to ensure they protrude
                    extents = (1 + crossbeam_protrusion, crossbeam_size, crossbeam_size)
                else:
                    extents = (crossbeam_size, crossbeam_size, 1 + crossbeam_protrusion)
                crossbeam = creation.box(extents, transform)
                crossbeam = unnormalize(crossbeam)
                crossbeam.visual = make_color_visuals(crossbeam, building.aesthetics.crossbeam_color)
                add_mesh_data(HIGH_POLY, crossbeam)

        for (z, x), corner_type, is_outside in corners_by_story[story_num]:
            if not is_outside:
                continue

            story = building.stories[story_num]

            corner_centroid = np.array(
                [x_offset + x + tile_size / 2, y_offset + (story.outer_height) / 2, z_offset + z + tile_size / 2]
            )
            rotation = Rotation.from_euler("x", 90, degrees=True)
            transform = homogeneous_transform_matrix(position=corner_centroid, rotation=rotation)
            round_corner_cylinder = creation.cylinder(tile_size / 2, story.outer_height, transform=transform)
            round_corner_cylinder = unnormalize(round_corner_cylinder)

            multiplier = 1 if corner_type in [CornerType.NE, CornerType.NW] else -1
            box_z = corner_centroid[2] + (multiplier * (tile_size / 4))
            box_centroid = np.array([corner_centroid[0], corner_centroid[1], box_z])
            transform = homogeneous_transform_matrix(position=box_centroid)
            horizontal_box = creation.box((tile_size, story.outer_height, tile_size / 2), transform)
            horizontal_box = unnormalize(horizontal_box)

            multiplier = 1 if corner_type in [CornerType.NW, CornerType.SW] else -1
            box_x = corner_centroid[0] + (multiplier * (tile_size / 4))
            box_centroid = np.array([box_x, corner_centroid[1], corner_centroid[2]])
            transform = homogeneous_transform_matrix(position=box_centroid)
            vertical_box = creation.box((tile_size / 2, story.outer_height, tile_size), transform)
            vertical_box = unnormalize(vertical_box)

            for component in [round_corner_cylinder, horizontal_box, vertical_box]:
                component.visual = make_color_visuals(component, building.aesthetics.exterior_color)
                add_mesh_data(HIGH_POLY, component)

    meshes = {}
    for name, mesh_data in mesh_datasets.items():
        mesh = Trimesh(
            mesh_data.vertices,
            mesh_data.faces,
            face_normals=mesh_data.face_normals,
            face_colors=mesh_data.face_colors,
            process=False,  # Don't merge identical vertices!
        )
        mesh.invert()  # Godot uses the opposite winding order of Trimesh
        meshes[name] = mesh
    return meshes
