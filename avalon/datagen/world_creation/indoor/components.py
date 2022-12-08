import copy
import math
from itertools import product
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import cast

import attr
import networkx as nx
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from networkx import Graph
from scipy.spatial.transform import Rotation

from avalon.common.errors import SwitchError
from avalon.contrib.serialization import Serializable
from avalon.datagen.godot_base_types import IntRange
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.indoor.blocks import FloorBlock
from avalon.datagen.world_creation.indoor.blocks import LadderBlock
from avalon.datagen.world_creation.indoor.blocks import LevelBlock
from avalon.datagen.world_creation.indoor.constants import CEILING_THICKNESS
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import LADDER_THICKNESS
from avalon.datagen.world_creation.indoor.constants import MIN_BUILDING_SIZE
from avalon.datagen.world_creation.indoor.constants import SLOPE_THICKNESS
from avalon.datagen.world_creation.indoor.constants import TILE_SIZE
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.constants import TileIdentity
from avalon.datagen.world_creation.indoor.tiles import draw_line_in_grid
from avalon.datagen.world_creation.types import BuildingBoolNP
from avalon.datagen.world_creation.types import BuildingFloatNP
from avalon.datagen.world_creation.types import BuildingIntNP
from avalon.datagen.world_creation.types import Point3DNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Room:
    id: int
    position: BuildingTile
    width: int
    length: int
    outer_height: float
    floor_heightmap: BuildingFloatNP = attr.field(
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
        default=attr.Factory(lambda self: np.ones((self.length, self.width), dtype=np.float_), takes_self=True),
    )
    climbable_mask: BuildingBoolNP = attr.field(
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
        default=attr.Factory(lambda self: np.zeros((self.length, self.width), dtype=np.bool_), takes_self=True),
    )

    def with_heightmap(self, new_heightmap: BuildingFloatNP) -> "Room":
        return attr.evolve(self, floor_heightmap=new_heightmap)

    @property
    def tiles(self) -> BuildingIntNP:
        return np.full((self.length, self.width), TileIdentity.ROOM.value, dtype=np.int_)

    def get_tile_positions(self, buffer: int = 0) -> Tuple[Tuple[int, int], ...]:
        """
        Returns (z, x) tuples of building-space coordinates that constitute this room, optionally including a buffer
        around the room to include its wall-space. Naive - doesn't account for values that would be OOB in the story.
        """
        return tuple(
            product(
                range(self.position.z - buffer, self.position.z + self.length + buffer),
                range(self.position.x - buffer, self.position.x + self.width + buffer),
            )
        )

    def with_position(self, new_position: BuildingTile) -> "Room":
        return attr.evolve(self, position=new_position)

    def with_id(self, new_id: int) -> "Room":
        return attr.evolve(self, id=new_id)

    @property
    def floor_space(self) -> int:
        return self.length * self.width

    @property
    def floor_negative_depth(self) -> float:
        return cast(float, abs(min(0, self.floor_heightmap.min())))

    @property
    def center(self) -> BuildingTile:
        return BuildingTile(x=self.position.x + (self.width // 2), z=self.position.z + (self.length // 2))

    @property
    def z_range(self) -> IntRange:
        return IntRange(self.position.z, self.position.z + self.length)

    @property
    def x_range(self) -> IntRange:
        return IntRange(self.position.x, self.position.x + self.width)


class RoomPlacementError(Exception):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Hallway(Serializable):
    """
    Hallway points  include joint points with rooms, so we can easily tell if the joint is horizontal / vertical.
    Points MUST be in the correct order (from_room, to_room) when creating the Hallway.
    """

    points: Tuple[BuildingTile, ...]
    from_room_id: int
    to_room_id: int
    width: int
    height: Optional[int] = None  # None => entire room height

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

    def _azimuth(self, point_a: BuildingTile, point_b: BuildingTile) -> Azimuth:
        if point_a.x == point_b.x:
            return Azimuth.SOUTH if point_a.z < point_b.z else Azimuth.NORTH
        elif point_a.z == point_b.z:
            return Azimuth.EAST if point_a.x < point_b.x else Azimuth.WEST
        else:
            raise SwitchError((point_a, point_b))


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StoryLink(Serializable):
    bottom_story_id: int
    bottom_room_id: int
    bottom_position: BuildingTile  # in story space
    top_story_id: int
    top_room_id: int
    top_position: BuildingTile  # in story space

    def get_link_length(self, bottom_story: "Story", top_story: "Story") -> float:
        raise NotImplementedError

    def get_level_blocks(self, bottom_story: "Story", top_story: "Story") -> Sequence[LevelBlock]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Ramp(StoryLink):
    width: int  # width will be expanded right+down from specified positions

    def get_link_length(self, bottom_story: "Story", top_story: "Story") -> float:
        run = self.slope_run
        rise = bottom_story.outer_height + top_story.floor_negative_depth
        return math.sqrt(run**2 + rise**2)

    @property
    def slope_run(self) -> float:
        # start/end positions are the landings - the actual slope floor length does not include these
        if self.bottom_position.x == self.top_position.x:
            return abs(self.bottom_position.z - self.top_position.z) - 1
        else:
            return abs(self.bottom_position.x - self.top_position.x) - 1

    def _get_geometry(self, bottom_story: "Story", top_story: "Story") -> Tuple[float, ...]:
        # TODO: 1d9f913a-3870-41b0-839e-8b29d0b89983
        bottom_room = bottom_story.rooms[self.bottom_room_id]
        bottom_position_in_room_space = BuildingTile(
            self.bottom_position.x - bottom_room.position.x, self.bottom_position.z - bottom_room.position.z
        )
        bottom_story_floor_positive_depth = (
            bottom_room.floor_heightmap[bottom_position_in_room_space.z, bottom_position_in_room_space.x]
            - DEFAULT_FLOOR_THICKNESS
        )
        bottom_story_floor_depth = bottom_story.floor_negative_depth + bottom_story_floor_positive_depth

        top_room = top_story.rooms[self.top_room_id]
        top_position_in_room_space = BuildingTile(
            self.top_position.x - top_room.position.x, self.top_position.z - top_room.position.z
        )
        top_story_floor_depth = (
            top_story.floor_negative_depth
            + top_room.floor_heightmap[top_position_in_room_space.z, top_position_in_room_space.x]
            - DEFAULT_FLOOR_THICKNESS
        )

        slope_width = float(self.width)
        slope_run = float(self.slope_run)
        slope_rise = (bottom_story.outer_height - bottom_story_floor_positive_depth) + top_story_floor_depth
        slope_length = math.sqrt(slope_run**2 + slope_rise**2)
        slope_angle = math.atan(slope_rise / slope_run)
        slope_thickness = SLOPE_THICKNESS

        complementary_angle = math.radians(90) - slope_angle
        # Our geometry is based on placing a plane, but we're placing a block, so we need to account for thickness
        thickness_y_offset = (slope_thickness / 2) * math.sin(complementary_angle)
        thickness_xz_offset = (slope_thickness / 2) * math.cos(complementary_angle)

        is_horizontal = self.bottom_position.z == self.top_position.z
        # +1s are to offset the slope from its bottom landing
        if is_horizontal:
            plane_width = slope_length
            plane_length = slope_width
            angle_multiplier = 1 if self.top_position.x > self.bottom_position.x else -1
            pitch, roll = 0.0, angle_multiplier * math.degrees(slope_angle)
            x = min(self.bottom_position.x, self.top_position.x) + (slope_run / 2) + 1 + thickness_xz_offset
            z = min(self.bottom_position.z, self.top_position.z) + (slope_width / 2)
        else:
            plane_width = slope_width
            plane_length = slope_length
            angle_multiplier = -1 if self.top_position.z > self.bottom_position.z else 1
            pitch, roll = angle_multiplier * math.degrees(slope_angle), 0.0
            x = min(self.bottom_position.x, self.top_position.x) + (slope_width / 2)
            z = min(self.bottom_position.z, self.top_position.z) + (slope_run / 2) + 1 + thickness_xz_offset

        y = math.tan(slope_angle) * (slope_run / 2) - thickness_y_offset + bottom_story_floor_depth
        return x, y, z, plane_width, slope_thickness, plane_length, pitch, 0, roll

    def get_level_blocks(self, bottom_story: "Story", top_story: "Story") -> List[LevelBlock]:
        x, y, z, plane_width, slope_thickness, plane_length, pitch, yaw, roll = self._get_geometry(
            bottom_story, top_story
        )
        return [
            FloorBlock.make(
                np.array([x, y, z]),
                np.array([plane_width, slope_thickness, plane_length]),
                rotation=Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True),
            )
        ]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Ladder(StoryLink):
    width: int
    azimuth: Azimuth

    def get_link_length(self, bottom_story: "Story", top_story: "Story") -> float:
        return bottom_story.outer_height + top_story.floor_negative_depth

    def get_geometry(self, bottom_story: "Story", top_story: "Story") -> Tuple[float, ...]:
        # todo: dedup
        bottom_room = bottom_story.rooms[self.bottom_room_id]

        bottom_position_in_room_space = BuildingTile(
            self.bottom_position.x - bottom_room.position.x, self.bottom_position.z - bottom_room.position.z
        )
        bottom_story_floor_positive_depth = (
            bottom_room.floor_heightmap[bottom_position_in_room_space.z, bottom_position_in_room_space.x]
            - DEFAULT_FLOOR_THICKNESS
        )
        bottom_story_floor_depth = bottom_story.floor_negative_depth + bottom_story_floor_positive_depth

        top_room = top_story.rooms[self.top_room_id]
        top_position_in_room_space = BuildingTile(
            self.top_position.x - top_room.position.x, self.top_position.z - top_room.position.z
        )
        top_story_floor_depth = (
            top_story.floor_negative_depth
            + top_room.floor_heightmap[top_position_in_room_space.z, top_position_in_room_space.x]
            - DEFAULT_FLOOR_THICKNESS
        )

        tile_size = float(TILE_SIZE)
        ladder_height = (bottom_story.outer_height - bottom_story_floor_positive_depth) + top_story_floor_depth
        ladder_thickness = LADDER_THICKNESS

        y = bottom_story_floor_depth + ladder_height / 2
        if self.azimuth == Azimuth.NORTH:
            x, z = self.top_position.x + tile_size / 2, self.top_position.z + ladder_thickness / 2
            ladder_width, ladder_length = tile_size, ladder_thickness
        elif self.azimuth == Azimuth.EAST:
            x, z = self.top_position.x + tile_size - ladder_thickness / 2, self.top_position.z + tile_size / 2
            ladder_width, ladder_length = ladder_thickness, tile_size
        elif self.azimuth == Azimuth.SOUTH:
            x, z = self.top_position.x + tile_size / 2, self.top_position.z + tile_size - ladder_thickness / 2
            ladder_width, ladder_length = tile_size, ladder_thickness
        elif self.azimuth == Azimuth.WEST:
            x, z = self.top_position.x + ladder_thickness / 2, self.top_position.z + tile_size / 2
            ladder_width, ladder_length = ladder_thickness, tile_size
        else:
            raise SwitchError(f"Unexpected Azimuth {self.azimuth}")
        return x, y, z, ladder_width, ladder_height, ladder_length

    def get_level_blocks(self, bottom_story: "Story", top_story: "Story") -> Sequence[LevelBlock]:
        x, y, z, ladder_width, ladder_height, ladder_length = self.get_geometry(bottom_story, top_story)

        collision_block = LadderBlock.make(
            np.array([x, y, z]),
            np.array([ladder_width, ladder_height, ladder_length]),
            rotation=Rotation.identity(),
            is_visual=False,
            is_collider=True,
            is_climbable=True,
        )

        tile_size = 1
        rail_thickness = LADDER_THICKNESS
        if self.azimuth in {Azimuth.NORTH, Azimuth.SOUTH}:
            left_rail_x, left_rail_z = x - tile_size / 2 + rail_thickness / 2, z
        else:
            left_rail_x, left_rail_z = x, z - tile_size / 2 + rail_thickness / 2
        left_rail = LadderBlock.make(
            np.array([left_rail_x, y, left_rail_z]),
            np.array([rail_thickness, ladder_height, rail_thickness]),
            rotation=Rotation.identity(),
            is_visual=True,
            is_collider=False,
        )

        if self.azimuth in {Azimuth.NORTH, Azimuth.SOUTH}:
            right_rail_x, right_rail_z = x + tile_size / 2 - rail_thickness / 2, z
        else:
            right_rail_x, right_rail_z = x, z + tile_size / 2 - rail_thickness / 2
        right_rail = LadderBlock.make(
            np.array([right_rail_x, y, right_rail_z]),
            np.array([rail_thickness, ladder_height, rail_thickness]),
            rotation=Rotation.identity(),
            is_visual=True,
            is_collider=False,
        )

        step_size = 0.4
        step_thickness = LADDER_THICKNESS
        step_count, margin = divmod(ladder_height, step_size)
        if ladder_width > ladder_length:
            step_width = ladder_width - 2 * step_thickness
            step_length = ladder_length
        else:
            step_width = ladder_width
            step_length = ladder_length - 2 * step_thickness
        steps = []
        for i in range(int(step_count)):
            steps.append(
                LadderBlock.make(
                    np.array([x, margin + i * step_size, z]),
                    np.array([step_width, step_thickness, step_length]),
                    rotation=Rotation.identity(),
                    is_visual=True,
                    is_collider=False,
                )
            )

        visual_blocks = [left_rail, right_rail, *steps]

        return [collision_block] + visual_blocks


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Window(Serializable):
    position: Point3DNP  # centroid
    size: Point3DNP
    is_passable: bool = False  # should the cutout have a collision shape


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Entrance(Serializable):
    story_num: int
    connected_room_id: int
    azimuth: Azimuth
    points: Tuple[BuildingTile, ...]
    width: int = 1
    height: int = 3

    exterior_id_by_azimuth = {
        Azimuth.NORTH: -1,
        Azimuth.SOUTH: -2,
        Azimuth.WEST: -3,
        Azimuth.EAST: -4,
    }

    @property
    def hallway(self) -> Hallway:
        return Hallway(
            self.points, Entrance.exterior_id_by_azimuth[self.azimuth], self.connected_room_id, self.width, self.height
        )

    def get_points_in_story_outline(self, story: "Story") -> List[Tuple[int, int]]:
        outline_mask = story.get_outline_mask()
        entrance_mask = np.zeros_like(outline_mask)
        draw_line_in_grid(self.points, entrance_mask, TileIdentity.HALLWAY.value, include_ends=False)
        entrance_mask_points_in_outline = np.where(outline_mask & entrance_mask)
        return list(zip(*entrance_mask_points_in_outline))  # type: ignore

    def get_connected_room_and_landing_position(self, story: "Story") -> Tuple[Room, BuildingTile]:
        """note: landing position is returned in room space"""
        room = story.rooms[self.connected_room_id]
        entrance_grid = np.zeros((story.length, story.width))
        draw_line_in_grid(self.points, entrance_grid, TileIdentity.HALLWAY.value, include_ends=True)
        room_grid = np.zeros_like(entrance_grid)
        set_room_tiles_in_grid(room, room.tiles, room_grid)
        entrance_hallway_overlaps_with_room = entrance_grid.astype(np.bool_) & room_grid.astype(np.bool_)
        room_landing_coords = list(zip(*np.where(entrance_hallway_overlaps_with_room)))
        room_landing = list(room_landing_coords[0])
        room_landing[0] -= room.position.z
        room_landing[1] -= room.position.x
        return room, BuildingTile(x=room_landing[1], z=room_landing[0])


@attr.s(auto_attribs=True, eq=False, collect_by_mro=True)
class Story:
    r"""
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
    width: int = attr.field(validator=attr.validators.ge(MIN_BUILDING_SIZE))
    length: int = attr.field(validator=attr.validators.ge(MIN_BUILDING_SIZE))
    footprint: BuildingBoolNP = attr.field(
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
        default=attr.Factory(lambda self: np.ones((self.length, self.width), dtype=np.bool_), takes_self=True),
    )
    rooms: List[Room] = attr.Factory(list)
    hallways: List[Hallway] = attr.Factory(list)
    story_links: List[StoryLink] = attr.Factory(list)
    windows: List[Window] = attr.Factory(list)
    entrances: List[Entrance] = attr.Factory(list)
    has_ceiling: bool = False

    @property
    def outer_height(self) -> float:
        return max([r.outer_height for r in self.rooms])

    @property
    def inner_height(self) -> float:
        # The EXACT inner height can vary tile by tile depending on terrain; this is the "base" inner height assuming
        # the floor uses DEFAULT_FLOOR_THICKNESS.
        return self.outer_height - DEFAULT_FLOOR_THICKNESS - (CEILING_THICKNESS if self.has_ceiling else 0)

    @property
    def floor_negative_depth(self) -> float:
        # Note that all positive story terrain is projected INTO the room (rather than offsetting the room & walls),
        # see diagram above
        return max([room.floor_negative_depth for room in self.rooms])

    @property
    def floor_heightmap(self) -> BuildingFloatNP:
        """returns positive floor thickness for all walkable tiles, np.nan elsewhere"""
        tiles = np.full((self.length, self.width), np.nan, dtype=np.float32)
        for room in self.rooms:
            set_room_tiles_in_grid(room, room.floor_heightmap, tiles)
        for hallway in self.hallways:
            draw_line_in_grid(
                hallway.points, tiles, DEFAULT_FLOOR_THICKNESS, include_ends=False, drawable_grid_value=None
            )
        return tiles

    def get_room_at_point(self, point: BuildingTile) -> Optional[Room]:
        for room in self.rooms:
            if (
                room.position.x <= point.x <= room.position.x + room.width - 1
                and room.position.z <= point.z <= room.position.z + room.length - 1
            ):
                return room
        return None

    def get_outline_mask(self) -> BuildingBoolNP:
        is_building = self.footprint.astype(np.int8) != 0
        convolution = scipy.ndimage.convolve(self.footprint.astype(np.int8), np.ones((3, 3)), mode="constant")
        return cast(BuildingBoolNP, is_building & (convolution != 9))

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

    def generate_tiles(
        self,
        include_links_going_up: bool = True,
        include_links_going_down: bool = True,
        include_hallway_landings: bool = False,
    ) -> BuildingIntNP:
        tiles = np.invert(self.footprint).astype(np.float32) * TileIdentity.VOID.value
        for room in self.rooms:
            set_room_tiles_in_grid(room, room.tiles, tiles)
        for hallway in self.hallways:
            draw_line_in_grid(hallway.points, tiles, TileIdentity.HALLWAY.value, include_ends=include_hallway_landings)
        for link in self.story_links:
            if link.bottom_story_id == self.num and include_links_going_up:
                set_link_in_grid(link, tiles, set_bottom_landing=True, set_top_landing=False)
            elif link.top_story_id == self.num and include_links_going_down:
                set_link_in_grid(link, tiles, set_bottom_landing=False, set_top_landing=True)
        return tiles

    def generate_room_id_tiles(self) -> BuildingIntNP:
        tiles = np.invert(self.footprint).astype(np.int8) * TileIdentity.VOID.value
        for room in self.rooms:
            set_room_tiles_in_grid(room, np.full((room.length, room.width), room.id), tiles)
        return tiles


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Obstacle:
    story_id: int
    room_id: int

    def apply(self, stories: List[Story]) -> None:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Wall(Obstacle):
    points: Tuple[BuildingTile, ...]
    thickness: int
    height: float

    def apply(self, stories: List[Story]) -> None:
        height = self.height
        room = stories[self.story_id].rooms[self.room_id]
        new_heightmap = room.floor_heightmap.copy()
        draw_line_in_grid(self.points, new_heightmap, height, drawable_grid_value=None)
        deformed_room = room.with_heightmap(new_heightmap)
        stories[self.story_id].rooms[self.room_id] = deformed_room


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloorChasm(Obstacle):
    site_mask: BuildingBoolNP
    climbable_mask: BuildingBoolNP
    depth_map: BuildingFloatNP

    @property
    def chasm_mask(self) -> BuildingBoolNP:
        return self.site_mask & ~np.isnan(self.depth_map)

    def apply(self, stories: List[Story]) -> None:
        room = stories[self.story_id].rooms[self.room_id]
        new_heightmap = room.floor_heightmap.copy()
        new_heightmap[self.chasm_mask] = -self.depth_map[self.chasm_mask]
        new_climbable_mask = room.climbable_mask.copy()
        new_climbable_mask[self.site_mask] = self.climbable_mask[self.site_mask]
        stories[self.story_id].rooms[self.room_id] = attr.evolve(
            room, floor_heightmap=new_heightmap, climbable_mask=new_climbable_mask
        )


class TileGraph(Graph):
    def plot(self) -> None:
        fig, ax = plt.subplots(1, figsize=(10, 10))
        plt.gca().set_aspect("equal")
        plt.gca().invert_yaxis()
        nx.draw_networkx(self, pos={node: tuple(reversed(node)) for node in self.nodes}, ax=ax)


class RoomTileGraph(TileGraph):
    def __init__(self, rooms: Iterable[Room]) -> None:
        super().__init__()
        length = max([r.position.z + r.length for r in rooms])
        width = max([r.position.x + r.width for r in rooms])
        nx.grid_2d_graph(length, width, create_using=self)

    def find_shortest_path(
        self, point_a: BuildingTile, point_b: BuildingTile, avoid_rooms: Tuple[Room, ...] = tuple()
    ) -> List[Tuple[int, int]]:
        graph_copy = copy.deepcopy(self)
        avoid_points: List[Tuple[int, int]] = sum(
            [list(room.get_tile_positions(buffer=1)) for room in avoid_rooms], []
        )
        graph_copy.remove_nodes_from(avoid_points)
        return nx.astar_path(graph_copy, (point_a.z, point_a.x), (point_b.z, point_b.x))  # type: ignore


class StoryNavGraph(TileGraph):
    def __init__(self, story: Story, exclude_exterior: bool = True) -> None:
        super().__init__()
        nx.grid_2d_graph(story.length, story.width, create_using=self)
        tiles = story.generate_tiles()
        # Note: we do not include void below, since you could have unconnected rooms that each have their own entrance
        unwalkable_tile_identities = {TileIdentity.LINK, TileIdentity.FULL}
        if exclude_exterior:
            unwalkable_tile_identities.add(TileIdentity.VOID)
        unwalkable_tiles = list(
            zip(*np.where(np.isin(tiles, [identity.value for identity in unwalkable_tile_identities])))
        )
        self.remove_nodes_from(unwalkable_tiles)


def generate_room(room_id: int, width: int, height: int, length: int) -> Room:
    assert room_id != 0
    return Room(id=room_id, position=BuildingTile(0, 0), width=width, length=length, outer_height=height)


def add_room_tiles_to_grid(room: Room, tiles: BuildingIntNP, grid: BuildingIntNP) -> None:
    local_grid = grid[room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width]
    local_grid[local_grid == 0] += tiles[local_grid == 0]


def set_room_tiles_in_grid(room: Room, tiles: BuildingIntNP, grid: BuildingIntNP) -> None:
    grid[room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width] = tiles


def set_link_in_grid(link: StoryLink, grid: BuildingIntNP, set_bottom_landing: bool, set_top_landing: bool) -> None:
    if set_bottom_landing:
        grid[link.bottom_position.z, link.bottom_position.x] = TileIdentity.LINK_BOTTOM_LANDING.value
    if set_top_landing:
        grid[link.top_position.z, link.top_position.x] = TileIdentity.LINK_TOP_LANDING.value
    start = min(link.bottom_position.z, link.top_position.z), min(link.bottom_position.x, link.top_position.x)
    end = max(link.bottom_position.z, link.top_position.z), max(link.bottom_position.x, link.top_position.x)
    if start[0] == end[0]:
        start = start[0], start[1] + 1
        end = end[0], end[1] - 1
    else:
        start = start[0] + 1, start[1]
        end = end[0] - 1, end[1]
    grid[start[0] : end[0] + 1, start[1] : end[1] + 1] = TileIdentity.LINK.value
