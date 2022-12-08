import math
import sys
from itertools import combinations
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple

import attr
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from networkx import Graph
from networkx.algorithms.approximation import traveling_salesman_problem
from scipy import stats
from scipy.ndimage import convolve
from scipy.spatial import Delaunay
from scipy.spatial import QhullError

from avalon.common.errors import SwitchError
from avalon.common.utils import only
from avalon.contrib.serialization import Serializable
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.godot_base_types import IntRange
from avalon.datagen.godot_base_types import Vector3
from avalon.datagen.world_creation.debug_plots import IS_DEBUG_VIS
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import euclidean_distance
from avalon.datagen.world_creation.geometry import get_triangulation_edges
from avalon.datagen.world_creation.indoor.blocks import make_blocks_from_tiles
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.components import Entrance
from avalon.datagen.world_creation.indoor.components import Hallway
from avalon.datagen.world_creation.indoor.components import Ladder
from avalon.datagen.world_creation.indoor.components import Ramp
from avalon.datagen.world_creation.indoor.components import Room
from avalon.datagen.world_creation.indoor.components import RoomPlacementError
from avalon.datagen.world_creation.indoor.components import RoomTileGraph
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.indoor.components import StoryLink
from avalon.datagen.world_creation.indoor.components import Window
from avalon.datagen.world_creation.indoor.components import add_room_tiles_to_grid
from avalon.datagen.world_creation.indoor.components import generate_room
from avalon.datagen.world_creation.indoor.components import set_link_in_grid
from avalon.datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from avalon.datagen.world_creation.indoor.constants import MIN_BUILDING_SIZE
from avalon.datagen.world_creation.indoor.constants import MIN_ROOM_SIZE
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.constants import Orientation
from avalon.datagen.world_creation.indoor.constants import TileIdentity
from avalon.datagen.world_creation.indoor.constants import WallType
from avalon.datagen.world_creation.indoor.tiles import draw_line_in_grid
from avalon.datagen.world_creation.indoor.tiles import find_exterior_wall_footprints
from avalon.datagen.world_creation.indoor.tiles import get_neighbor_tiles
from avalon.datagen.world_creation.indoor.tiles import visualize_tiles
from avalon.datagen.world_creation.indoor.utils import get_evenly_spaced_centroids
from avalon.datagen.world_creation.indoor.utils import inset_borders
from avalon.datagen.world_creation.indoor.utils import rand_integer
from avalon.datagen.world_creation.indoor.wall_footprint import WallFootprint
from avalon.datagen.world_creation.types import BuildingBoolNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FootprintBuilder(Serializable):
    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[BuildingBoolNP],
    ) -> BuildingBoolNP:
        """returns a `np.array` of shape (building_length, building_width) where True represents buildable tiles"""
        raise NotImplementedError

    @staticmethod
    def inset_footprint(
        footprint: BuildingBoolNP, inset: int = 1, min_leftover_size: int = MIN_BUILDING_SIZE
    ) -> BuildingBoolNP:
        reduced_footprint = footprint.copy()
        for i in range(inset):
            reduced_footprint = inset_borders(reduced_footprint, min_leftover_size)
        return reduced_footprint


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RectangleFootprintBuilder(FootprintBuilder):
    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[BuildingBoolNP],
    ) -> BuildingBoolNP:
        if story_num != 0:
            assert footprint_below is not None, "Must pass footprint below for non-ground stories"
            return FootprintBuilder.inset_footprint(footprint_below)
        return np.ones((building_length, building_width), dtype=np.bool_)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class TLShapeFootprintBuilder(FootprintBuilder):
    allow_t_shape: bool = True

    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[BuildingBoolNP],
    ) -> BuildingBoolNP:
        if building_width == building_length == MIN_BUILDING_SIZE:
            raise ImpossibleWorldError("Building too small to form a T/L shape footprint")

        if story_num == 1:
            assert footprint_below is not None, "Must pass footprint below if story_num > 0"
            # Optionally inverting block generation direction gives us more variety
            inverted = rand.choice([True, False])
            footprint_below = footprint_below.copy().T if inverted else footprint_below.copy()
            blocks = make_blocks_from_tiles(footprint_below, solid_tile_value=True)
            min_block_size = MIN_BUILDING_SIZE  # avoid tiny sections that can't have rooms built into them
            block = rand.choice([b for b in blocks if b.x.size >= min_block_size and b.z.size >= min_block_size])  # type: ignore[arg-type]
            footprint = np.zeros((building_length, building_width), dtype=np.bool_)
            if inverted:
                tmp = block.x
                block.x = block.z
                block.z = tmp
            footprint[block.z.min_ge : block.z.max_lt, block.x.min_ge : block.x.max_lt] = 1
            return footprint
        elif story_num > 1:
            assert footprint_below is not None, "Must pass footprint below if story_num > 0"
            return FootprintBuilder.inset_footprint(footprint_below)

        rectangles = []
        footprint = np.zeros((building_length, building_width), dtype=np.bool_)

        min_width = max(round(building_width * 0.4), MIN_BUILDING_SIZE)
        max_width = max(round(building_width * 0.8), MIN_BUILDING_SIZE)

        rectangle_width = rand_integer(rand, min_width, max_width)
        if building_length > building_width and self.allow_t_shape:
            x_offset = rand.choice([0, building_width - rectangle_width])
        else:
            x_offset = rand_integer(rand, 0, building_width - rectangle_width)
        rectangles.append((x_offset, 0, rectangle_width, building_length))

        min_length = max(round(building_length * 0.4), MIN_BUILDING_SIZE)
        max_length = max(round(building_length * 0.8), MIN_BUILDING_SIZE)
        rectangle_length = rand_integer(rand, min_length, max_length)
        if building_width > building_length and self.allow_t_shape:
            z_offset = rand.choice([0, building_length - rectangle_length])
        else:
            z_offset = rand_integer(rand, 0, building_length - rectangle_length)
        rectangles.append((0, z_offset, building_width, rectangle_length))

        for x_offset, z_offset, building_width, building_length in rectangles:
            footprint[z_offset : z_offset + building_length, x_offset : x_offset + building_width] = 1
        return footprint


def _smooth(array: np.ndarray) -> None:
    raveled = array.ravel()
    cuts = np.where(np.diff(raveled) != 0)[0] + 1
    chunks = np.split(raveled, cuts)
    for chunk in chunks:
        if (chunk == 1).all() and chunk.size <= 2:
            chunk[True] = 0


def smooth_footprint(footprint: np.ndarray) -> None:
    """
    Removes sections of the footprint that are effectively zero-width (left wall
    is adjacent to right wall / top wall is adjacent to bottom wall).
    """
    _smooth(footprint)
    transposed_footprint = footprint.T.copy()
    _smooth(transposed_footprint)
    footprint[transposed_footprint.T == 0] = 0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class IrregularFootprintBuilder(FootprintBuilder):
    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[BuildingBoolNP],
    ) -> BuildingBoolNP:
        if building_width == building_length == MIN_BUILDING_SIZE:
            raise ImpossibleWorldError("Building too small to form an irregular shape footprint")

        if story_num > 0:
            assert footprint_below is not None, "Must pass footprint below if story_num > 0"
            footprint = FootprintBuilder.inset_footprint(footprint_below)
            smooth_footprint(footprint)
            return footprint

        footprint = np.zeros((building_length, building_width), dtype=np.bool_)
        rectangles: List[Tuple[int, int, int, int]] = []
        for i in range(5):
            min_width = max(round(building_width * 0.5), MIN_BUILDING_SIZE)
            max_width = max(round(building_width * 0.9), MIN_BUILDING_SIZE)
            min_length = max(round(building_length * 0.5), MIN_BUILDING_SIZE)
            max_length = max(round(building_length * 0.9), MIN_BUILDING_SIZE)

            rectangle_width = rand_integer(rand, min_width, max_width)
            rectangle_length = rand_integer(rand, min_length, max_length)

            if i == 0:
                x_offset = rand_integer(rand, 0, building_width - rectangle_width)
                z_offset = rand_integer(rand, 0, building_length - rectangle_length)
            else:
                another_rectangle = rand.choice(rectangles)
                possible_x_offsets = [another_rectangle[0]]
                another_rectangle_width = another_rectangle[2]
                if rectangle_width < another_rectangle_width:
                    possible_x_offsets.append(another_rectangle_width - rectangle_width)

                possible_z_offsets = [another_rectangle[1]]
                another_rectangle_length = another_rectangle[3]
                if rectangle_length < another_rectangle_length:
                    possible_z_offsets.append(another_rectangle_length - rectangle_length)

                aligned_offset = rand.choice(["x", "z"])
                if aligned_offset == "x":
                    x_offset = rand.choice(possible_x_offsets)
                    z_offset = rand_integer(rand, 0, building_length - rectangle_length)
                else:
                    x_offset = rand_integer(rand, 0, building_width - rectangle_width)
                    z_offset = rand.choice(possible_z_offsets)
            rectangles.append((x_offset, z_offset, rectangle_width, rectangle_length))

        for x_offset, z_offset, building_width, building_length in rectangles:
            footprint[z_offset : z_offset + building_length, x_offset : x_offset + building_width] = 1
        return footprint


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RoomBuilder(Serializable):
    def build(self, story_footprint: np.ndarray, rand: np.random.Generator) -> List[Room]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultRoomBuilder(RoomBuilder):
    room_count: int
    room_options: List[Vector3]

    def build(self, story_footprint: np.ndarray, rand: np.random.Generator) -> List[Room]:
        world_length, world_width = story_footprint.shape

        rooms = []
        room_floor_space = 0
        for i in range(self.room_count):
            width, height, length = rand.choice(self.room_options)  # type: ignore[arg-type]
            room = generate_room(i + 1, width, height, length)
            assert room.width < world_width and room.length < world_length, "Room too large to fit world"
            rooms.append(room)
            room_floor_space += width * length
        assert room_floor_space < world_width * world_length, "Room floor space exceeds world size"
        return self._place_rooms(story_footprint, rooms)

    def _place_rooms(self, story_footprint: BuildingBoolNP, rooms: List[Room]) -> List[Room]:
        """
        At its core, this is a rectangle packing problem, which is NP-hard. There are many published algorithms out there
        that we could use if we want to, but the one below is a quick and dirty homemade one while we iterate this.
        """
        grid = np.invert(story_footprint).astype(np.int64)
        placed_rooms = []
        sorted_rooms = sorted(rooms, key=lambda r: r.floor_space, reverse=True)
        blocks = make_blocks_from_tiles(grid)
        margin = 1
        for i, room in enumerate(sorted_rooms):
            for block in blocks:
                block_length = block.z.max_lt - block.z.min_ge
                block_width = block.x.max_lt - block.x.min_ge
                if block_width < room.width + (margin * 2) or block_length < room.length + (margin * 2):
                    continue
                center = block.x.min_ge + block_width // 2, block.z.min_ge + block_length // 2
                room = room.with_position(
                    BuildingTile(x=center[0] - room.width // 2, z=center[1] - room.length // 2)
                ).with_id(i)
                add_room_tiles_to_grid(room, room.tiles, grid)
                placed_rooms.append(room)
                blocks = make_blocks_from_tiles(grid)
                break
            else:
                # The options here are: a) try a more compact placement algorithm; b) skip room; for now we just raise
                visualize_tiles(grid)
                raise RoomPlacementError(
                    f"can't fit room of size {room.width, room.length} with {margin=} in any block"
                )
        return placed_rooms


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HouseLikeRoomBuilder(RoomBuilder):
    min_room_size: int = 3  # Note: this param will not be obeyed if the building is too small to fit
    max_rooms: int = sys.maxsize
    partition_thickness: int = 1
    room_height: float = DEFAULT_STORY_HEIGHT

    def build(self, story_footprint: np.ndarray, rand: np.random.Generator) -> List[Room]:
        world_length, world_width = story_footprint.shape
        random_state = np.random.RandomState(seed=rand.integers(0, 2**32 - 1))

        # Separate footprint into rectangles as "initial rooms"
        padded_footprint = story_footprint.copy()
        for i in range(self.partition_thickness):
            # We want rooms to be at least of width 2 to be able to place story links that we can walk around
            padded_footprint = inset_borders(padded_footprint, min_leftover_size=MIN_ROOM_SIZE)
        footprint_blocks = make_blocks_from_tiles(padded_footprint, solid_tile_value=True)
        initial_rooms = [
            Room(-1, BuildingTile(block.x.min_ge, block.z.min_ge), block.x.size, block.z.size, self.room_height)
            for block in footprint_blocks
        ]

        if len(initial_rooms) == 1 and initial_rooms[0].width == 1 or initial_rooms[0].length == 1:
            # Don't make rooms that can't have walls built between them anyway
            return [room.with_id(i) for i, room in enumerate(initial_rooms)]

        initial_orientation = rand.choice(list(Orientation))  # type: ignore[arg-type]

        # Then partition each of the rooms
        rooms = []
        if len(initial_rooms) > self.max_rooms:
            raise ImpossibleWorldError(
                f"max_rooms is {self.max_rooms}, but initial footprint split yields {len(initial_rooms)}"
            )
        elif len(initial_rooms) == self.max_rooms:
            rooms = initial_rooms
        else:
            leftover_credits = self.max_rooms - len(initial_rooms)  # how many more rooms can you create
            for room in initial_rooms:
                new_rooms, leftover_credits = self._partition(
                    room,
                    initial_orientation,
                    world_width,
                    world_length,
                    self.partition_thickness,
                    leftover_credits,
                    random_state,
                )
                rooms.extend(new_rooms)
        return [room.with_id(i) for i, room in enumerate(rooms)]

    def _partition(
        self,
        room: Room,
        orientation: Orientation,
        world_width: int,
        world_length: int,
        partition_thickness: int,
        leftover_room_credits: int,
        random_state: np.random.RandomState,
    ) -> Tuple[List[Room], int]:
        if leftover_room_credits == 0:
            return [room], leftover_room_credits

        if orientation == Orientation.HORIZONTAL:
            edge_size = room.width
        else:
            edge_size = room.length
        if edge_size < 2 * self.min_room_size + partition_thickness:
            return [room], leftover_room_credits
        partition_location_distribution = stats.uniform(
            self.min_room_size, edge_size - 2 * self.min_room_size - partition_thickness
        )
        partition_location = round(partition_location_distribution.rvs(random_state=random_state))
        second_room_offset = partition_location + partition_thickness
        if orientation == Orientation.HORIZONTAL:
            partition_a = Room(
                -1,
                BuildingTile(room.position.x, room.position.z),
                width=partition_location,
                length=room.length,
                outer_height=room.outer_height,
            )
            partition_b = Room(
                -1,
                BuildingTile(room.position.x + second_room_offset, room.position.z),
                width=room.width - second_room_offset,
                length=room.length,
                outer_height=room.outer_height,
            )

        else:
            partition_a = Room(
                -1,
                BuildingTile(room.position.x, room.position.z),
                width=room.width,
                length=partition_location,
                outer_height=room.outer_height,
            )
            partition_b = Room(
                -1,
                BuildingTile(room.position.x, room.position.z + second_room_offset),
                width=room.width,
                length=room.length - second_room_offset,
                outer_height=room.outer_height,
            )
        next_partition_dimension_size = world_width if orientation == Orientation.HORIZONTAL else world_length
        probability_keep_partitioning = edge_size / next_partition_dimension_size
        keep_partitioning_probabilities = [probability_keep_partitioning, 1 - probability_keep_partitioning]
        keep_partitioning_a = random_state.choice([True, False], p=keep_partitioning_probabilities)
        keep_partitioning_b = random_state.choice([True, False], p=keep_partitioning_probabilities)
        orientation = orientation.other()
        leftover_room_credits -= 1
        partitioned_a, leftover_room_credits = (
            self._partition(
                partition_a,
                orientation,
                world_width,
                world_length,
                partition_thickness,
                leftover_room_credits,
                random_state,
            )
            if keep_partitioning_a
            else ([partition_a], leftover_room_credits)
        )
        partitioned_b, leftover_room_credits = (
            self._partition(
                partition_b,
                orientation,
                world_width,
                world_length,
                partition_thickness,
                leftover_room_credits,
                random_state,
            )
            if keep_partitioning_b
            else ([partition_b], leftover_room_credits)
        )
        return [*partitioned_a, *partitioned_b], leftover_room_credits


def get_room_overlap(room_a: Room, room_b: Room) -> Tuple[Optional[IntRange], Optional[IntRange]]:
    return room_a.x_range.overlap(room_b.x_range), room_a.z_range.overlap(room_b.z_range)


def are_rooms_neighboring(room_a: Room, room_b: Room, max_gap: int = 1) -> bool:
    x_overlap, z_overlap = get_room_overlap(room_a, room_b)
    if x_overlap is not None:
        topmost_bottom = min(room_a.position.z + room_a.length, room_b.position.z + room_b.length) - 1
        bottommost_top = max(room_a.position.z, room_b.position.z)
        # The gap is one less than index difference (adjacent room idx delta is 1, but the "gap" is 0)
        gap = bottommost_top - topmost_bottom - 1
        return gap <= max_gap
    elif z_overlap is not None:
        leftmost_right = min(room_a.position.x + room_a.width, room_b.position.x + room_b.width) - 1
        rightmost_left = max(room_a.position.x, room_b.position.x)
        gap = rightmost_left - leftmost_right - 1
        return gap <= max_gap
    else:
        return False


def get_hallway_attachment_points(room_a: Room, room_b: Room) -> Tuple[BuildingTile, BuildingTile]:
    overlap_x, overlap_z = get_room_overlap(room_a, room_b)
    rightmost_room = room_a if room_a.center.x > room_b.center.x else room_b
    leftmost_room = room_a if rightmost_room == room_b else room_b
    bottommost_room = room_a if room_a.center.z > room_b.center.z else room_b
    topmost_room = room_a if bottommost_room == room_b else room_b
    if overlap_x:
        x_left = x_right = math.floor(overlap_x.midpoint)
        z_top = topmost_room.z_range.max_lt - 1
        z_bottom = bottommost_room.position.z
    elif overlap_z:
        z_top = z_bottom = math.floor(overlap_z.midpoint)
        x_left = leftmost_room.x_range.max_lt - 1
        x_right = rightmost_room.position.x
    else:
        x_left = min(room_a.x_range.max_lt, room_b.x_range.max_lt) - 1
        x_right = max(room_a.x_range.min_ge, room_b.x_range.min_ge)
        z_top = min(room_a.z_range.max_lt, room_b.z_range.max_lt) - 1
        z_bottom = max(room_a.z_range.min_ge, room_b.z_range.min_ge)

    attachment_a_z, attachment_b_z = (z_top, z_bottom) if topmost_room == room_a else (z_bottom, z_top)
    attachment_a_x, attachment_b_x = (x_left, x_right) if leftmost_room == room_a else (x_right, x_left)
    room_a_attachment_point = BuildingTile(attachment_a_x, attachment_a_z)
    room_b_attachment_point = BuildingTile(attachment_b_x, attachment_b_z)
    return room_a_attachment_point, room_b_attachment_point


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HallwayBuilder(Serializable):
    def build(self, rooms: List[Room], rand: np.random.Generator) -> List[Hallway]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class NoHallwayBuilder(HallwayBuilder):
    """For testing only - you want connections between rooms!"""

    def build(self, rooms: List[Room], rand: np.random.Generator) -> List[Hallway]:
        return []


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultHallwayBuilder(HallwayBuilder):
    proportion_additional_edges: float = 0.18

    def _get_room_centroids(self, rooms: List[Room]) -> Dict[int, Tuple[int, int]]:
        return {i: (r.center.x, r.center.z) for i, r in enumerate(rooms)}

    def _make_complete_room_graph(self, rooms: List[Room]) -> Graph:
        graph = nx.complete_graph(len(rooms))
        centroids_by_room = self._get_room_centroids(rooms)
        nx.set_node_attributes(graph, centroids_by_room, "pos")
        distance_by_rooms = {}
        for i, room_a in enumerate(rooms):
            for j, room_b in enumerate(rooms):
                distance_by_rooms[(i, j)] = euclidean_distance(room_a.center, room_b.center)
        nx.set_edge_attributes(graph, distance_by_rooms, "distance")
        return graph

    def _make_final_room_graph(self, rooms: List[Room], rand: np.random.Generator) -> Graph:
        graph = self._make_complete_room_graph(rooms)
        mst = nx.minimum_spanning_tree(graph, weight="distance")
        centroids_by_room = self._get_room_centroids(rooms)
        room_centroids = np.array([list(p) for p in centroids_by_room.values()])
        if len(rooms) > 2:
            try:
                room_triangulation = Delaunay(room_centroids)
            except QhullError:
                # This means the rooms are in a vertical/horizontal line and won't have meaningful extra connections
                return mst
            edges = get_triangulation_edges(room_triangulation)
            unjoined_edges = list(edges - set(mst.edges))
            additional_edge_count = round(self.proportion_additional_edges * len(unjoined_edges))
            additional_edges = rand.choice(unjoined_edges, additional_edge_count, replace=False)
            for edge_start, edge_end in additional_edges:
                mst.add_edge(edge_start, edge_end)
        return mst

    def build(self, rooms: List[Room], rand: np.random.Generator) -> List[Hallway]:
        if len(rooms) == 1:
            return []

        connection_graph = self._make_final_room_graph(rooms, rand)
        tile_graph = self._make_tile_graph(rooms)

        hallways = []
        for from_idx, to_idx in connection_graph.edges:
            from_room, to_room = rooms[from_idx], rooms[to_idx]
            hallway = self._build_hallway(tile_graph, from_room, to_room)
            hallways.append(hallway)
        return hallways

    def _build_hallway(self, tile_graph: RoomTileGraph, from_room: Room, to_room: Room) -> Hallway:
        from_room_point, to_room_point = get_hallway_attachment_points(from_room, to_room)
        path = tile_graph.find_shortest_path(from_room_point, to_room_point)

        optimized_path = [path[0]]
        for previous_point, this_point, next_point in zip(path[:-2], path[1:-1], path[2:]):
            previous_direction = Orientation.HORIZONTAL if this_point[0] == previous_point[0] else Orientation.VERTICAL
            next_direction = Orientation.HORIZONTAL if this_point[0] == next_point[0] else Orientation.VERTICAL
            if previous_direction != next_direction:
                optimized_path.append(this_point)
        optimized_path.append(path[-1])
        points = tuple(BuildingTile(x=pt[1], z=pt[0]) for pt in optimized_path)
        return Hallway(points, from_room.id, to_room.id, width=1)

    def _make_tile_graph(self, rooms: List[Room]) -> RoomTileGraph:
        return RoomTileGraph(rooms)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HamiltonianHallwayBuilder(DefaultHallwayBuilder):
    """
    Tries to create hallways such that they form a Hamiltonian path between all the rooms.

    Not all graphs have a Hamiltonian path and the implementation uses a heuristic-based TSP solutions, so the resulting
    path may not necessarily be Hamiltonian - but it's the closest we can trivially get without a brute-force solution.

    The builder checks for self-loops in the path (which would mean hallways traverse through other rooms), and raises
    an ImpossibleWorldError if all methods fail to create a path without self-loops.

    NOTE: This hallway builder is only compatible with HouseLikeRoomBuilder-built rooms.
    """

    def _make_final_room_graph(self, rooms: List[Room], rand: np.random.Generator) -> Graph:
        graph = self._make_complete_room_graph(rooms)
        for from_node, to_node in combinations(graph.nodes, 2):
            from_room = rooms[from_node]
            to_room = rooms[to_node]
            if not are_rooms_neighboring(from_room, to_room, max_gap=1):
                graph.remove_edge(from_node, to_node)

        if len(rooms) == 1:
            return graph
        path = traveling_salesman_problem(graph, cycle=False)
        is_path_hamiltonian = len(path) == len(set(path)) and len(path) == len(rooms)
        if not is_path_hamiltonian:
            raise ImpossibleWorldError("TSP solution yielded non-Hamiltonian paths")
        final_graph = nx.create_empty_copy(graph)  # keeps nodes, removes edges
        for from_node, to_node in zip(path[:-1], path[1:]):
            final_graph.add_edge(from_node, to_node)
        return final_graph


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StoryLinker(Serializable):
    def link_stories(self, stories: List[Story], rand: np.random.Generator) -> List[StoryLink]:
        raise NotImplementedError


class CantFitRampError(ImpossibleWorldError):
    pass


class CantFitLadderError(ImpossibleWorldError):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultStoryLinker(StoryLinker):
    raise_on_failure: bool = True
    allow_ramps: bool = True
    allow_ladders: bool = True

    def link_stories(self, stories: List[Story], rand: np.random.Generator) -> List[StoryLink]:
        links = []
        for bottom_story, top_story in zip(stories[:-1], stories[1:]):
            link = None
            if self.allow_ramps:
                try:
                    link = self.make_ramp(bottom_story, top_story, rand)
                except CantFitRampError:
                    pass

            if link is None and self.allow_ladders:
                link = self.make_ladder(bottom_story, top_story, rand)

            if link is None:
                if self.raise_on_failure:
                    raise ImpossibleWorldError(f"Could not link stories {bottom_story.num} and {top_story.num}")
                else:
                    continue
            links.append(link)
        return links

    def try_fit_ramp(
        self,
        bottom_story: Story,
        top_story: Story,
        bottom_room: Room,
        top_room: Room,
        bottom_story_tile_z: int,
        bottom_story_tile_x: int,
        bottom_story_tile_empty: bool,
        ramp_block: np.ndarray,
        ramp_mask: BuildingBoolNP,
        top_landing_mask: BuildingBoolNP,
        reverse: bool,
    ) -> Optional[Ramp]:
        ramp_block_length, ramp_block_width = ramp_block.shape

        if not bottom_story_tile_empty:
            return None

        z, x = bottom_story_tile_z, bottom_story_tile_x
        if reverse:
            ramp_within_bounds = z - ramp_block_length + 1 >= 0 and x - ramp_block_width + 1 >= 0
        else:
            ramp_within_bounds = (
                z + ramp_block_length - 1 <= ramp_mask.shape[0] and x + ramp_block_width - 1 <= ramp_mask.shape[1]
            )
        if not ramp_within_bounds:
            return None

        if reverse:
            ramp_space_free = (
                ramp_mask[z - ramp_block_length + 1 : z + 1, x - ramp_block_width + 1 : x + 1] == 1
            ).all()
        else:
            ramp_space_free = (ramp_mask[z : z + ramp_block_length, x : x + ramp_block_width] == 1).all()
        if not ramp_space_free:
            return None

        # TODO: check top landing properly when doing 1d9f913a-3870-41b0-839e-8b29d0b89983
        if reverse:
            top_landing_coords = z - ramp_block_length + 1, x - ramp_block_width + 1
        else:
            top_landing_coords = z + ramp_block_length - 1, x + ramp_block_width - 1
        top_landing_in_bounds = (
            0 <= top_landing_coords[0] < top_landing_mask.shape[0]
            and 0 <= top_landing_coords[1] < top_landing_mask.shape[1]
        )
        if not top_landing_in_bounds:
            return None

        top_landing_space_free = top_landing_mask[top_landing_coords] == 1
        if not top_landing_space_free:
            return None

        return Ramp(
            bottom_story.num,
            bottom_room.id,
            BuildingTile(x, z),
            top_story.num,
            top_room.id,
            BuildingTile(top_landing_coords[1], top_landing_coords[0]),
            width=1,
        )

    def make_ladder(self, bottom_story: Story, top_story: Story, rand: np.random.Generator) -> Optional[Ladder]:
        azimuths: List[Azimuth] = list(Azimuth)
        rand.shuffle(azimuths)  # type: ignore

        for azimuth in azimuths:
            bottom_story_free_mask = _get_story_free_mask(bottom_story)
            top_story_free_mask = _get_story_free_mask(top_story)

            # Ladders require two tiles: the principal tile where the ladder is placed + an adjacent tile one step
            # backwards from the ladder to make room for the fat player. The azimuth mask tells us if that tile is free
            backwards_adjacent_tile_kernel = azimuth.convolution_kernel
            forwards_adjacent_tile_kernel = azimuth.opposite.convolution_kernel
            # Top story needs a tile forward from ladder to "land" on
            has_top_landing_tile_mask = convolve(top_story_free_mask, forwards_adjacent_tile_kernel)
            ladder_site_mask = bottom_story_free_mask & top_story_free_mask
            # Both stories need a tile backwards as it'll be cut out
            backwards_adjacent_tile_mask = convolve(ladder_site_mask, backwards_adjacent_tile_kernel)
            ladder_site_mask &= backwards_adjacent_tile_mask & has_top_landing_tile_mask

            viable_tiles = list(zip(*np.where(ladder_site_mask)))
            if not viable_tiles:
                # For narrow buildings, only horizontal or vertical azimuths might work out
                continue

            ladder_tile = rand.choice(viable_tiles)
            ladder_position = BuildingTile(x=ladder_tile[1], z=ladder_tile[0])

            top_position = ladder_position
            if azimuth == Azimuth.NORTH:
                bottom_position = attr.evolve(top_position, z=top_position.z + 1)
            elif azimuth == Azimuth.EAST:
                bottom_position = attr.evolve(top_position, x=top_position.x - 1)
            elif azimuth == Azimuth.SOUTH:
                bottom_position = attr.evolve(top_position, z=top_position.z - 1)
            elif azimuth == Azimuth.WEST:
                bottom_position = attr.evolve(top_position, x=top_position.x + 1)
            else:
                raise SwitchError(f"Unknown azimuth {azimuth}")

            bottom_room = bottom_story.get_room_at_point(bottom_position)
            top_room = top_story.get_room_at_point(top_position)
            assert bottom_room is not None and top_room is not None

            # TODO: 1d9f913a-3870-41b0-839e-8b29d0b89983
            return Ladder(
                bottom_story_id=bottom_story.num,
                bottom_room_id=bottom_room.id,
                bottom_position=bottom_position,
                top_story_id=top_story.num,
                top_room_id=top_room.id,
                top_position=top_position,
                width=1,
                azimuth=azimuth,
            )

        if not self.raise_on_failure:
            return None
        raise CantFitLadderError(
            "Stories don't have enough overlap to place ladder."
            "Either something is very wrong or the building is too small"
        )

    def make_ramp(self, bottom_story: Story, top_story: Story, rand: np.random.Generator) -> Optional[StoryLink]:
        """
        The rough algorithm for connecting two stories with a ramp:
        1) Iterate over all bottom story rooms, starting with the biggest bottom one
        2) Iterate over all top story rooms, starting with nearest to current bottom one
        3) Calculate:
            bottom landing mask (which tiles can we place a landing on the bottom floor?)
            top landing mask
            ramp mask (basically all tiles except for rooms that we are not connecting and all hallways+landings)
        4) Iterate tile-by-tile in free bottom landing space
        5) Iterate over the 4 possible directions
        6) Try placing the ramp w/ landings: if ramp space and landing spaces are all clear, we can place the ramp!
        """
        assert (
            bottom_story.width == top_story.width and bottom_story.length == bottom_story.length
        ), "Different-sized story linking not implemented"

        for bottom_room in sorted(bottom_story.rooms, key=lambda room: room.tiles.size):
            for top_room in sorted(
                top_story.rooms, key=lambda top_room: euclidean_distance(top_room.center, bottom_room.center)
            ):
                ramp = self._try_link_rooms_by_ramp(bottom_story, top_story, bottom_room, top_room)
                if ramp is not None:
                    return ramp

        if not self.raise_on_failure:
            return None
        else:
            raise CantFitRampError("Building too small to place ramp")

    def _try_link_rooms_by_ramp(
        self, bottom_story: Story, top_story: Story, bottom_room: Room, top_room: Room
    ) -> Optional[Ramp]:
        ramp_width = 1
        landing_length = 1
        max_ramp_angle = math.radians(45)
        vertical_distance = bottom_story.outer_height + top_story.floor_negative_depth
        rounding_error_threshold_decimals = 2
        ramp_floor_length = math.ceil(
            round(vertical_distance / math.tan(max_ramp_angle), rounding_error_threshold_decimals)
        )
        required_space = (ramp_floor_length + 2 * landing_length, ramp_width)
        ramp_block = np.empty(required_space)

        bottom_room_hallways = [h for h in bottom_story.hallways if h.from_room_id == bottom_room.id]
        bottom_landing_mask = _get_room_free_mask(
            bottom_story.width,
            bottom_story.length,
            bottom_room,
            bottom_room_hallways,
            bottom_story.story_links,
        )
        top_room_hallways = [h for h in top_story.hallways if h.from_room_id == top_room.id]
        top_landing_mask = _get_room_free_mask(
            top_story.width, top_story.length, top_room, top_room_hallways, top_story.story_links
        )
        if not (bottom_landing_mask == 1).any():
            return None
        if not (top_landing_mask == 1).any():
            return None

        # free space = ones
        other_bottom_rooms = [room for room in bottom_story.rooms if room != bottom_room]
        other_top_rooms = [room for room in top_story.rooms if room != top_room]
        ramp_mask = _get_ramp_free_mask(
            bottom_story.width,
            bottom_story.length,
            exclude_rooms=[*other_bottom_rooms, *other_top_rooms],
            exclude_hallways=[*bottom_story.hallways, *top_story.hallways],
            exclude_story_links=[*bottom_story.story_links, *top_story.story_links],
        )

        for ramp_block in [ramp_block, ramp_block.T]:
            ramp_block_length, ramp_block_width = ramp_block.shape
            max_free_length = (ramp_mask == 1).sum(axis=0).max()
            max_free_width = (ramp_mask == 1).sum(axis=1).max()
            if ramp_block_length > max_free_length or ramp_block_width > max_free_width:
                continue
            for reverse in [False, True]:
                for (z, x), empty in np.ndenumerate(bottom_landing_mask == 1):
                    ramp = self.try_fit_ramp(
                        bottom_story,
                        top_story,
                        bottom_room,
                        top_room,
                        bottom_story_tile_z=z,
                        bottom_story_tile_x=x,
                        bottom_story_tile_empty=empty,
                        ramp_block=ramp_block,
                        ramp_mask=ramp_mask,
                        top_landing_mask=top_landing_mask,
                        reverse=reverse,
                    )
                    if ramp:
                        return ramp
        return None


def _get_room_free_mask(
    story_width: int, story_length: int, room: Room, hallways: List[Hallway], story_links: List[StoryLink]
) -> BuildingBoolNP:
    """Returns a mask where ones represent the free space in a room (free of ramps, landings, etc.)"""
    grid = np.zeros((story_length, story_width), dtype=np.float32)
    add_room_tiles_to_grid(room, room.tiles, grid)
    grid = grid.astype(bool)
    for hallway in hallways:
        grid[hallway.points[0].z, hallway.points[0].x] = 0
        grid[hallway.points[-1].z, hallway.points[-1].x] = 0
    for story_link in story_links:
        draw_line_in_grid((story_link.bottom_position, story_link.top_position), grid, 0, include_ends=True)
    return grid


def _get_story_free_mask(story: Story) -> BuildingBoolNP:
    story_free_mask = np.zeros((story.length, story.width), dtype=np.bool_)
    for room in story.rooms:
        room_free_mask = _get_room_free_mask(story.width, story.length, room, story.hallways, story.story_links)
        story_free_mask |= room_free_mask
    return story_free_mask.astype(np.bool_)


def _get_ramp_free_mask(
    story_width: int,
    story_length: int,
    exclude_rooms: List[Room],
    exclude_hallways: List[Hallway],
    exclude_story_links: List[StoryLink],
) -> BuildingBoolNP:
    """
    Returns a mask where ones represent the free space for building a ramp across multiple floors
    """
    grid = np.zeros((story_length, story_width), dtype=np.int_)
    for room in exclude_rooms:
        add_room_tiles_to_grid(room, room.tiles, grid)
    for hallway in exclude_hallways:
        draw_line_in_grid(hallway.points, grid, TileIdentity.HALLWAY.value, include_ends=True)
    for story_link in exclude_story_links:
        draw_line_in_grid(
            (story_link.bottom_position, story_link.top_position), grid, TileIdentity.LINK.value, include_ends=True
        )
    return np.invert(grid.astype(bool))


class ObstacleSite(NamedTuple):
    story_num: int
    room_id: int
    site_index: int  # not called index to avoid collision with NamedTuple.index
    length: int
    vertical: bool


def get_free_obstacle_sites(stories: List[Story]) -> List[ObstacleSite]:
    options = []
    for story in stories:
        for room in story.rooms:
            # TODO (1d9f913a-3870-41b0-839e-8b29d0b89983): support sites with width > 1
            # Gets all free verticals/horizontals, possibly in front of hallway joints
            global_room_free_mask = _get_room_free_mask(
                story.width, story.length, room, hallways=[], story_links=story.story_links
            )
            local_room_free_mask = global_room_free_mask[
                room.z_range.min_ge : room.z_range.max_lt, room.x_range.min_ge : room.x_range.max_lt
            ]
            free_vertical_idxs = only(np.where((local_room_free_mask == 1).all(axis=0)))
            options.extend([ObstacleSite(story.num, room.id, idx, room.length, True) for idx in free_vertical_idxs])
            free_horizontal_idxs = only(np.where((local_room_free_mask == 1).all(axis=1)))
            options.extend([ObstacleSite(story.num, room.id, idx, room.width, False) for idx in free_horizontal_idxs])
    return options


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class EntranceBuilder:
    def build(self, stories: List[Story], rand: np.random.Generator) -> List[Story]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultEntranceBuilder(EntranceBuilder):
    top: bool
    bottom: bool

    def build(self, stories: List[Story], rand: np.random.Generator) -> List[Story]:
        if self.top:
            # TODO: 8451efda-5f27-4ea3-a1d3-cde7f84f652c
            # self.add_entrance_ramp(stories[-1], rand)
            raise NotImplementedError
        if self.bottom:
            first_story = stories[0]
            self.add_entrance(first_story, rand)
        return stories

    def add_entrance(
        self,
        story: Story,
        rand: np.random.Generator,
        permitted_room_ids: Optional[List[int]] = None,
        permitted_azimuths: Optional[List[Azimuth]] = None,
    ) -> Entrance:
        entrance = self.make_entrance(story, rand, permitted_room_ids, permitted_azimuths)
        self.remove_windows_in_way_of_entrance(story, entrance)
        # TODO: think about how to do entrance/hallway relationship better
        story.entrances.append(entrance)
        story.hallways.append(entrance.hallway)
        return entrance

    def remove_windows_in_way_of_entrance(self, story: Story, entrance: Entrance) -> None:
        entrance_points_in_outline = set(entrance.get_points_in_story_outline(story))
        adjacent_points: Set[Tuple[int, int]] = set()
        for point in entrance_points_in_outline:
            adjacent_points.update(get_neighbor_tiles(story.get_outline_mask(), point, only_if_equals=True))
        entrance_points_in_outline.update(adjacent_points)

        for entry_z, entry_x in sorted(list(entrance_points_in_outline)):
            window_idxs_to_remove = []
            for i, window in enumerate(story.windows):
                window_x, _, window_z = window.position
                window_x_range = IntRange(
                    window.position[0] - window.size[0] / 2, window.position[0] + window.size[0] / 2
                )
                window_z_range = IntRange(
                    window.position[2] - window.size[2] / 2, window.position[2] + window.size[2] / 2
                )
                entry_tile_x_range = IntRange(entry_x, entry_x + 1)
                entry_tile_z_range = IntRange(entry_z, entry_z + 1)
                if entry_tile_x_range.overlap(window_x_range) and entry_tile_z_range.overlap(window_z_range):
                    window_idxs_to_remove.append(i)
            for idx in sorted(window_idxs_to_remove, reverse=True):
                del story.windows[idx]

    def make_entrance(
        self,
        story: Story,
        rand: np.random.Generator,
        permitted_room_ids: Optional[List[int]] = None,
        permitted_azimuths: Optional[List[Azimuth]] = None,
    ) -> Entrance:
        permitted_azimuths = list(Azimuth) if permitted_azimuths is None else permitted_azimuths
        tiles: np.ndarray = story.generate_tiles(include_hallway_landings=True)
        room_id_tiles = story.generate_room_id_tiles()
        outline_mask = story.get_outline_mask()
        is_tile_free = tiles == TileIdentity.ROOM.value

        # Find all possible story outline tiles that are not blocked
        # To do this, we create a fitting mask, where all free tiles are 1s (is_tile_free) and where the outline
        # is an arbitrary number other than 1.
        arbitrary_multiplier = 9
        fitting_mask = outline_mask.astype(np.int_) * arbitrary_multiplier + is_tile_free.astype(int)
        viable_connections = []
        for wall_type in WallType:
            azimuth = wall_type.azimuth
            if azimuth not in permitted_azimuths:
                continue

            # To check if a tile is suitable, we check that it is free (is_tile_free) AND that it  has an outline wall
            # of the desired azimuth as a neighbor (through the convolution with the wall kernel).
            # The expected mask value is 3 * arbitrary multiplier, since the wall kernels are all 1s on the wall side.
            kernel = wall_type.convolution_kernel
            assert kernel.shape[0] == kernel.shape[1]
            kernel_size = kernel.shape[0]
            is_suitable_entrance_site = is_tile_free.astype(int) & (
                convolve(fitting_mask, kernel, mode="constant") == kernel_size * arbitrary_multiplier
            )
            suitable_tiles = list(zip(*np.where(is_suitable_entrance_site == True)))
            for tile in suitable_tiles:
                interior_point = BuildingTile(z=tile[0], x=tile[1])
                room_id = room_id_tiles[interior_point.z, interior_point.x]
                if permitted_room_ids is not None and room_id not in permitted_room_ids:
                    continue

                if wall_type == WallType.NORTH:
                    exterior_point = BuildingTile(z=interior_point.z - 2, x=interior_point.x)
                elif wall_type == WallType.EAST:
                    exterior_point = BuildingTile(z=interior_point.z, x=interior_point.x + 2)
                elif wall_type == WallType.SOUTH:
                    exterior_point = BuildingTile(z=interior_point.z + 2, x=interior_point.x)
                elif wall_type == WallType.WEST:
                    exterior_point = BuildingTile(z=interior_point.z, x=interior_point.x - 2)
                else:
                    raise SwitchError("Unexpected wall type")

                viable_connections.append((room_id, azimuth, (exterior_point, interior_point)))

        # TODO: 1d9f913a-3870-41b0-839e-8b29d0b89983
        if len(viable_connections) == 0:
            raise ImpossibleWorldError(f"Can't build entryway to any room ({permitted_room_ids=})")
        connection_idx = rand.choice(list(range(len(viable_connections))))
        connected_room_id, azimuth, hallway_points = viable_connections[connection_idx]

        return Entrance(story.num, connected_room_id, azimuth, hallway_points, width=1)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WindowBuilder:
    def build(
        self, stories: List[Story], rand: np.random.Generator, aesthetics: BuildingAestheticsConfig
    ) -> List[Story]:
        stories_with_windows = []
        for story in stories:
            windows = []

            no_build_mask = np.zeros((story.length, story.width))
            for hallway in story.hallways:
                draw_line_in_grid(hallway.points, no_build_mask, TileIdentity.HALLWAY.value, include_ends=False)
            for link in story.story_links:
                set_link_in_grid(
                    link,
                    no_build_mask,
                    set_bottom_landing=link.bottom_story_id == story.num,
                    set_top_landing=link.top_story_id == story.num,
                )
            no_build_mask = no_build_mask.astype(np.bool_)

            wall_footprints = find_exterior_wall_footprints(story.footprint, no_build_mask=no_build_mask)
            for wall_footprint in wall_footprints:
                min_gap = aesthetics.window_min_gap
                window_width = aesthetics.window_width
                window_height = aesthetics.window_height
                window_centroid_y = story.outer_height * aesthetics.window_y_proportion_of_height
                centroids = get_evenly_spaced_centroids(
                    wall_footprint, window_width, min_gap, window_centroid_y, aesthetics.max_windows_per_wall
                )
                windows_for_wall = []
                for centroid in centroids:
                    if wall_footprint.is_vertical:
                        size = np.array([wall_footprint.wall_thickness, window_height, window_width])
                    else:
                        size = np.array([window_width, window_height, wall_footprint.wall_thickness])
                    window = Window(centroid, size)
                    windows_for_wall.append(window)
                windows.extend(windows_for_wall)

            if IS_DEBUG_VIS:
                self.visualise(story, wall_footprints, windows)

            stories_with_windows.append(attr.evolve(story, windows=windows))
        return stories_with_windows

    def visualise(self, story: Story, wall_footprints: List[WallFootprint], windows: List[Window]) -> None:
        fig, ax = plt.subplots()
        plt.gca().set_aspect("equal")
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.invert_yaxis()
        ax.set_xlim(-1, story.width + 1)
        ax.set_ylim(story.length + 1, -1)
        ax.grid()
        ax.set_title(f"Story {story.num}")

        for footprint in wall_footprints:
            rect = patches.Rectangle(
                footprint.top_left,
                footprint.footprint_width,
                footprint.footprint_length,
                0,
                linewidth=1,
                edgecolor="blue",
                facecolor="none",
            )
            ax.add_patch(rect)
        for window in windows:
            top_left = window.position[0] - window.size[0] / 2, window.position[2] - window.size[2] / 2
            rect = patches.Rectangle(
                top_left, window.size[0], window.size[2], linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

        plt.show()
