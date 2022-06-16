import math
from random import Random
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import networkx as nx
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from networkx import Graph
from scipy import stats
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

from common.errors import SwitchError
from common.utils import only
from contrib.serialization import Serializable
from datagen.godot_base_types import NewRange
from datagen.godot_base_types import Vector3
from datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from datagen.world_creation.geometry import Position
from datagen.world_creation.geometry import euclidean_distance
from datagen.world_creation.geometry import get_triangulation_edges
from datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from datagen.world_creation.indoor.helpers import draw_line_in_grid
from datagen.world_creation.indoor.helpers import visualize_tiles
from datagen.world_creation.indoor.objects import Azimuth
from datagen.world_creation.indoor.objects import BuildingAestheticsConfig
from datagen.world_creation.indoor.objects import Entrance
from datagen.world_creation.indoor.objects import FloorChasm
from datagen.world_creation.indoor.objects import Hallway
from datagen.world_creation.indoor.objects import Obstacle
from datagen.world_creation.indoor.objects import Orientation
from datagen.world_creation.indoor.objects import Ramp
from datagen.world_creation.indoor.objects import Room
from datagen.world_creation.indoor.objects import RoomPlacementError
from datagen.world_creation.indoor.objects import Story
from datagen.world_creation.indoor.objects import StoryLink
from datagen.world_creation.indoor.objects import TileIdentity
from datagen.world_creation.indoor.objects import Wall
from datagen.world_creation.indoor.objects import Window
from datagen.world_creation.indoor.objects import add_room_tiles_to_grid
from datagen.world_creation.indoor.objects import find_exterior_wall_footprints
from datagen.world_creation.indoor.objects import generate_blocks
from datagen.world_creation.indoor.objects import generate_room
from datagen.world_creation.indoor.objects import get_evenly_spaced_centroids
from datagen.world_creation.indoor.objects import set_link_in_grid
from datagen.world_creation.utils import IS_DEBUG_VIS
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.utils import inset_borders


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FootprintBuilder(Serializable):
    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[np.ndarray],
    ) -> np.ndarray:
        """returns a np.array of shape (building_length, building_width) where True represents buildable tiles"""
        raise NotImplementedError

    @staticmethod
    def inset_footprint(footprint: np.ndarray, inset: int = 1):
        reduced_footprint = footprint.copy()
        for i in range(inset):
            reduced_footprint = inset_borders(reduced_footprint)
        return reduced_footprint


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RectangleFootprintBuilder(FootprintBuilder):
    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[np.ndarray],
    ) -> np.ndarray:
        if story_num != 0:
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
        footprint_below: Optional[np.ndarray],
    ) -> np.ndarray:
        if story_num == 1:
            # Optionally inverting block generation direction gives us more variety
            inverted = rand.choice([True, False])
            footprint_below = footprint_below.copy().T if inverted else footprint_below.copy()
            blocks = generate_blocks(footprint_below, SOLID=True)
            min_block_size = 3  # avoid tiny sections that can't have rooms built into them
            block = rand.choice([b for b in blocks if b.x.size >= min_block_size and b.z.size > min_block_size])
            footprint = np.zeros((building_length, building_width), dtype=np.bool_)
            if inverted:
                tmp = block.x
                block.x = block.z
                block.z = tmp
            footprint[block.z.min_ge : block.z.max_lt, block.x.min_ge : block.x.max_lt] = 1
            return footprint
        elif story_num > 1:
            return FootprintBuilder.inset_footprint(footprint_below)

        rectangles = []
        footprint = np.zeros((building_length, building_width), dtype=np.bool_)

        rectangle_width = rand.integers(round(building_width * 0.4), round(building_width * 0.8))
        if building_length > building_width and self.allow_t_shape:
            x_offset = rand.choice([0, building_width - rectangle_width])
        else:
            x_offset = rand.integers(0, building_width - rectangle_width)
        rectangles.append((x_offset, 0, rectangle_width, building_length))

        rectangle_length = rand.integers(round(building_length * 0.4), round(building_length * 0.8))
        if building_width > building_length and self.allow_t_shape:
            z_offset = rand.choice([0, building_length - rectangle_length])
        else:
            z_offset = rand.integers(0, building_length - rectangle_length)
        rectangles.append((0, z_offset, building_width, rectangle_length))

        for x_offset, z_offset, building_width, building_length in rectangles:
            footprint[z_offset : z_offset + building_length, x_offset : x_offset + building_width] = 1
        return footprint


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class IrregularFootprintBuilder(FootprintBuilder):
    def build(
        self,
        building_width: int,
        building_length: int,
        story_num: int,
        rand: np.random.Generator,
        footprint_below: Optional[np.ndarray],
    ) -> np.ndarray:
        if story_num != 0:
            return FootprintBuilder.inset_footprint(footprint_below)

        footprint = np.zeros((building_length, building_width), dtype=np.bool_)
        rectangles = []
        for i in range(5):
            rectangle_width = rand.integers(round(building_width * 0.5), round(building_width * 0.9))
            rectangle_length = rand.integers(round(building_length * 0.5), round(building_length * 0.9))
            if i == 0:
                x_offset = rand.integers(0, building_width - rectangle_width)
                z_offset = rand.integers(0, building_length - rectangle_length)
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
                    z_offset = rand.integers(0, building_length - rectangle_length)
                else:
                    x_offset = rand.integers(0, building_width - rectangle_width)
                    z_offset = rand.choice(possible_z_offsets)
            rectangles.append((x_offset, z_offset, rectangle_width, rectangle_length))

        for x_offset, z_offset, building_width, building_length in rectangles:
            footprint[z_offset : z_offset + building_length, x_offset : x_offset + building_width] = 1
        return footprint


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RoomBuilder(Serializable):
    def build(self, story_footprint: np.array, rand: np.random.Generator) -> List[Room]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultRoomBuilder(RoomBuilder):
    room_count: int
    room_options: List[Vector3]

    def build(self, story_footprint: np.array, rand: np.random.Generator) -> List[Room]:
        world_length, world_width = story_footprint.shape

        rooms = []
        room_floor_space = 0
        for i in range(self.room_count):
            width, height, length = rand.choice(self.room_options)
            room = generate_room(i + 1, width, height, length)
            assert room.width < world_width and room.length < world_length, "Room too large to fit world"
            rooms.append(room)
            room_floor_space += width * length
        assert room_floor_space < world_width * world_length, "Room floor space exceeds world size"
        return self._place_rooms(story_footprint, rooms)

    def _place_rooms(self, story_footprint: np.ndarray, rooms: List[Room]) -> List[Room]:
        """
        At its core, this is a rectangle packing problem, which is NP-hard. There are many published algorithms out there
        that we could use if we want to, but the one below is a quick and dirty homemade one while we iterate this.
        """
        grid = np.invert(story_footprint).astype(np.int64)
        placed_rooms = []
        sorted_rooms = sorted(rooms, key=lambda r: r.floor_space, reverse=True)
        blocks = generate_blocks(grid)
        margin = 1
        for i, room in enumerate(sorted_rooms):
            for block in blocks:
                block_length = block.z.max_lt - block.z.min_ge
                block_width = block.x.max_lt - block.x.min_ge
                if block_width < room.width + (margin * 2) or block_length < room.length + (margin * 2):
                    continue
                center = block.x.min_ge + block_width // 2, block.z.min_ge + block_length // 2
                room = room.with_position(
                    Position(x=center[0] - room.width // 2, z=center[1] - room.length // 2)
                ).with_id(i)
                add_room_tiles_to_grid(room, room.tiles, grid)
                placed_rooms.append(room)
                blocks = generate_blocks(grid)
                break
            else:
                # The options here are: a) try a more compact placement algorithm; b) skip room; for now we just raise
                visualize_tiles(grid)
                raise RoomPlacementError(
                    f"can't fit room of size {room.width, room.length} with {margin=} in any block"
                )
        return placed_rooms


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CustomRoomBuilder(RoomBuilder):
    make_rooms: Callable[[Tuple[int, int], Random], List[Room]]

    def build(self, *args, **kwargs) -> List[Room]:
        return self.make_rooms(*args, **kwargs)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HouseLikeRoomBuilder(RoomBuilder):
    min_room_size: int = 3
    max_rooms: int = np.inf
    partition_thickness: int = 1

    def build(self, story_footprint: np.array, rand: np.random.Generator) -> List[Room]:
        world_length, world_width = story_footprint.shape
        random_state = np.random.RandomState(seed=rand.integers(0, 2 ** 32 - 1))

        # Separate footprint into rectangles as "initial rooms"
        padded_footprint = story_footprint.copy()
        for i in range(self.partition_thickness):
            padded_footprint = inset_borders(padded_footprint)
        footprint_blocks = generate_blocks(padded_footprint, SOLID=True)
        initial_rooms = [
            Room(-1, Position(block.x.min_ge, block.z.min_ge), block.x.size, block.z.size, DEFAULT_STORY_HEIGHT)
            for block in footprint_blocks
        ]
        initial_orientation = Orientation.HORIZONTAL

        # Then partition each of the rooms
        rooms = []
        if len(initial_rooms) > self.max_rooms:
            raise ImpossibleWorldError(
                f"max_rooms is {self.max_rooms}, but initial footprint split yields {len(initial_rooms)}"
            )
        elif len(initial_rooms) == self.max_rooms:
            rooms = initial_rooms
        else:
            room_credits = self.max_rooms - len(initial_rooms)  # how many more rooms can you create
            for room in initial_rooms:
                new_rooms = self._partition(
                    room,
                    initial_orientation,
                    world_width,
                    world_length,
                    self.partition_thickness,
                    room_credits,
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
    ) -> List[Room]:
        if leftover_room_credits == 0:
            return [room]

        if orientation == Orientation.HORIZONTAL:
            edge_size = room.width
        else:
            edge_size = room.length
        if edge_size < 2 * self.min_room_size + partition_thickness:
            return [room]
        # todo: use reduced-kurtosis normal distribution and scale ourselves
        partition_location_distribution = stats.uniform(
            self.min_room_size, edge_size - 2 * self.min_room_size - partition_thickness
        )
        partition_location = round(partition_location_distribution.rvs(random_state=random_state))
        second_room_offset = partition_location + partition_thickness
        if orientation == Orientation.HORIZONTAL:
            partition_a = Room(
                -1,
                Position(room.position.x, room.position.z),
                width=partition_location,
                length=room.length,
                outer_height=room.outer_height,
            )
            partition_b = Room(
                -1,
                Position(room.position.x + second_room_offset, room.position.z),
                width=room.width - second_room_offset,
                length=room.length,
                outer_height=room.outer_height,
            )

        else:
            partition_a = Room(
                -1,
                Position(room.position.x, room.position.z),
                width=room.width,
                length=partition_location,
                outer_height=room.outer_height,
            )
            partition_b = Room(
                -1,
                Position(room.position.x, room.position.z + second_room_offset),
                width=room.width,
                length=room.length - second_room_offset,
                outer_height=room.outer_height,
            )
        next_partition_dimension_size = world_width if orientation == Orientation.HORIZONTAL else world_length
        probability_keep_partitioning = edge_size / next_partition_dimension_size
        keep_partitioning_probabilities = [probability_keep_partitioning, 1 - probability_keep_partitioning]
        # todo: make this not independent (have branches that have a high value and ones that have low)
        keep_partitioning_a = random_state.choice([True, False], p=keep_partitioning_probabilities)
        keep_partitioning_b = random_state.choice([True, False], p=keep_partitioning_probabilities)
        orientation = orientation.other()
        leftover_room_credits -= 1
        return [
            *(
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
                else [partition_a]
            ),
            *(
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
                else [partition_b]
            ),
        ]


def get_room_overlap(room_a: Room, room_b: Room) -> Tuple[NewRange, NewRange]:
    return room_a.x_range.overlap(room_b.x_range), room_a.z_range.overlap(room_b.z_range)


def get_closest_points(room_a: Room, room_b: Room) -> Tuple[Position, Position]:
    # assumes no overlap; todo: assert?
    rightmost_room = room_a if room_a.center.x > room_b.center.x else room_b
    bottommost_room = room_a if room_a.center.z > room_b.center.z else room_b
    x_start = min(room_a.x_range.max_lt, room_b.x_range.max_lt) - 1
    x_end = max(room_a.x_range.min_ge, room_b.x_range.min_ge)
    if rightmost_room == bottommost_room:
        z_start = min(room_a.z_range.max_lt, room_b.z_range.max_lt) - 1
        z_end = max(room_a.z_range.min_ge, room_b.z_range.min_ge)
    else:
        z_start = max(room_a.z_range.min_ge, room_b.z_range.min_ge)
        z_end = min(room_a.z_range.max_lt, room_b.z_range.max_lt) - 1
    return Position(x_start, z_start), Position(x_end, z_end)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HallwayBuilder(Serializable):
    def build(self, rooms: List[Room], rng: Random) -> List[Hallway]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class NoHallwayBuilder(HallwayBuilder):
    """For testing only - you want connections between rooms!"""

    def build(self, rooms: List[Room], rng: Random) -> List[Hallway]:
        return []


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultHallwayBuilder(HallwayBuilder):
    proportion_additional_edges: float = 0.18

    def _generate_room_connection_tree(self, rooms: List[Room], rng: Random) -> Graph:
        graph = nx.complete_graph(len(rooms))
        centroids_by_room = {i: (r.center.x, r.center.z) for i, r in enumerate(rooms)}
        nx.set_node_attributes(graph, centroids_by_room, "pos")
        distance_by_rooms = {}
        for i, room_a in enumerate(rooms):
            for j, room_b in enumerate(rooms):
                distance_by_rooms[(i, j)] = euclidean_distance(room_a.center, room_b.center)
        nx.set_edge_attributes(graph, distance_by_rooms, "distance")
        mst = nx.minimum_spanning_tree(graph, weight="distance")
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
            additional_edges = rng.choices(unjoined_edges, k=additional_edge_count)
            for edge_start, edge_end in additional_edges:
                mst.add_edge(edge_start, edge_end)
        return mst

    def build(self, rooms: List[Room], rng: Random) -> List[Hallway]:
        if len(rooms) == 1:
            return []

        tree = self._generate_room_connection_tree(rooms, rng)

        hallways = []
        for from_idx, to_idx in tree.edges:
            from_room, to_room = rooms[from_idx], rooms[to_idx]
            overlap_x, overlap_z = get_room_overlap(from_room, to_room)
            if overlap_x:
                overlap_midpoint = overlap_x.min_ge + (overlap_x.max_lt - overlap_x.min_ge) // 2
                from_room_bottom = from_room.position.z + from_room.length
                to_room_bottom = to_room.position.z + to_room.length
                top_point_z = min(from_room_bottom, to_room_bottom)
                bottom_point_z = max(from_room.position.z, to_room.position.z)
                start = Position(x=overlap_midpoint, z=top_point_z - 1)
                end = Position(x=overlap_midpoint, z=bottom_point_z)
                if top_point_z == from_room_bottom:
                    points = [start, end]
                else:
                    points = [end, start]
            elif overlap_z:
                overlap_midpoint = overlap_z.min_ge + (overlap_z.max_lt - overlap_z.min_ge) // 2
                from_room_right = from_room.position.x + from_room.width
                to_room_right = to_room.position.x + to_room.width
                left_point_x = min(from_room_right, to_room_right)
                right_point_x = max(from_room.position.x, to_room.position.x)
                start = Position(x=left_point_x - 1, z=overlap_midpoint)
                end = Position(x=right_point_x, z=overlap_midpoint)
                if left_point_x == from_room_right:
                    points = [start, end]
                else:
                    points = [end, start]
            else:
                # todo: need to swap start/end to ensure points are in [from_room, to_room] order
                # raise NotImplementedError("L-shaped hallways are partly broken, talk to mx")
                closest_points = get_closest_points(from_room, to_room)
                start, end = closest_points

                points = [start, Position(x=start.x, z=end.z), end]
            hallway = Hallway(points, from_room.id, to_room.id, width=1)
            hallways.append(hallway)
        return hallways


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StoryLinker(Serializable):
    def link_stories(self, stories: List[Story], rng: Random) -> Tuple[List[Story], List[StoryLink]]:
        raise NotImplementedError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DefaultStoryLinker(StoryLinker):
    raise_on_failure: bool = True

    def link_stories(self, stories: List[Story], rng: Random) -> Tuple[List[Story], List[StoryLink]]:
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
        links = []
        for bottom_story, top_story in zip(stories[:-1], stories[1:]):
            link = self.get_story_link(bottom_story, top_story)
            if not link:
                continue
            links.append(link)
            bottom_story.story_links.append(link)
            top_story.story_links.append(link)
        return stories, links

    def try_fit_ramp(
        self,
        bottom_story,
        top_story,
        bottom_room,
        top_room,
        bottom_story_tile_z,
        bottom_story_tile_x,
        bottom_story_tile_empty,
        ramp_block,
        ramp_mask,
        top_landing_mask,
        reverse: bool,
    ) -> Optional[Ramp]:
        ramp_block_length, ramp_block_width = ramp_block.shape

        if not bottom_story_tile_empty:
            return

        z, x = bottom_story_tile_z, bottom_story_tile_x
        if reverse:
            ramp_within_bounds = z - ramp_block_length + 1 >= 0 and x - ramp_block_width + 1 >= 0
        else:
            ramp_within_bounds = (
                z + ramp_block_length - 1 <= ramp_mask.shape[0] and x + ramp_block_width - 1 <= ramp_mask.shape[1]
            )
        if not ramp_within_bounds:
            return

        if reverse:
            ramp_space_free = (
                ramp_mask[z - ramp_block_length + 1 : z + 1, x - ramp_block_width + 1 : x + 1] == 1
            ).all()
        else:
            ramp_space_free = (ramp_mask[z : z + ramp_block_length, x : x + ramp_block_width] == 1).all()
        if not ramp_space_free:
            return

        # todo: check top landing properly if width > 1
        if reverse:
            top_landing_coords = z - ramp_block_length + 1, x - ramp_block_width + 1
        else:
            top_landing_coords = z + ramp_block_length - 1, x + ramp_block_width - 1
        top_landing_in_bounds = (
            0 <= top_landing_coords[0] < top_landing_mask.shape[0]
            and 0 <= top_landing_coords[1] < top_landing_mask.shape[1]
        )
        if not top_landing_in_bounds:
            return

        top_landing_space_free = top_landing_mask[top_landing_coords] == 1
        if not top_landing_space_free:
            return

        return Ramp(
            bottom_story.num,
            bottom_room.id,
            Position(x, z),
            top_story.num,
            top_room.id,
            Position(top_landing_coords[1], top_landing_coords[0]),
            width=1,
        )

    def get_story_link(self, bottom_story: Story, top_story: Story) -> Optional[StoryLink]:
        assert (
            bottom_story.width == top_story.width and bottom_story.length == bottom_story.length
        ), "Different-sized story linking not implemented"
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

        ramp = None
        linked = False
        for bottom_room in sorted(bottom_story.rooms, key=lambda room: room.tiles.size):
            for top_room in sorted(
                top_story.rooms, key=lambda top_room: euclidean_distance(top_room.center, bottom_room.center)
            ):
                # todo: add margins around hallways & rooms if we care about having enclosed ramps vs free ramps?
                bottom_room_hallways = [h for h in bottom_story.hallways if h.from_room_id == bottom_room.id]
                bottom_landing_mask = _get_room_free_mask(
                    bottom_story.width, bottom_story.length, bottom_room, bottom_room_hallways
                )
                top_room_hallways = [h for h in top_story.hallways if h.from_room_id == top_room.id]
                # todo: add ramps=existing_ramps
                top_landing_mask = _get_room_free_mask(top_story.width, top_story.length, top_room, top_room_hallways)
                if not (bottom_landing_mask == 1).any():
                    continue
                if not (top_landing_mask == 1).any():
                    continue

                # free space = ones
                other_bottom_rooms = [room for room in bottom_story.rooms if room != bottom_room]
                other_top_rooms = [room for room in top_story.rooms if room != top_room]
                ramp_mask = _get_ramp_free_mask(
                    bottom_story.width,
                    bottom_story.length,
                    exclude_rooms=[*other_bottom_rooms, *other_top_rooms],
                    exclude_hallways=[*bottom_story.hallways, *top_story.hallways]
                    # todo: exclude_story_links=existing_ramps
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
                            if not ramp:
                                continue
                            linked = True
                            break

                        # todo: are these breaks an anti-pattern?
                        if linked:
                            break
                    if linked:
                        break
                if linked:
                    break
            if linked:
                break
        if not linked or ramp is None:
            if not self.raise_on_failure:
                return None
            raise RuntimeError("can't place ramp ;_;")
        return ramp


def _get_room_free_mask(story_width: int, story_length: int, room: Room, hallways: List[Hallway]) -> np.ndarray:
    """Returns a mask where ones represent the free space in a room (free of ramps, landings, etc.)"""
    grid = np.zeros((story_length, story_width), dtype=np.float32)
    add_room_tiles_to_grid(room, room.tiles, grid)
    grid = grid.astype(bool)
    for hallway in hallways:
        grid[hallway.points[0].z, hallway.points[0].x] = 0
        grid[hallway.points[-1].z, hallway.points[-1].x] = 0
    return grid


def _get_ramp_free_mask(
    story_width: int, story_length: int, exclude_rooms: List[Room], exclude_hallways: List[Hallway]
) -> np.ndarray:
    """
    Returns a mask where ones represent the free space for building a ramp across multiple floors
    """
    grid = np.zeros((story_length, story_width), dtype=np.int_)
    for room in exclude_rooms:
        add_room_tiles_to_grid(room, room.tiles, grid)
    for hallway in exclude_hallways:
        draw_line_in_grid(hallway.points, grid, TileIdentity.HALLWAY.value, include_ends=True)
    return np.invert(grid.astype(bool))


class ObstacleSite(NamedTuple):
    story_num: int
    room_id: int
    index: int
    length: int
    vertical: bool


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ObstacleBuilder:
    obstacle_count: Dict[Type[Obstacle], int]

    # todo: move out of class?
    @staticmethod
    def get_free_sites(stories: List[Story]) -> List[ObstacleSite]:
        options = []
        for story in stories:
            for room in story.rooms:
                # todo: width > 1
                # Place obstacle anywhere there's a free vertical/horizontal, possibly in front of hallway joints
                global_room_free_mask = _get_room_free_mask(story.width, story.length, room, hallways=[])
                local_room_free_mask = global_room_free_mask[
                    room.z_range.min_ge : room.z_range.max_lt, room.x_range.min_ge : room.x_range.max_lt
                ]
                free_vertical_idxs = only(np.where((local_room_free_mask == 1).all(axis=0)))
                options.extend(
                    [ObstacleSite(story.num, room.id, idx, room.length, True) for idx in free_vertical_idxs]
                )
                free_horizontal_idxs = only(np.where((local_room_free_mask == 1).all(axis=1)))
                options.extend(
                    [ObstacleSite(story.num, room.id, idx, room.width, False) for idx in free_horizontal_idxs]
                )
        return options

    @staticmethod
    def apply(stories: List[Story], obstacles: List[Obstacle]) -> None:
        for obstacle in obstacles:
            if isinstance(obstacle, Wall) or isinstance(obstacle, FloorChasm):
                if isinstance(obstacle, Wall):
                    height = obstacle.height
                else:
                    height = -1
                room: Room = stories[obstacle.story_id].rooms[obstacle.room_id]
                new_heightmap = room.floor_heightmap.copy()
                draw_line_in_grid(obstacle.points, new_heightmap, height)
                deformed_room = room.with_heightmap(new_heightmap)
                stories[obstacle.story_id].rooms[obstacle.room_id] = deformed_room
            else:
                raise SwitchError(obstacle)

    def generate(self, stories: List[Story], rand: np.random.Generator) -> List[Obstacle]:
        all_obstacles = []
        for obstacle_type, count in self.obstacle_count.items():
            for _ in range(count):
                free_sites = self.get_free_sites(stories)
                site_idx = rand.choice(np.array(range(len(free_sites))), replace=False)
                site = free_sites[site_idx]
                story_num, room_id, idx, distance, vertical = site
                if vertical:
                    points = [Position(x=idx, z=0), Position(x=idx, z=distance - 1)]
                else:
                    points = [Position(x=0, z=idx), Position(x=distance - 1, z=idx)]
                if issubclass(obstacle_type, Wall):
                    height = round(max(MAX_JUMP_HEIGHT_METERS / 1, rand.random() * MAX_JUMP_HEIGHT_METERS), 1)
                    obstacle = Wall(story_num, room_id, points, 1, height)
                elif issubclass(obstacle_type, FloorChasm):
                    obstacle = FloorChasm(story_num, room_id, points, 1)
                else:
                    raise SwitchError(obstacle_type)
                all_obstacles.append(obstacle)
        return all_obstacles


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CustomObstacleBuilder(ObstacleBuilder):
    obstacle_count = np.nan
    generate_obstacles: Callable[[List[Story], np.random.Generator], List[Obstacle]]

    def generate(self, stories: List[Story], rand: np.random.Generator) -> List[Obstacle]:
        return self.generate_obstacles(stories, rand)


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
        entrance_points_in_outline = entrance.get_points_in_story_outline(story)
        for entry_z, entry_x in entrance_points_in_outline:
            window_idxs_to_remove = []
            for i, window in enumerate(story.windows):
                window_x, _, window_z = window.position
                window_x_range = NewRange(
                    window.position[0] - window.size[0] / 2, window.position[0] + window.size[0] / 2
                )
                window_z_range = NewRange(
                    window.position[2] - window.size[2] / 2, window.position[2] + window.size[2] / 2
                )
                entry_tile_x_range = NewRange(entry_x, entry_x + 1)
                entry_tile_z_range = NewRange(entry_z, entry_z + 1)
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
        exterior_space_by_azimuth = {
            Azimuth.NORTH: Room(-1, Position(0, -1), story.width, 1, 0),  # top
            Azimuth.SOUTH: Room(-2, Position(0, story.length), story.width, 1, 0),  # bottom
            Azimuth.WEST: Room(-3, Position(-1, 0), 1, story.length, 0),  # left
            Azimuth.EAST: Room(-4, Position(story.width + 1, 0), 1, story.length, 0),  # right
        }

        exterior_wall_azimuths_by_room_id = story.get_exterior_wall_azimuths_by_room_id()

        closest_distance = np.inf
        viable_connections = []
        for room in story.rooms:
            if room.id not in permitted_room_ids:
                continue

            viable_azimuths = set(exterior_wall_azimuths_by_room_id[room.id]).intersection(permitted_azimuths)
            if len(viable_azimuths) == 0:
                continue
            azimuth = rand.choice(list(viable_azimuths))

            exterior_space = exterior_space_by_azimuth[azimuth]
            overlap_x, overlap_z = get_room_overlap(exterior_space, room)
            if overlap_x:
                overlap_midpoint = overlap_x.min_ge + (overlap_x.max_lt - overlap_x.min_ge) // 2
                exterior_space_bottom = exterior_space.position.z + exterior_space.length
                room_bottom = room.position.z + room.length
                top_point_z = min(exterior_space_bottom, room_bottom)
                bottom_point_z = max(exterior_space.position.z, room.position.z)
                start = Position(x=overlap_midpoint, z=top_point_z - 1)
                end = Position(x=overlap_midpoint, z=bottom_point_z)
                if top_point_z == exterior_space_bottom:
                    points = [start, end]
                else:
                    points = [end, start]
                distance_to_exterior = bottom_point_z - top_point_z
            elif overlap_z:
                overlap_midpoint = overlap_z.min_ge + (overlap_z.max_lt - overlap_z.min_ge) // 2
                exterior_space_right = exterior_space.position.x + exterior_space.width
                room_right = room.position.x + room.width
                left_point_x = min(exterior_space_right, room_right)
                right_point_x = max(exterior_space.position.x, room.position.x)
                start = Position(x=left_point_x - 1, z=overlap_midpoint)
                end = Position(x=right_point_x, z=overlap_midpoint)
                if left_point_x == exterior_space_right:
                    points = [start, end]
                else:
                    points = [end, start]
                distance_to_exterior = right_point_x - left_point_x
            else:
                continue

            if distance_to_exterior <= closest_distance:
                connection = (room, azimuth, points)
                if distance_to_exterior == closest_distance:
                    viable_connections.append(connection)
                else:
                    viable_connections = [connection]
                closest_distance = distance_to_exterior

        # todo: width > 1
        assert len(viable_connections) > 0, f"Can't build entryway to any room ({permitted_room_ids=})"
        connection_idx = rand.choice(list(range(len(viable_connections))))
        room_to_connect, azimuth, hallway_points = viable_connections[connection_idx]

        return Entrance(story.num, room_to_connect.id, azimuth, hallway_points, width=1)


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
                set_link_in_grid(link, no_build_mask, bottom=True)
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

    def visualise(self, story, wall_footprints, windows):
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
