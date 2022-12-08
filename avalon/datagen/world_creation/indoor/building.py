import colorsys
import math
from collections import defaultdict
from enum import Enum
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import cast

import attr
import networkx as nx
import numpy as np
from godot_parser import GDSubResourceSection
from godot_parser import Node as GDNode
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes as MplAxes
from networkx import Graph
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box
from trimesh import Trimesh
from trimesh import creation

from avalon.common.errors import SwitchError
from avalon.common.utils import only
from avalon.datagen.godot_base_types import FloatRange
from avalon.datagen.godot_base_types import IntRange
from avalon.datagen.world_creation.debug_plots import IS_DEBUG_VIS
from avalon.datagen.world_creation.geometry import Axis
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import euclidean_distance
from avalon.datagen.world_creation.geometry import global_to_local_coords
from avalon.datagen.world_creation.geometry import local_to_global_coords
from avalon.datagen.world_creation.geometry import midpoint
from avalon.datagen.world_creation.indoor.blocks import BlocksByStory
from avalon.datagen.world_creation.indoor.blocks import CeilingBlock
from avalon.datagen.world_creation.indoor.blocks import FloorBlock
from avalon.datagen.world_creation.indoor.blocks import LadderBlock
from avalon.datagen.world_creation.indoor.blocks import LevelBlock
from avalon.datagen.world_creation.indoor.blocks import WallBlock
from avalon.datagen.world_creation.indoor.blocks import WindowBlock
from avalon.datagen.world_creation.indoor.blocks import make_blocks_from_tiles
from avalon.datagen.world_creation.indoor.blocks import new_make_block_nodes
from avalon.datagen.world_creation.indoor.components import Ladder
from avalon.datagen.world_creation.indoor.components import Ramp
from avalon.datagen.world_creation.indoor.components import Room
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.indoor.components import StoryLink
from avalon.datagen.world_creation.indoor.components import Window
from avalon.datagen.world_creation.indoor.components import set_link_in_grid
from avalon.datagen.world_creation.indoor.components import set_room_tiles_in_grid
from avalon.datagen.world_creation.indoor.constants import CEILING_THICKNESS
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import EXPORT_MODE
from avalon.datagen.world_creation.indoor.constants import HIGH_POLY
from avalon.datagen.world_creation.indoor.constants import LOW_POLY
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.constants import CornerType
from avalon.datagen.world_creation.indoor.constants import ExportMode
from avalon.datagen.world_creation.indoor.constants import TileIdentity
from avalon.datagen.world_creation.indoor.helpers import rotate_position
from avalon.datagen.world_creation.indoor.mesh import MeshData
from avalon.datagen.world_creation.indoor.mesh import homogeneous_transform_matrix
from avalon.datagen.world_creation.indoor.mesh import make_color_visuals
from avalon.datagen.world_creation.indoor.mesh import unnormalize
from avalon.datagen.world_creation.indoor.tiles import draw_line_in_grid
from avalon.datagen.world_creation.indoor.tiles import find_corners
from avalon.datagen.world_creation.indoor.tiles import find_exterior_wall_footprints
from avalon.datagen.world_creation.indoor.utils import get_evenly_spaced_centroids
from avalon.datagen.world_creation.indoor.utils import inset_borders
from avalon.datagen.world_creation.region import Region
from avalon.datagen.world_creation.types import BuildingBoolNP
from avalon.datagen.world_creation.types import GDLinkedSection
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.types import RGBATuple
from avalon.datagen.world_creation.types import RawGDObject
from avalon.datagen.world_creation.utils import ARRAY_MESH_TEMPLATE
from avalon.datagen.world_creation.utils import make_transform


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BuildingAestheticsConfig:
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

    crossbeam_color: RGBATuple = (0.19462, 0.13843, 0.02624, 1.0)
    exterior_color: RGBATuple = (0.66539, 0.56471, 0.28744, 1.0)
    trim_color: RGBATuple = (0.88792, 0.78354, 0.50289, 1.0)
    interior_wall_color: RGBATuple = (0.66539, 0.56471, 0.28744, 1.0)
    interior_wall_color_jitter: float = 0.02
    interior_wall_brightness_jitter: float = 0.025
    floor_color: RGBATuple = (0.88792, 0.78354, 0.50289, 1.0)
    ceiling_color: RGBATuple = (0.88792, 0.78354, 0.50289, 1.0)
    ladder_color: RGBATuple = (0.223529, 0.196078, 0.141176, 1.0)

    block_downsize_epsilon: float = 1e-5  # By how much to reduce block extents to avoid coincident faces


LevelBlockType = TypeVar("LevelBlockType", bound=LevelBlock)


@attr.s(auto_attribs=True, eq=False, collect_by_mro=True)
class Building:
    id: int
    stories: List[Story]
    story_links: List[StoryLink]
    position: Point3DNP
    yaw_degrees: float = 0
    is_climbable: bool = False
    aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig()

    def __attrs_post_init__(self) -> None:
        assert self.yaw_degrees == 0, "Building rotation currently not working, yaw_degrees must equal 0"

    @property
    def region(self) -> Region:
        return Region(
            x=FloatRange(self.offset_point[0], self.offset_point[0] + self.width),
            z=FloatRange(self.offset_point[2], self.offset_point[2] + self.length),
        )

    @property
    def height(self) -> FloatRange:
        return FloatRange(self.position[1], self.position[1] + sum([s.outer_height for s in self.stories]))

    @property
    def width(self) -> int:
        """
        Building width and height refers to its "bounding box"; the actual structure width/height is dependent
        on the footprints of individual stories. All story widths/heights are equal to each other and to the
        building width/height - variation in effective (traversable) width/height is all done via footprints.
        """
        return self.stories[0].width

    @property
    def length(self) -> int:
        return self.stories[0].length

    @property
    def offset_point(self) -> Point3DNP:
        # Where to offset any building space coordinates from to get the global ones
        return cast(Point3DNP, self.position - np.array([self.width / 2, 0, self.length / 2]))

    def with_transform(
        self, new_position: Optional[Point3DNP] = None, new_yaw_degrees: Optional[float] = None
    ) -> "Building":
        kwargs: Dict[str, Any] = {}
        if new_position is not None:
            kwargs["position"] = new_position
        if new_yaw_degrees is not None:
            kwargs["yaw_degrees"] = new_yaw_degrees
        return attr.evolve(self, **kwargs)

    def get_positive_height_at_point(self, point: Point3DNP) -> float:
        """
        Positive height refers to the exterior building height from it's y origin.
        Note: param point must be in building space.
        Returns np.nan if point lies outside building.
        """
        tile = math.floor(point[0]), math.floor(point[2])
        positive_height_at_point = 0.0
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

    def get_footprint_outline(self, story_num: int = 0) -> Polygon:
        tile_size = 1
        story = self.stories[story_num]
        corners = find_corners(story.footprint)
        points = []

        # Get corner centroids (positions are their top-left corners)
        for local_position, corner_type, is_outside in corners:
            # Note #1: Shapely ignores the Z coordinate, so we must make sure to only pass two coordinates here
            # We pass them as X -> X and Z -> Y
            # Note #2: Although this footprint is in building space, we _do_ have to offset by negative half-width since
            # the building origin (0,0) in Godot is its centroid and not it's top left-left corner as it is for tiles.
            points.append(
                Point(
                    local_position[1] - self.width / 2 + tile_size / 2,
                    local_position[0] - self.length / 2 + tile_size / 2,
                )
            )
        outline = Polygon(points)

        # Scale polygon to match OUTER edge of self, not tile midline
        min_x, min_y, max_x, max_y = outline.bounds
        width, length = max_x - min_x, max_y - min_y
        tile_size = 1
        x_scaling_factor = (width + tile_size) / width
        z_scaling_factor = (length + tile_size) / length
        outline = affinity.scale(outline, x_scaling_factor, z_scaling_factor)
        return outline

    def translate_block_to_building_space(self, block: LevelBlock, story: Story, epsilon: float = 0) -> LevelBlock:
        x_offset = -story.width / 2
        y_offset = self.get_story_y_offset(story.num)
        z_offset = -story.length / 2

        width, height, length = block.size
        adjusted_size = np.array([width - epsilon, height - epsilon, length - epsilon])
        x = x_offset + block.x.min_ge + width / 2
        y = y_offset + block.y.min_ge + height / 2
        z = z_offset + block.z.min_ge + length / 2
        centroid = np.array([x, y, z])
        return type(block).make(centroid, adjusted_size, block.rotation)

    def _get_pretty_visual_nodes(
        self,
        scene: GodotScene,
        blocks_by_story_by_kind: BlocksByStory,
        building_material: GDLinkedSection,
        building_name: str,
    ) -> List[GDNode]:
        DEFAULT_MESH = HIGH_POLY
        meshes = make_pretty_building_meshes(self, blocks_by_story_by_kind)
        visual_nodes = []
        for mesh_name, mesh in meshes.items():

            def _print_array(array: np.ndarray, precision: int) -> str:
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
            visual_nodes.append(mesh_node)
        return visual_nodes

    def export(
        self,
        scene: GodotScene,
        parent_node: GDNode,
        building_name: str,
        is_indoor_lighting_enabled: bool = False,
        scaling_factor: float = 1,
    ) -> None:
        blocks_by_story: BlocksByStory = {}
        for story in self.stories:
            blocks_by_story[story.num] = self._generate_blocks(story.num)

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

        if EXPORT_MODE == ExportMode.PRETTY:
            for visual_node in self._get_pretty_visual_nodes(scene, blocks_by_story, building_material, building_name):
                mesh_group.add_child(visual_node)

        light_group = GDNode("lights", type="Spatial")
        building_node.add_child(light_group)
        if is_indoor_lighting_enabled:
            building_y_midpoint = (self.height.size / 2) - DEFAULT_FLOOR_THICKNESS
            light_centroid = 0, building_y_midpoint, 0
            first_story = self.stories[0]
            indoor_light = GDNode(
                f"OmniLight{self.id}",
                type="OmniLight",
                properties={
                    "transform": make_transform(position=light_centroid),
                    "light_energy": 0.5,
                    "omni_range": max(
                        math.sqrt(first_story.width**2 + first_story.length**2) / 2, self.height.size / 2
                    ),
                },
            )
            light_group.add_child(indoor_light)

        static_body_group = GDNode("static_bodies", type="Spatial")
        building_node.add_child(static_body_group)

        for story_num, blocks in blocks_by_story.items():
            story = self.stories[story_num]

            for i, block in enumerate(blocks):
                if not block.is_collider:
                    # When generating meshes, we only care for collider blocks here
                    continue

                block_in_building_space = self.translate_block_to_building_space(block, story)
                block_script = parent_extra_props = None
                if isinstance(block, FloorBlock) and block.is_climbable:
                    block_script = scene.add_ext_resource("res://items/climbable.gd", "Script").reference
                    parent_extra_props = dict(entity_id=0)
                story_block_nodes = new_make_block_nodes(
                    scene,
                    position=np.array(block_in_building_space.centroid),
                    size=np.array(block_in_building_space.size),
                    rotation=block_in_building_space.rotation.as_matrix().flatten(),
                    make_mesh=EXPORT_MODE == ExportMode.DEBUG and block.is_visual,
                    mesh_material=building_material,
                    parent_name=f"story_{story_num}_{block.type_name}_static_body_{i}",
                    parent_script=block_script,
                    parent_extra_props=parent_extra_props,
                )
                for block_node in story_block_nodes:
                    static_body_group.add_child(block_node)

        story_link_material = scene.add_ext_resource("res://shaders/BasicColor.material", "Material")
        story_links_node = GDNode(f"story_links", type="Spatial")
        building_node.add_child(story_links_node)
        include_meshes = EXPORT_MODE == ExportMode.DEBUG

        for story_link_num, story_link in enumerate(self.story_links):
            bottom_story = self.stories[story_link.bottom_story_id]
            top_story = self.stories[story_link.top_story_id]
            story_link_blocks = story_link.get_level_blocks(bottom_story, top_story)
            for i, level_block in enumerate(story_link_blocks):
                translated_block = self.translate_block_to_building_space(level_block, bottom_story)
                # Story links need to be additionally offset by the floor thickness
                translated_block.y = FloatRange(
                    translated_block.y.min_ge + DEFAULT_FLOOR_THICKNESS,
                    translated_block.y.max_lt + DEFAULT_FLOOR_THICKNESS,
                )

                parent_script = None
                if isinstance(level_block, (WallBlock, LadderBlock)) and level_block.is_climbable:
                    parent_script = scene.add_ext_resource("res://items/climbable.gd", "Script").reference

                story_link_block_nodes = new_make_block_nodes(
                    scene,
                    position=np.array(translated_block.centroid),
                    size=np.array(translated_block.size),
                    rotation=translated_block.rotation.as_matrix().flatten(),
                    make_mesh=include_meshes,
                    mesh_material=story_link_material,
                    parent_name=f"story_link_{story_link_num}_{i}",
                    parent_script=parent_script,
                    parent_extra_props=dict(entity_id=0),
                )
                for story_link_block_node in story_link_block_nodes:
                    story_links_node.add_child(story_link_block_node)

    def generate_tiles(
        self, story_num: int, include_links_going_up: bool = True, include_links_going_down: bool = True
    ) -> BuildingBoolNP:
        story = self.stories[story_num]
        return story.generate_tiles(include_links_going_up, include_links_going_down)

    def _generate_blocks(self, story_num: int) -> List[LevelBlock]:
        story = self.stories[story_num]
        return [
            *self._generate_terrain_blocks(story_num),
            *self._generate_hallway_floor_blocks(story_num),
            *self._generate_hallway_ceiling_blocks(story_num),
            *self._generate_interior_wall_blocks(story_num),
            *self._generate_exterior_wall_blocks(story_num),
            *(self._generate_ceiling_blocks(story_num) if story.has_ceiling else []),
            *self._generate_window_blocks(story_num),
        ]

    def get_story_y_offset(self, requested_story_num: int) -> float:
        # Returns the y position for where it's positive floor should begin
        assert requested_story_num <= len(self.stories)

        y: float = -DEFAULT_FLOOR_THICKNESS  # so that y=0 usually equates to top of ground story floor
        for i, story in enumerate(self.stories):
            if story.num > 0:
                y += story.floor_negative_depth
            if story.num == requested_story_num:
                return y
            y += story.outer_height
        raise ValueError("Requested story num was not hit")

    def _generate_interior_wall_blocks(self, story_num: int) -> Sequence[WallBlock]:
        story = self.stories[story_num]
        wall_blocks: List[WallBlock] = []
        story_tiles = self.generate_tiles(story_num)
        is_outline = story.get_outline_mask()
        is_wall = np.invert(story_tiles.astype(np.bool_))
        interior_wall_tiles = is_wall & ~is_outline
        interior_wall_blocks = make_blocks_from_tiles(interior_wall_tiles, solid_tile_value=True)

        for block in interior_wall_blocks:
            stretched_block = WallBlock(
                block.x,
                FloatRange(-story.floor_negative_depth, story.outer_height - CEILING_THICKNESS),
                block.z,
                is_interior=True,
            )
            cut_blocks = self._cut_windows(stretched_block, story.windows)
            wall_blocks.extend(cut_blocks)
        return wall_blocks

    def _generate_exterior_wall_blocks(self, story_num: int) -> Sequence[WallBlock]:
        story = self.stories[story_num]
        wall_blocks: List[WallBlock] = []
        story_tiles = self.generate_tiles(story_num)
        is_outline = story.get_outline_mask()

        exterior_wall_tiles = np.full_like(story_tiles, fill_value=TileIdentity.ROOM.value)
        exterior_wall_tiles[is_outline] = story_tiles[is_outline]
        exterior_wall_blocks = make_blocks_from_tiles(exterior_wall_tiles)

        for block in exterior_wall_blocks:
            offset_block = WallBlock(
                block.x, FloatRange(-story.floor_negative_depth, story.outer_height), block.z, is_interior=False
            )
            cut_blocks = self._cut_windows(offset_block, story.windows)
            wall_blocks.extend(cut_blocks)
        return wall_blocks

    def _generate_hallway_floor_blocks(self, story_num: int) -> Sequence[FloorBlock]:
        # Note that to cover all cases, we actually want to build each hallway separately, making staircases between
        # rooms of different heights. For now assume hallways only connect height=1(default floor thickness) terrain.
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
            FloorBlock(block.x, FloatRange(-story.floor_negative_depth, DEFAULT_FLOOR_THICKNESS), block.z)
            for block in make_blocks_from_tiles(hallway_tiles, solid_tile_value=TileIdentity.HALLWAY.value)
        ]

    def _generate_hallway_ceiling_blocks(self, story_num: int) -> Sequence[CeilingBlock]:
        """
        These blocks restrict the hallway height if is less than the room inner height.
        """
        story = self.stories[story_num]
        hallway_height_tiles = np.zeros((story.length, story.width), dtype=np.int8)
        for hallway in story.hallways:
            if hallway.height is not None:
                draw_line_in_grid(hallway.points, hallway_height_tiles, hallway.height, include_ends=False)

        room_tiles = np.zeros((story.length, story.width))
        for room in story.rooms:
            set_room_tiles_in_grid(room, room.tiles, room_tiles)
        no_room_mask = ~room_tiles.astype(np.bool_)

        # We don't want reduced height inside rooms or anything outside the building footprint
        valid_hallway_mask = (hallway_height_tiles.astype(np.bool_) & story.footprint & no_room_mask).astype(
            np.float64
        )
        hallway_height_tiles[valid_hallway_mask != 1] = 0
        unique_heights = np.unique(hallway_height_tiles[hallway_height_tiles != 0])

        blocks = []
        for height in unique_heights:
            height_marker = -1
            tiles_for_height = hallway_height_tiles.copy()
            tiles_for_height[tiles_for_height == height] = height_marker
            blocks.extend(
                [
                    CeilingBlock(block.x, FloatRange(DEFAULT_FLOOR_THICKNESS + height, story.outer_height), block.z)
                    for block in make_blocks_from_tiles(tiles_for_height, solid_tile_value=height_marker)
                ]
            )
        return blocks

    def _generate_terrain_blocks(self, story_num: int) -> Sequence[FloorBlock]:
        # "Terrain" for rooms and buildings works as follows:
        # 0 is ground level - a floor heightmap of all 0s equates to no floor blocks. Walls are built atop this level.
        # >=1 is positive terrain - blocks built inside the height of the room. A floor heightmap equating to wall
        #   height would end up in a non-existent room (floor touches ceiling).
        # <=-1 is chasms (gaps) downward into the floor, and only really make sense if you have some non-negative
        #   terrain. Walls blocks and positive terrain are all extended downwards to enclose the chasms.

        all_blocks: List[FloorBlock] = []
        story = self.stories[story_num]
        min_height = -story.floor_negative_depth

        ramp_cut_tiles = np.zeros_like(story.footprint, dtype=np.int_)
        for story_link in story.story_links:
            if story_link.top_story_id == story.num:
                set_landings = isinstance(story_link, Ladder)
                set_link_in_grid(
                    story_link, ramp_cut_tiles, set_bottom_landing=set_landings, set_top_landing=set_landings
                )
        ramp_cut_tiles = ramp_cut_tiles.astype(np.bool_)

        for room in story.rooms:
            # The floor heightmap gets overridden by cuts necessary in the floor to accommodate ramps leading up here
            tiles = room.floor_heightmap.copy()
            room_cut_tiles_array = ramp_cut_tiles[
                room.position.z : room.position.z + room.length, room.position.x : room.position.x + room.width
            ]
            room_cut_tiles = list(zip(*[list(a) for a in room_cut_tiles_array.nonzero()]))
            for z, x in room_cut_tiles:
                tiles[z, x] = min_height

            unique_heights = np.unique(tiles)
            for height in unique_heights:
                if height == min_height:
                    continue
                for is_climbable in [True, False]:
                    tiles_for_height_and_type = tiles.copy()
                    tiles_for_height_and_type[
                        (tiles_for_height_and_type == height) & (room.climbable_mask == is_climbable)
                    ] = np.inf
                    blocks = make_blocks_from_tiles(tiles_for_height_and_type, solid_tile_value=np.inf)
                    for block in blocks:
                        offset_block = FloorBlock(
                            x=FloatRange(block.x.min_ge + room.position.x, block.x.max_lt + room.position.x),
                            y=FloatRange(min_height, height),
                            z=FloatRange(block.z.min_ge + room.position.z, block.z.max_lt + room.position.z),
                            is_climbable=is_climbable,
                        )
                        # Yes, there *can* be windows in terrain blocks (for separating walls inside a room for example)
                        cut_blocks = self._cut_windows(offset_block, story.windows)
                        all_blocks.extend(cut_blocks)
        return all_blocks

    def _cut_windows(self, parent_block: LevelBlockType, windows: List[Window]) -> Sequence[LevelBlockType]:
        window_blocks = []
        # TODO: improve efficiency - we shouldn't need to do a O(N^2) loop with these
        for window in windows:
            window_block_in_story_space = WindowBlock.make(window.position, window.size)
            if (
                window_block_in_story_space.x.overlap(parent_block.x)
                and window_block_in_story_space.y.overlap(parent_block.y)
                and window_block_in_story_space.z.overlap(parent_block.z)
            ):
                local_coords = global_to_local_coords(
                    window.position,
                    np.array([parent_block.x.min_ge, parent_block.y.min_ge, parent_block.z.min_ge]),
                )
                window_block_in_parent_block_space = WindowBlock.make(local_coords, window.size)
                window_blocks.append(window_block_in_parent_block_space)
        if len(window_blocks) > 0:
            return self._cut_blocks(parent_block, window_blocks)
        else:
            return [parent_block]

    def _cut_blocks(
        self, parent_block: LevelBlockType, child_blocks: Sequence[LevelBlock]
    ) -> Sequence[LevelBlockType]:
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
                raise ValueError("Child blocks need to meet or exceed parent block size in at least one dimension")
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
        elif cut_axis == Axis.Z:
            first_axis, second_axis = Axis.X, Axis.Y
        else:
            raise SwitchError(f"Unset / unknown cut axis: {cut_axis}")

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

        cut_blocks = make_blocks_from_tiles(grid)
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
            resulting_block = attr.evolve(parent_block, x=IntRange(0, 0), y=IntRange(0, 0), z=IntRange(0, 0))
            cut_axis_range: IntRange = getattr(parent_block, cut_axis.value)
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
            offset_block_range = IntRange(
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
            offset_block_range = IntRange(
                second_axis_range.min_ge / mult + parent_second_axis_offset,
                second_axis_range.max_lt / mult + end_offset,
            )
            setattr(resulting_block, second_axis.value, offset_block_range)
            resulting_blocks.append(resulting_block)
        return resulting_blocks

    def _generate_ceiling_blocks(self, story_num: int) -> Sequence[CeilingBlock]:
        story = self.stories[story_num]
        footprint = inset_borders(story.footprint).astype(np.int8)
        for story_link in story.story_links:
            if story_link.bottom_story_id == story_num:
                set_top_landing = isinstance(story_link, Ladder)
                set_link_in_grid(story_link, footprint, set_bottom_landing=True, set_top_landing=set_top_landing)

        return [
            CeilingBlock(
                x=block.x, y=FloatRange(story.outer_height - CEILING_THICKNESS, story.outer_height), z=block.z
            )
            for block in make_blocks_from_tiles(footprint, solid_tile_value=1)
        ]

    def _generate_window_blocks(self, story_num: int) -> Sequence[WindowBlock]:
        level_blocks: List[WindowBlock] = []
        for window in self.stories[story_num].windows:
            level_blocks.append(WindowBlock.make(window.position, window.size, is_collider=not window.is_passable))
        return level_blocks

    def plot(self) -> None:
        import seaborn as sns

        vmin = min(tile_symbol.value for tile_symbol in TileIdentity)
        colors = {
            "gray": TileIdentity.FULL.value,
            "lightgreen": TileIdentity.ROOM.value,
            "blue": TileIdentity.LINK.value,
            "darkblue": TileIdentity.LINK_BOTTOM_LANDING.value,
            "lightblue": TileIdentity.LINK_TOP_LANDING.value,
            "lightgray": TileIdentity.HALLWAY.value,
            "#eee": TileIdentity.VOID.value,
        }
        cmap = sorted(colors, key=colors.__getitem__)

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

    def rebuild_rotated(self, yaw_degrees: int) -> "Building":
        if yaw_degrees == -180:
            yaw_degrees = 180
        assert yaw_degrees in {0, 90, -90, 180}
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

                new_position = rotate_position(room.position, story.width, story.length, yaw_degrees, tile_like=True)
                # Switch from rotated point to new top-left point
                if yaw_degrees == 90:
                    new_position.x -= room.length - 1
                elif yaw_degrees == -90:
                    new_position.z -= room.width - 1
                elif yaw_degrees == 180:
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
                new_points = tuple(
                    [rotate_position(point, story.width, story.length, yaw_degrees) for point in hallway.points]
                )
                new_hallways.append(attr.evolve(hallway, points=new_points))

            new_story_links_for_story = [
                story_link
                for story_link in new_story_links
                if story.num in {story_link.bottom_story_id, story_link.top_story_id}
            ]

            new_windows = []
            for window in story.windows:
                position_2d = BuildingTile(window.position[0], window.position[2])
                new_position_2d = rotate_position(position_2d, story.width, story.length, yaw_degrees, tile_like=False)
                new_position_3d = np.array([new_position_2d.x, window.position[1], new_position_2d.z])
                if yaw_degrees in {90, -90}:
                    new_size = np.array([window.size[2], window.size[1], window.size[0]])
                else:
                    new_size = window.size.copy()
                new_windows.append(attr.evolve(window, position=new_position_3d, size=new_size))

            new_entrances = []
            for entrance in story.entrances:
                new_angle = (entrance.azimuth.angle_from_positive_x + yaw_degrees) % 360
                new_azimuth = only([a for a in list(Azimuth) if a.angle_from_positive_x % 360 == new_angle])
                new_entrance_points = tuple(
                    [rotate_position(point, story.width, story.length, yaw_degrees) for point in entrance.points]
                )
                new_entrances.append(
                    attr.evolve(
                        entrance,
                        azimuth=new_azimuth,
                        points=new_entrance_points,
                    )
                )

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
                    entrances=new_entrances,
                )
            )
        return attr.evolve(self, stories=rotated_stories, story_links=new_story_links)


class BuildingNavGraph(Graph):
    def __init__(self, building: Building, excluded_stories: Tuple[int, ...] = tuple()) -> None:
        """
        Represents the building with rooms as nodes and hallways + story links as edges.
        Does not check for room traversability (there may be blocked hallways / links or impassable terrain) - use
        StoryNavGraph for these.
        """
        super(BuildingNavGraph, self).__init__()
        self.excluded_stories = excluded_stories
        self._add_story_edges(building)
        self._add_story_link_edges(building)

    def _add_story_edges(self, building: Building) -> None:
        y_offset = 0.0
        for story in building.stories:
            if story.num in self.excluded_stories:
                continue

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

    def _add_story_link_edges(self, building: Building) -> None:
        for link in building.story_links:
            if link.bottom_story_id in self.excluded_stories or link.top_story_id in self.excluded_stories:
                continue
            assert isinstance(link, (Ramp, Ladder))
            from_story = building.stories[link.bottom_story_id]
            from_room = from_story.rooms[link.bottom_room_id]
            to_story = building.stories[link.top_story_id]
            to_room = to_story.rooms[link.top_room_id]
            from_room_node = self.get_room_node(from_story, from_room)
            to_room_node = self.get_room_node(to_story, to_room)
            distance_to_landing = euclidean_distance(from_room.center, link.bottom_position)
            distance_from_landing = euclidean_distance(link.top_position, to_room.center)
            total_length = link.get_link_length(from_story, to_story) + distance_to_landing + distance_from_landing
            self.add_edge(from_room_node, to_room_node, distance=total_length)

    def get_nearest_node(self, point: Point3DNP) -> str:
        positions: Dict[str, Tuple[int, int]] = nx.get_node_attributes(self, "position")
        kd_tree = KDTree(list(positions.values()))
        distance, position_idx = kd_tree.query(point)
        return cast(str, list(positions.keys())[position_idx])

    def get_room_node(self, story: Story, room: Room) -> str:
        return f"{story.num}_R{room.id}"

    def get_stories_and_rooms_by_node_id(self, stories: List[Story]) -> Dict[str, Tuple[Story, Room]]:
        stories_and_rooms_by_node_id = {}
        for story in stories:
            for room in story.rooms:
                stories_and_rooms_by_node_id[self.get_room_node(story, room)] = (story, room)
        return stories_and_rooms_by_node_id

    def plot(self, origin: Point3DNP = np.array([0, 0, 0]), ax: Optional[MplAxes] = None):
        plt.figure(figsize=(25, 25))
        positions = nx.get_node_attributes(self, "position")

        def translate_and_swap_axes(p: Point3DNP) -> Point3DNP:
            # This swap ensures the plot matches the building tile plot
            # to avoid axis swap, we need mpl 3.6.0+ to change up-axis: https://stackoverflow.com/a/56457693/5814943
            x, y, z = p
            x += origin[0]
            y += origin[1]
            z += origin[2]
            return np.array([z, x, y])

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
            middle = midpoint(edge_coordinates[0], edge_coordinates[1])
            ax.plot(*edge_coordinates.T, color="tab:gray")
            ax.text(*middle, f"{distance_by_edge[edge]:.1f}", color="darkgray", size=7)

        if not custom_ax:
            zs = [z for x, y, z in node_xyz]
            ax.set_zlim(min(zs), max(zs) + 0.1)
            ax.set_xlabel("z")
            ax.set_ylabel("x")
            ax.set_zlabel("y")
        return ax


def get_building_footprint_outline(building: Building, story_num: int = 0) -> Polygon:
    """
    Returns a 2D outline (y is always 0) of specified story footprint outline in world space.
    NB! Z is mapped to Y in the polygon.
    """
    outline = building.get_footprint_outline(story_num)
    x_offset, _y_offset, z_offset = building.position
    outline = affinity.rotate(outline, -building.yaw_degrees)
    outline = affinity.translate(outline, xoff=x_offset, yoff=z_offset)
    return outline


def get_building_positive_height_at_point(point_in_world_space: Point3DNP, building: Building) -> float:
    """
    Positive height refers to the exterior building height from it's y origin.
    Returns np.nan if point is not within building footprint.
    NOTE: both point and building must be in world space (rotated and positioned)
    """
    outline = get_building_footprint_outline(building)
    if IS_DEBUG_VIS:
        x, z = zip(*outline.exterior.coords)
        ax = plt.axes()
        ax.plot(x, z)
        ax.plot(point_in_world_space[0], point_in_world_space[2], "ro")
        ax.invert_yaxis()
        plt.gca().set_aspect("equal")
        plt.show()
    point = Point(point_in_world_space[0], point_in_world_space[2])
    if not outline.contains(point):
        # Note: points ON the edge are not included
        return np.nan
    else:
        building_centroid_2d = building.position[0], building.position[2]
        unrotated_point = affinity.rotate(point, -building.yaw_degrees, origin=building_centroid_2d)
        point_in_building_space = global_to_local_coords(
            np.array([unrotated_point.x, 0, unrotated_point.y]), building.position
        )
        return building.get_positive_height_at_point(point_in_building_space)


class BuildingTask(Enum):
    EXPLORE = "EXPLORE"
    OPEN = "OPEN"
    PUSH = "PUSH"
    STACK = "STACK"
    CLIMB = "CLIMB"
    JUMP = "JUMP"


def get_block_color(block: LevelBlock, aesthetics: BuildingAestheticsConfig, rand: np.random.Generator) -> RGBATuple:
    if isinstance(block, WallBlock):
        if block.is_interior:
            color = aesthetics.interior_wall_color
            brightness_jitter = rand.normal(0.0, aesthetics.interior_wall_brightness_jitter)
            alpha = color[-1]
            h, s, v = colorsys.rgb_to_hsv(*color[:-1])
            v = float(np.clip(v + brightness_jitter, 0, 1))
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb_jittered = tuple(channel + rand.normal(0.0, aesthetics.interior_wall_color_jitter) for channel in rgb)
            return (*rgb_jittered, alpha)  # type: ignore
        else:
            return aesthetics.exterior_color
    elif isinstance(block, FloorBlock):
        return aesthetics.floor_color
    elif isinstance(block, CeilingBlock):
        return aesthetics.ceiling_color
    elif isinstance(block, LadderBlock):
        return aesthetics.ladder_color
    else:
        raise SwitchError(f"Unknown block type {type(block)}")


def make_pretty_building_meshes(building: Building, blocks_by_story: BlocksByStory) -> Dict[str, Trimesh]:
    rand = np.random.default_rng(building.id)
    corners_by_story = defaultdict(list)
    wall_footprints_by_story = defaultdict(list)
    exterior_wall_blocks_by_story: DefaultDict[int, List[WallBlock]] = defaultdict(list)
    original_outlines_by_story = {}

    cornerless_building = building.rebuild_rotated(0)
    for story in cornerless_building.stories:
        outline = cornerless_building.get_footprint_outline(story.num)
        original_outlines_by_story[story.num] = outline

        new_footprint = story.footprint.copy()
        wall_footprints_by_story[story.num] = find_exterior_wall_footprints(new_footprint)
        corners = find_corners(new_footprint)
        corners_by_story[story.num] = corners
        for corner_position, corner_type, is_outside in corners:
            if is_outside:
                new_footprint[corner_position[0], corner_position[1]] = False
        story.footprint = new_footprint
        exterior_wall_blocks_by_story[story.num].extend(cornerless_building._generate_exterior_wall_blocks(story.num))

    mesh_datasets: DefaultDict[str, MeshData] = defaultdict(lambda: MeshData())

    x_offset = -building.width / 2
    z_offset = -building.length / 2
    tile_size = 1

    def add_mesh_data(parent_name: str, child_mesh: Trimesh) -> None:
        nonlocal mesh_datasets
        parent_mesh_data = mesh_datasets[parent_name]
        parent_mesh_data.vertices.extend(child_mesh.vertices)
        parent_mesh_data.faces.extend(child_mesh.faces + parent_mesh_data.index_offset)
        parent_mesh_data.face_normals.extend(child_mesh.face_normals)
        if child_mesh.visual:
            parent_mesh_data.face_colors.extend(child_mesh.visual.face_colors)
        else:
            parent_mesh_data.face_colors.extend([0.0, 0.0, 0.0] * len(child_mesh.faces))
        parent_mesh_data.index_offset += len(child_mesh.vertices)

    for story_num, blocks in blocks_by_story.items():
        story = building.stories[story_num]
        y_offset = building.get_story_y_offset(story_num)
        exterior_wall_blocks = exterior_wall_blocks_by_story[story.num]
        other_blocks = [block for block in blocks if not (isinstance(block, WallBlock) and not block.is_interior)]
        for block in [*exterior_wall_blocks, *other_blocks]:
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
            box.visual = make_color_visuals(box, get_block_color(block, building.aesthetics, rand))
            add_mesh_data(HIGH_POLY, box)
            if (isinstance(block, WallBlock) and not block.is_interior) or isinstance(block, CeilingBlock):
                add_mesh_data(LOW_POLY, box)

        for story_link in story.story_links:
            if story_link.bottom_story_id != story.num:
                continue
            story_link_level_blocks = story_link.get_level_blocks(
                building.stories[story_link.bottom_story_id], building.stories[story_link.top_story_id]
            )
            for level_block in story_link_level_blocks:
                if not level_block.is_visual:
                    continue

                block_width, block_height, block_length = level_block.size
                block_x, block_y, block_z = level_block.centroid
                trimesh_box = creation.box((block_width, block_height, block_length))
                position = np.array(
                    [
                        x_offset + block_x,
                        building.get_story_y_offset(story_link.bottom_story_id) + DEFAULT_FLOOR_THICKNESS + block_y,
                        z_offset + block_z,
                    ]
                )
                rotation_transform = homogeneous_transform_matrix(rotation=level_block.rotation)
                trimesh_box.apply_transform(rotation_transform)
                translation_transform = homogeneous_transform_matrix(position)
                trimesh_box.apply_transform(translation_transform)
                trimesh_box = unnormalize(trimesh_box)
                trimesh_box.visual = make_color_visuals(
                    trimesh_box, get_block_color(level_block, building.aesthetics, rand)
                )
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
