from itertools import product
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import attr
import godot_parser
import numpy as np
from godot_parser import ExtResource
from godot_parser import Node as GDNode
from nptyping import Float
from nptyping import NDArray
from nptyping import Shape
from scipy.spatial.transform import Rotation

from avalon.datagen.godot_base_types import FloatRange
from avalon.datagen.godot_base_types import IntRange
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from avalon.datagen.world_creation.types import GDLinkedSection
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import make_transform

RangeType = TypeVar("RangeType")
LBT = TypeVar("LBT", bound="LevelBlock")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Block(Generic[RangeType]):
    x: RangeType
    y: RangeType
    z: RangeType


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LevelBlock(Block):
    """
    A representation of a box-like shape to be used in an Avalon level context - these can then later be converted to
    CubeMeshes, BoxShapes, and any other rectangle prisms in Godot.

    Blocks are rotated around their origin (centroid).
    """

    rotation: Rotation = Rotation.identity()
    is_visual: bool = True
    is_collider: bool = True
    is_climbable: bool = False

    def __attrs_post_init__(self) -> None:
        assert self.is_visual or self.is_collider, "LevelBlocks must be either colliders, visual or both"

    @classmethod
    def make(
        cls: Type[LBT], centroid: Point3DNP, size: Point3DNP, rotation: Rotation = Rotation.identity(), **kwargs: Any
    ) -> LBT:
        x = FloatRange(centroid[0] - size[0] / 2, centroid[0] + size[0] / 2)
        y = FloatRange(centroid[1] - size[1] / 2, centroid[1] + size[1] / 2)
        z = FloatRange(centroid[2] - size[2] / 2, centroid[2] + size[2] / 2)
        rotation = rotation
        return cls(x, y, z, rotation, **kwargs)

    @property
    def type_name(self) -> str:
        return self.__class__.__name__.lower().removesuffix("block")

    @property
    def centroid(self) -> Tuple[float, float, float]:
        return self.x.midpoint, self.y.midpoint, self.z.midpoint

    @property
    def size(self) -> Tuple[float, float, float]:
        return self.x.size, self.y.size, self.z.size


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloorBlock(LevelBlock):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CeilingBlock(LevelBlock):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WindowBlock(LevelBlock):
    is_visual: bool = False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WallBlock(LevelBlock):
    is_interior: bool = True

    @property
    def type_name(self) -> str:
        return f"{('interior' if self.is_interior else 'exterior')}_{super().type_name}"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LadderBlock(LevelBlock):
    is_climbable: bool = True


def make_blocks_from_tiles(grid: np.ndarray, solid_tile_value: Any = 0) -> List[Block]:
    """
    Divide a grid of `grid` into as few solid `Block`s as possible. Blocks represent rectangular regions of the grid
    that all consist of solid tiles. Tiles are considered to be solid if their value equals `solid_tile_value`.

    The algorithm iterates tile-by-tile to find a solid tile that is not already part of a block. It then starts a new
    block. The block formation goes greedily left-to-right to find a row of solid tiles and then extends the row
    downward as far as possible.
    """

    blocks = []
    covered_tiles: Set[Tuple[int, int]] = set()
    for (z, x), _tile in np.ndenumerate(grid):
        if (z, x) in covered_tiles or grid[z][x] != solid_tile_value:
            continue
        block = _extract_next_block(grid, z, x, covered_tiles, solid_tile_value)
        if block is not None:
            blocks.append(block)
    return blocks


def _extract_next_block(
    grid: np.ndarray, z_min_ge: int, x_min_ge: int, covered_tiles: Set[Tuple[int, int]], solid_tile_value: int = 0
) -> Block:
    length, width = grid.shape

    # Extend along x
    xi = x_min_ge
    for xi in range(x_min_ge, width + 1):
        if xi >= width or grid[z_min_ge][xi] != solid_tile_value or (z_min_ge, xi) in covered_tiles:
            # This tile is outside grid, not solid or already covered, so this is the furthest we go along x
            break
    x_max_lt = xi

    # Extend along z
    zi = z_min_ge
    for zi in range(z_min_ge, length + 1):
        if zi >= length:
            # We've reached the end of the grid
            break

        candidate_tiles = grid[zi : zi + 1, x_min_ge:x_max_lt]
        if (candidate_tiles == solid_tile_value).all():
            candidate_coords = set(product(range(zi, zi + 1), range(x_min_ge, x_max_lt)))
            has_dupe_tiles = len(candidate_coords.intersection(covered_tiles)) > 0
            if has_dupe_tiles:
                break
        else:
            break
    z_max_lt = zi

    # Make sure we don't create overlapping blocks by marking the tiles as covered
    covered_tiles.update(product(range(z_min_ge, z_max_lt), range(x_min_ge, x_max_lt)))

    return Block(
        x=IntRange(x_min_ge, x_max_lt),
        y=IntRange(DEFAULT_FLOOR_THICKNESS, DEFAULT_STORY_HEIGHT + 1),
        z=IntRange(z_min_ge, z_max_lt),
    )


def new_make_block_nodes(
    scene: GodotScene,
    position: NDArray[Shape["3,0"], Float],
    size: NDArray[Shape["3,0"], Float],
    rotation: NDArray[Shape["12,0"], Float] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    make_parent: bool = True,
    parent_type: str = "StaticBody",
    parent_name: str = "StaticBody",
    parent_script: Optional[ExtResource] = None,
    parent_extra_props: Optional[Dict[str, Any]] = None,
    make_mesh: bool = True,
    mesh_name: str = "mesh",
    mesh_material: Optional[Union[str, GDLinkedSection]] = None,
    mesh_script: Optional[ExtResource] = None,
    make_collision_shape: bool = True,
    collision_shape_name: str = "collision_shape",
    collision_shape_script: Optional[ExtResource] = None,
    collision_shape_disabled: bool = False,
) -> List[GDNode]:
    nodes = []
    width, height, length = size

    transform = make_transform(position=position, rotation=rotation)

    if make_mesh:
        assert mesh_material is not None
        if isinstance(mesh_material, str):
            mesh_material = scene.add_ext_resource(mesh_material, "Material")
        mesh_props = {
            "mesh": scene.add_sub_resource("CubeMesh", size=godot_parser.Vector3(width, height, length)).reference,
            "material/0": mesh_material.reference,
        }
        if not make_parent:
            mesh_props["transform"] = transform
        if mesh_script:
            mesh_props["script"] = mesh_script
        mesh_instance = GDNode(mesh_name, type="MeshInstance", properties=mesh_props)
        nodes.append(mesh_instance)

    if make_collision_shape:
        collision_shape_props = {}
        if not make_parent:
            collision_shape_props["transform"] = transform

        collision_shape_props["shape"] = scene.add_sub_resource(
            "BoxShape",
            extents=godot_parser.Vector3(width / 2, height / 2, length / 2),
        ).reference

        if collision_shape_script:
            collision_shape_props["script"] = collision_shape_script
        if collision_shape_disabled:
            collision_shape_props["disabled"] = True
        collision_shape = GDNode(collision_shape_name, type="CollisionShape", properties=collision_shape_props)
        if make_collision_shape:
            nodes.append(collision_shape)

    if make_parent:
        parent_props = {}
        if parent_script:
            parent_props["script"] = parent_script
        parent_props.update({} if parent_extra_props is None else parent_extra_props)
        parent_props.update(transform=transform)
        parent_node = GDNode(parent_name, parent_type, properties=parent_props)
        for node in nodes:
            parent_node.add_child(node)
        nodes = [parent_node]

    return nodes


BlocksByStory = Dict[int, List[LevelBlock]]
