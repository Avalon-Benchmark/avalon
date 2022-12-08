from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import cast

import attr
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from scipy.spatial.transform import Rotation

from avalon.common.utils import to_immutable_array
from avalon.datagen.world_creation.constants import IDENTITY_BASIS
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_BASE_HEIGHT
from avalon.datagen.world_creation.entities.item import InstancedDynamicItem
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.entities.tools.weapons import LargeRock
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.entities.utils import facing_2d
from avalon.datagen.world_creation.entities.utils import get_random_ground_points
from avalon.datagen.world_creation.entities.utils import normalized
from avalon.datagen.world_creation.types import GodotScene
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import scale_basis
from avalon.datagen.world_creation.worlds.height_map import HeightMap

_FoodT = TypeVar("_FoodT", bound="Food")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Food(InstancedDynamicItem):
    mass: float = 1.0
    resource_file: str = "res://items/food.tscn"
    energy: float = 1.0
    plant_path: Optional[str] = None

    position: np.ndarray = attr.ib(
        default=np.array([0.0, 0.0, 0.0]),
        converter=to_immutable_array,
        eq=attr.cmp_using(eq=np.array_equal),  # type: ignore[attr-defined]
    )
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])
    is_grown_on_trees: bool = True
    is_found_on_ground: bool = True
    is_openable: bool = False
    count_distribution: Tuple[Tuple[float, ...], Tuple[float, ...]] = ((0.0, 1.0), (1.0, 1.7))

    @property
    def additional_offset(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def get_stem_joint(self, food_name: str) -> GDNode:
        basis = scale_basis(IDENTITY_BASIS, self.scale)
        point_five_over_parent = GDObject("Transform", *basis, 0, 0.5 * self.scale[1], 0)
        return GDNode(
            f"{food_name}_stem_joint",
            type="Generic6DOFJoint",
            properties={
                "transform": point_five_over_parent,
                "nodes/node_a": f"../../{food_name}",
                "nodes/node_b": f"../../{self.plant_path}",
                "linear_limit_x/upper_distance": 0.5,
                "linear_limit_y/upper_distance": 0.0,
                "linear_limit_z/upper_distance": 0.5,
            },
        )

    def attached_to(self: _FoodT, tree: "FoodTree") -> _FoodT:
        assert tree.entity_id > -1, f"Food must be attached_to the result of world.add_item({tree})"
        assert tree.is_food_on_tree, f"Food shouldn't be attached_too tree {tree} without is_food_on_tree=True"
        return attr.evolve(self, plant_path=tree.node_name)

    def get_node(self, scene: GodotScene) -> GDNode:
        food_node = super().get_node(scene)
        food_node.properties["energy"] = self.energy

        if self.plant_path is None:
            return food_node

        food_node.add_child(self.get_stem_joint(food_node.name))
        return food_node

    def get_count(self, rand: np.random.Generator) -> int:
        return round(float(np.interp(rand.uniform(), self.count_distribution[0], self.count_distribution[1])))

    def get_opened_version(self) -> "Food":
        if not self.is_openable:
            return self
        raise NotImplementedError("Openable food must implement get_opened_version")

    def get_tool_options(self) -> Tuple[float, Tuple[Type[Tool], ...]]:
        return 0, tuple()

    @property
    def is_tool_required(self) -> bool:
        return self.get_tool_options()[0] == 1

    @property
    def is_tool_useful(self) -> bool:
        return self.get_tool_options()[0] > 0

    def is_always_multiple(self) -> bool:
        return False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Banana(Food):
    resource_file: str = "res://items/food/banana.tscn"
    energy: float = 0.5

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, 0.3, 0.0])
        return super().additional_offset


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Apple(Food):
    resource_file: str = "res://items/food/apple.tscn"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Fig(Food):
    resource_file: str = "res://items/food/fig.tscn"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class OpenableFood(Food):
    is_openable: bool = True

    @property
    def is_open(self) -> bool:
        return not self.is_openable

    def get_opened_version(self: "OpenableFood") -> "OpenableFood":
        if self.is_open:
            return self
        opened: "OpenableFood"
        with self.mutable_clone() as opened:
            opened.resource_file = opened.resource_file.replace(".tscn", "_open.tscn")
            opened.is_openable = False
            return opened


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Orange(OpenableFood):
    resource_file: str = "res://items/food/orange.tscn"

    def get_tool_options(self):
        return 0.1, (Rock, Stick, LargeRock)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Avocado(OpenableFood):
    resource_file: str = "res://items/food/avocado.tscn"

    def get_tool_options(self):
        return 1, (Rock, LargeRock)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Coconut(OpenableFood):
    resource_file: str = "res://items/food/coconut.tscn"

    def get_tool_options(self):
        return 0.25, (Rock, Stick, LargeRock)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Honeycomb(Food):
    resource_file: str = "res://items/food/honeycomb.tscn"
    is_found_on_ground: bool = False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Cherry(Food):
    resource_file: str = "res://items/food/cherry.tscn"

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, -0.5, 0.0])
        return super().additional_offset

    def is_always_multiple(self) -> bool:
        return True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Mulberry(Food):
    resource_file: str = "res://items/food/mulberry.tscn"

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, 0.3, 0.0])
        return super().additional_offset


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Carrot(Food):
    is_grown_on_trees: bool = False
    resource_file: str = "res://items/food/carrot.tscn"

    def get_offset(self) -> float:
        return 0.0


FOODS: Tuple[Food, ...] = (
    Apple(),
    Banana(),
    Cherry(),
    Honeycomb(),
    Mulberry(),
    Fig(),
    Orange(),
    Avocado(),
    Coconut(),
    Carrot(),
)
NON_TREE_FOODS = [x for x in FOODS if not x.is_grown_on_trees]
CANONICAL_FOOD = FOODS[0]
CANONICAL_FOOD_CLASS = FOODS[0].__class__
CANONICAL_FOOD_HEIGHT = CANONICAL_FOOD.get_offset()
GetHeightAt = Callable[[Tuple[float, float]], float]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FoodTreeBase(InstancedDynamicItem):
    resource_file: str = "res://scenery/tree_base.tscn"
    rotation: Rotation = Rotation.identity()

    def get_offset(self) -> float:
        return -0.25


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FoodTree(InstancedDynamicItem):
    ROTATION_TO_SUITABLE_OFFSET: ClassVar[Dict[Tuple[float], Tuple[float, float]]] = {
        (0,): (1.25, CANONICAL_FOOD_HEIGHT_ON_TREE),
        # TODO add mechanism for multiple rotations
        # (45,): (0.875, 3.25),
    }

    @property
    def food_offset(self) -> Tuple[float, float]:
        return self.ROTATION_TO_SUITABLE_OFFSET[(0,)]

    def _relative_position(self, offset: Point3DNP) -> Point3DNP:
        return cast(Point3DNP, self.rotation.apply(offset) * self.scale)

    def get_food_height(self, food: _FoodT) -> float:
        xz, y = self.food_offset
        return cast(float, y * self.scale[1])

    resource_file: str = "res://scenery/trees/fruit_tree_normal.tscn"
    is_tool_helpful: bool = True
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])
    rotation: Rotation = Rotation.identity()

    is_food_on_tree: bool = False

    @property
    def height(self):
        scale_y = self.scale[1]
        return FOOD_TREE_BASE_HEIGHT * scale_y

    @staticmethod
    def build(tree_height: float, is_food_on_tree: bool = False) -> "FoodTree":
        scale_factor = tree_height / FOOD_TREE_BASE_HEIGHT
        return FoodTree(
            scale=scale_factor * np.array([1.0, 1.0, 1.0]),
            position=np.array([0.0, 0.0, 0.0]),
            is_food_on_tree=is_food_on_tree,
        )

    def _resolve_trunk_placement(
        self, spawn: Point3DNP, primary_food: Point3DNP, get_height_at: GetHeightAt
    ) -> Tuple[Point3DNP, float]:
        goal_vector: Point3DNP = primary_food - spawn
        goal_vector[1] = 0
        goal_vector = normalized(goal_vector)
        if self.is_food_on_tree:
            xz_offset, y_offset = self.food_offset
            away_from_spawn_and_down_from_food = goal_vector * xz_offset
            away_from_spawn_and_down_from_food[1] = -y_offset
            away_from_spawn_and_down_from_food *= self.scale
            position = primary_food + away_from_spawn_and_down_from_food
            return position, get_height_at((position[0], position[2]))

        # TODO: could use a bit more variety in how the tree is place relative to the food
        x, _, z = primary_food + 2.0 * goal_vector
        ground_level = get_height_at((x, z))
        ground_level_behind_food = np.array([x, ground_level, z])
        return ground_level_behind_food, ground_level

    def place(
        self, spawn: Point3DNP, primary_food: Point3DNP, get_height_at: GetHeightAt
    ) -> Tuple["FoodTree", Optional[FoodTreeBase]]:
        trunk_placement, ground_height = self._resolve_trunk_placement(spawn, primary_food, get_height_at)
        rotation = facing_2d(trunk_placement, spawn)
        root_affordance = 0.25 * self.scale[1]
        is_boulder_needed = self.is_food_on_tree and ground_height < (trunk_placement[1] - root_affordance)
        tree = attr.evolve(self, position=trunk_placement, rotation=rotation)
        if not is_boulder_needed:
            return (tree, None)

        return tree, FoodTreeBase(position=trunk_placement, rotation=rotation)

    def get_food_locations(
        self,
        rand: np.random.Generator,
        center: Point3DNP,
        count: int,
        map: HeightMap,
        min_radius: float,
        max_radius: float,
        offset: float,
        island_mask: MapBoolNP,
    ) -> Point3DListNP:
        # we originally intended that some food could sometimes be at different places on the tree and some on the
        # ground, but didn't get to finish this feature before the deadline
        ground_count = count
        ground_points = get_random_ground_points(
            rand, center, ground_count, map, min_radius, max_radius, offset, island_mask
        )
        tree_point_count = count - ground_count
        if tree_point_count == 0:
            return ground_points
        random_heights = rand.uniform(0.5, 0.9, size=tree_point_count) * self.height
        zeros = np.zeros_like(random_heights)
        height_vectors = np.stack([zeros, random_heights, zeros])
        tree_points = self.position + height_vectors
        return cast(Point3DListNP, tree_points)

    def get_offset(self) -> float:
        return 0.0

    def get_node(self, scene: GodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["scale"] = GDObject("Vector3", *self.scale)
        return node
