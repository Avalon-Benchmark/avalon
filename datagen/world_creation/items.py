import math
from enum import Enum
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Final
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import attr
import godot_parser
import numpy as np
from godot_parser import Color
from godot_parser import ExtResource
from godot_parser import GDObject
from godot_parser import Node as GDNode
from godot_parser import NodePath
from godot_parser import Vector3
from scipy.spatial.transform import Rotation

from common.utils import only
from contrib.serialization import Serializable
from datagen.world_creation.geometry import Axis
from datagen.world_creation.godot_utils import IDENTITY_BASIS
from datagen.world_creation.godot_utils import make_transform
from datagen.world_creation.godot_utils import scale_basis
from datagen.world_creation.heightmap import HeightMap
from datagen.world_creation.heightmap import MapBoolNP
from datagen.world_creation.heightmap import Point3DListNP
from datagen.world_creation.heightmap import Point3DNP
from datagen.world_creation.indoor.constants import DEFAULT_STORY_HEIGHT
from datagen.world_creation.new_godot_scene import ImprovedGodotScene
from datagen.world_creation.utils import ImpossibleWorldError


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Entity(Serializable):
    entity_id: int
    is_dynamic: bool
    position: np.ndarray

    @property
    def point2d(self) -> Tuple[float, float]:
        return (self.position[0], self.position[2])

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        raise NotImplementedError()


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Item(Entity):
    pass


def make_block_child_nodes(
    scene: ImprovedGodotScene,
    material: str,
    position: Vector3,
    size: Vector3,
    make_mesh=True,
    make_collision_shape=True,
    mesh_name="mesh",
    collision_shape_name="collision_shape",
    mesh_script: Optional[ExtResource] = None,
    collision_shape_script: Optional[ExtResource] = None,
    collision_shape_disabled: bool = False,
):
    # todo: make material optional
    nodes = []
    width, height, length = size

    if make_mesh:
        material = scene.add_ext_resource(material, "Material")
        mesh_props = {
            "transform": make_transform(position=position),
            "mesh": scene.add_sub_resource("CubeMesh", size=godot_parser.Vector3(width, height, length)).reference,
            "material/0": material.reference,
        }
        if mesh_script:
            mesh_props["script"] = mesh_script
        mesh_instance = GDNode(mesh_name, type="MeshInstance", properties=mesh_props)
        nodes.append(mesh_instance)

    if make_collision_shape:
        collision_shape_props = {
            "transform": make_transform(position=position),
            "shape": scene.add_sub_resource(
                "BoxShape",
                extents=godot_parser.Vector3(width / 2, height / 2, length / 2),
            ).reference,
        }
        if collision_shape_script:
            collision_shape_props["script"] = collision_shape_script
        if collision_shape_disabled:
            collision_shape_props["disabled"] = True
        collision_shape = GDNode(collision_shape_name, type="CollisionShape", properties=collision_shape_props)
        if make_collision_shape:
            nodes.append(collision_shape)

    return nodes


def make_block_node(
    scene: ImprovedGodotScene,
    name: str,
    material: str,
    size: Vector3,
    transform: GDObject,
    body_type: str = "RigidBody",
    scaling_factor: float = 1.0,
    extra_properties: Optional[Dict] = None,
):
    if extra_properties is None:
        extra_properties = {}

    body = GDNode(name, type=body_type, properties={"transform": transform, **extra_properties})
    size = Vector3(size.x * scaling_factor, size.y * scaling_factor, size.z * scaling_factor)
    block_children_nodes = make_block_child_nodes(scene, material, position=Vector3(0, 0, 0), size=size)
    for node in block_children_nodes:
        body.add_child(node)
    return body


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class InstancedItem(Item):
    resource_file: str
    rotation: Union[np.ndarray, Rotation] = IDENTITY_BASIS
    safe_scale: Optional[np.ndarray] = None
    base_color: Optional[str] = None

    @property
    def basis(self) -> np.ndarray:
        if isinstance(self.rotation, Rotation):
            return self.rotation.as_matrix().flatten()
        return self.rotation

    @property
    def node_name(self):
        assert self.entity_id != -1, (
            f"Attempt to access {self}.node_name before entity_id was set. "
            f"This is likely a bug due to NodePath-related code being called before world.add_item({self})"
        )
        item_name = self.resource_file.split("/")[-1].replace(".tscn", "")
        return f"{item_name}__{self.entity_id}"

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        resource = scene.add_ext_resource(self.resource_file, "PackedScene")
        properties = {
            "entity_id": self.entity_id,
            "transform": GDObject("Transform", *self.basis, *self.position),
        }
        if self.safe_scale is not None:
            properties["safe_scale"] = godot_parser.Vector3(*self.safe_scale)
        if self.base_color is not None:
            properties["base_color"] = self.base_color
        return GDNode(
            self.node_name,
            instance=resource.reference.id,
            properties=properties,
        )


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class InstancedDynamicItem(InstancedItem):
    is_dynamic: bool = True
    mass: float = 1.0

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["mass"] = self.mass
        return node

    @staticmethod
    def get_offset() -> float:
        return 0.5


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Scenery(InstancedItem):
    is_dynamic: bool = False
    scale: np.ndarray = np.array([1, 1, 1])
    entity_id: int = -1


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Tool(InstancedDynamicItem):
    position: np.ndarray = np.array([0.0, 0.0, 0.0])
    entity_id: int = 0
    solution_mask: Optional[MapBoolNP] = None


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Placeholder(Tool):
    position: np.ndarray = np.array([0.0, 0.0, 0.0])
    entity_id: int = 0
    resource_file: str = "res://items/not_real.tscn"
    offset: float = 0.0

    def get_offset(self) -> float:
        return self.offset

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        raise Exception("Not implemented for a reason--should never be called!")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ColoredSphere(Tool):
    resource_file: str = "res://items/sphere_marker.tscn"
    color: str = "#0000FF"
    scale: float = 2.0

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["color"] = self.color
        node.properties["scale"] = godot_parser.Vector3(self.scale, self.scale, self.scale)
        return node


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Stone(Tool):
    size: float = 1.0
    mass: float = 10.0
    resource_file: str = "res://items/stone.tscn"
    script_file: str = "res://items/stone.gd"
    default_mesh_extents: float = 1.0

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        mesh_node = super().get_node(scene)
        item_name = mesh_node.name
        mesh_node.name = "mesh"

        scale_factor = self.size / self.default_mesh_extents
        mesh_node.properties["transform"] = make_transform(
            scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), [scale_factor] * 3)
        )

        half_size = self.size / 2
        collision_shape_node = GDNode(
            "collision_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape",
                    extents=godot_parser.Vector3(half_size, half_size, half_size),
                ).reference,
            },
        )
        item_node = GDNode(
            item_name,
            type="RigidBody",
            properties={
                "transform": GDObject("Transform", *self.basis, *self.position),
                "script": scene.add_ext_resource(self.script_file, "Script").reference,
                "entity_id": self.entity_id,
                "mass": self.mass,
            },
        )
        item_node.add_child(collision_shape_node)
        item_node.add_child(mesh_node)
        return item_node

    def get_offset(self) -> float:
        return (self.size * self.default_mesh_extents) / 2.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Boulder(Stone):
    mass: float = 60.0
    # TODO: put back when we get the new asset
    resource_file: str = "res://items/stone.tscn"
    # resource_file: str = "res://items/boulder.tscn"
    script_file: str = "res://items/boulder.gd"
    size: float = 1.4


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Log(Tool):
    mass: float = 60.0
    resource_file: str = "res://items/log.tscn"

    def get_offset(self) -> float:
        return 0.7  # need to scale this if we scale size


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Weapon(Tool):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Stick(Weapon):
    WEAPON_VALUE: ClassVar = 1.0
    mass: float = 5.0
    resource_file: str = "res://items/stick.tscn"

    def get_offset(self) -> float:
        return 0.35


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LargeStick(Stick):
    WEAPON_VALUE: ClassVar = 1.0
    resource_file: str = "res://items/large_stick.tscn"

    def get_offset(self) -> float:
        return 0.35


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Rock(Weapon):
    WEAPON_VALUE: ClassVar = 0.5

    mass: float = 1.0
    resource_file: str = "res://items/rock.tscn"

    def get_offset(self) -> float:
        return 0.25


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class LargeRock(Rock):
    WEAPON_VALUE: ClassVar = 0.5
    resource_file: str = "res://items/large_rock.tscn"

    def get_offset(self) -> float:
        return 0.5


LARGEST_ANIMAL_SIZE: Final = 3.125


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Animal(InstancedDynamicItem):
    is_grounded: bool = False
    is_flying: bool = False
    is_able_to_climb: bool = False

    def get_offset(self):
        # TODO  fix height solution stuff
        # return get_box_collision_extents(self.resource_file).y
        return LARGEST_ANIMAL_SIZE / 2


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Prey(Animal):
    RESOURCES: ClassVar = "res://entities/animals/prey"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Predator(Animal):
    RESOURCES: ClassVar = "res://entities/animals/predators"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Frog(Prey):
    resource_file: str = f"{Prey.RESOURCES}/frog.tscn"
    is_grounded: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Turtle(Prey):
    resource_file: str = f"{Prey.RESOURCES}/turtle.tscn"
    is_grounded: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Mouse(Prey):
    resource_file: str = f"{Prey.RESOURCES}/mouse.tscn"
    is_grounded: bool = True
    is_able_to_climb: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Rabbit(Prey):
    resource_file: str = f"{Prey.RESOURCES}/rabbit.tscn"
    is_grounded: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Pigeon(Prey):
    resource_file: str = f"{Prey.RESOURCES}/pigeon.tscn"
    is_flying: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Squirrel(Prey):
    resource_file: str = f"{Prey.RESOURCES}/squirrel.tscn"
    is_grounded: bool = True
    is_able_to_climb: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Crow(Prey):
    resource_file: str = f"{Prey.RESOURCES}/crow.tscn"
    is_flying: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Deer(Prey):
    resource_file: str = f"{Prey.RESOURCES}/deer.tscn"
    is_grounded: bool = True
    mass: float = 3.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Bee(Predator):
    resource_file: str = f"{Predator.RESOURCES}/bee.tscn"
    is_flying: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Snake(Predator):
    resource_file: str = f"{Predator.RESOURCES}/snake.tscn"
    is_grounded: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Hawk(Predator):
    resource_file: str = f"{Predator.RESOURCES}/hawk.tscn"
    is_flying: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Hippo(Predator):
    resource_file: str = f"{Predator.RESOURCES}/hippo.tscn"
    is_grounded: bool = True
    mass: float = 10.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Alligator(Predator):
    resource_file: str = f"{Predator.RESOURCES}/alligator.tscn"
    is_grounded: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Eagle(Predator):
    resource_file: str = f"{Predator.RESOURCES}/eagle.tscn"
    is_flying: bool = True


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Wolf(Predator):
    resource_file: str = f"{Predator.RESOURCES}/wolf.tscn"
    is_grounded: bool = True
    mass: float = 3.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Jaguar(Predator):
    resource_file: str = f"{Predator.RESOURCES}/jaguar.tscn"
    is_grounded: bool = True
    is_able_to_climb: bool = True
    mass: float = 3.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Bear(Predator):
    resource_file: str = f"{Predator.RESOURCES}/bear.tscn"
    is_grounded: bool = True
    is_able_to_climb: bool = True
    mass: float = 10.0


STICK_LENGTH: Final = 4.0

ALL_PREY_CLASSES: Final[List[Type[Prey]]] = [
    Frog,
    Turtle,
    Mouse,
    Rabbit,
    Pigeon,
    Squirrel,
    Crow,
    Deer,
]

ALL_PREDATOR_CLASSES: Final[List[Type[Predator]]] = [
    Bee,
    Snake,
    Hawk,
    Hippo,
    Alligator,
    Eagle,
    Wolf,
    Jaguar,
    Bear,
]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SpawnPoint(Item):
    pitch: float
    yaw: float
    is_dynamic: bool = False
    is_visibility_required: bool = True

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        spawn_yaw_rotation = Rotation.from_euler("y", self.yaw, degrees=True)
        spawn_pitch_rotation = Rotation.from_euler("x", self.pitch, degrees=True)
        spawn_rotation = (spawn_yaw_rotation * spawn_pitch_rotation).as_matrix().flatten()
        spawn_transform = GDObject("Transform", *spawn_rotation, *self.position)
        return GDNode("SpawnPoint", type="Spatial", properties={"transform": spawn_transform})


_FoodT = TypeVar("_FoodT", bound="Food")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Food(InstancedDynamicItem):
    mass: float = 1.0
    resource_file: str = "res://items/food.tscn"
    energy: float = 1.0
    plant_path: Optional[str] = None

    position: np.ndarray = np.array([0, 0, 0])
    entity_id: int = 0
    scale: np.ndarray = np.array([1, 1, 1])
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

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        food_node = super().get_node(scene)
        food_node.properties["energy"] = self.energy

        if self.plant_path is None:
            return food_node

        food_node.add_child(self.get_stem_joint(food_node.name))
        return food_node

    def get_count(self, rand: np.random.Generator) -> int:
        return round(np.interp(rand.uniform(), self.count_distribution[0], self.count_distribution[1]))

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


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Banana(Food):
    resource_file: str = "res://items/food/banana.tscn"
    energy: float = 0.5

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, 0.3, 0.0])
        return super().additional_offset

    # def attached_to(self: _FoodT, tree: "FoodTree") -> _FoodT:
    #     banana = super().attached_to(tree)
    #     rotation = Rotation.from_euler("z", 180, degrees=True)
    #     return attr.evolve(banana, rotation=rotation)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Apple(Food):
    resource_file: str = "res://items/food/apple.tscn"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Fig(Food):
    resource_file: str = "res://items/food/fig.tscn"

    @property
    def additional_offset(self) -> np.ndarray:
        if self.plant_path is not None:
            return np.array([0.0, 0.3, 0.0])
        return super().additional_offset


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class OpenableFood(Food):
    is_openable: bool = True

    @property
    def is_open(self):
        return not self.is_openable

    def get_opened_version(self) -> "Food":
        if self.is_open:
            return self
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


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Mulberry(Food):
    resource_file: str = "res://items/food/mulberry.tscn"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Carrot(Food):
    is_grown_on_trees: bool = False
    resource_file: str = "res://items/food/carrot.tscn"

    def get_offset(self) -> float:
        return 0.0


GetHeightAt = Callable[[Tuple[float, float]], float]


def normalized(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _facing_2d(from_point: Point3DNP, to_point: Point3DNP) -> Rotation:
    dir = to_point - from_point
    dir[1] = 0
    dir = -normalized(dir)
    yaw = np.arctan2(dir[0], dir[2])
    return Rotation.from_euler("y", yaw)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FoodTreeBase(InstancedDynamicItem):
    resource_file: str = "res://scenery/tree_base.tscn"
    entity_id: int = -1
    rotation: Rotation = Rotation.identity()

    def get_offset(self) -> float:
        return -0.25


FOOD_TREE_BASE_HEIGHT = 2.0

CANONICAL_FOOD_HEIGHT_ON_TREE = 2.0
FOOD_TREE_VISIBLE_HEIGHT = 11.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FoodTree(InstancedDynamicItem):
    ROTATION_TO_SUITABLE_OFFSET: ClassVar = {
        (0,): (1.25, CANONICAL_FOOD_HEIGHT_ON_TREE),
        # TODO add mechanism for multiple rotations
        # (45,): (0.875, 3.25),
    }

    @property
    def food_offset(self):
        return self.ROTATION_TO_SUITABLE_OFFSET[(0,)]

    def _relative_position(self, offset: np.array) -> np.ndarray:
        return self.rotation.apply(offset) * self.scale

    def get_food_height(self, food: _FoodT) -> float:
        xz, y = self.food_offset
        return y * self.scale[1]

    entity_id: int = -1
    resource_file: str = "res://scenery/trees/fruit_tree_normal.tscn"
    is_tool_helpful: bool = True
    ground_probability: float = 1.0
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])
    rotation: Rotation = Rotation.identity()

    is_food_on_tree: bool = False

    @property
    def height(self):
        scale_y = self.scale[1]
        return FOOD_TREE_BASE_HEIGHT * scale_y

    @staticmethod
    def build(tree_height: float, ground_probability: float, is_food_on_tree: bool = False) -> "FoodTree":
        scale_factor = tree_height / FOOD_TREE_BASE_HEIGHT
        return FoodTree(
            scale=scale_factor * np.array([1.0, 1.0, 1.0]),
            ground_probability=ground_probability,
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
        rotation = _facing_2d(trunk_placement, spawn)
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
        scales: Point3DListNP,
        min_radius: float,
        max_radius: float,
        offset: float,
        island_mask: MapBoolNP,
    ) -> Point3DListNP:
        # TODO some food should be on tree, but this doesn't work now
        ground_count = count  # (rand.uniform(size=count) < self.ground_probability).sum()
        ground_points = get_random_ground_points(
            rand, center, ground_count, map, scales, min_radius, max_radius, offset, island_mask
        )
        tree_point_count = count - ground_count
        if tree_point_count == 0:
            return ground_points
        random_heights = rand.uniform(0.5, 0.9, size=tree_point_count) * self.height
        zeros = np.zeros_like(random_heights)
        height_vectors = np.stack([zeros, random_heights, zeros])
        tree_points = self.position + height_vectors
        return tree_points

    def get_offset(self) -> float:
        return 0.0

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        node = super().get_node(scene)
        node.properties["scale"] = GDObject("Vector3", *self.scale)
        return node


FOODS = (
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


TREE_FOOD_OFFSET = 2.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Pillar(Item):
    size: np.ndarray
    is_dynamic: bool = False
    yaw: float = 0.0

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        yaw_rotation = Rotation.from_euler("y", self.yaw, degrees=True)
        rotation = yaw_rotation.as_matrix().flatten()
        transform = GDObject("Transform", *rotation, *self.position)
        root = GDNode(
            f"pillar_{self.entity_id}",
            type="StaticBody",
            properties={
                "transform": transform,
            },
        )
        root.add_child(
            GDNode(
                "mesh",
                type="MeshInstance",
                properties={
                    "mesh": scene.add_sub_resource(
                        "CubeMesh",
                        material=scene.add_ext_resource("res://shaders/BasicColor.material", "Material").reference,
                        size=Vector3(*self.size),
                    ).reference,
                    "material/0": None,
                },
            )
        )
        root.add_child(
            GDNode(
                "collision",
                type="CollisionShape",
                properties={
                    "shape": scene.add_sub_resource(
                        "BoxShape", resource_local_to_scene=True, extents=Vector3(*self.size / 2.0)
                    ).reference
                },
            )
        )

        return root


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Altar(Pillar):
    position: np.ndarray = np.array([0, 0, 0])
    entity_id: int = 0
    is_tool_helpful: bool = False

    @staticmethod
    def build(height: float, table_dim: float = 1.0) -> "Altar":
        return Altar(size=np.array([table_dim, height, table_dim]))

    def place(self, food_ground_position: Point3DNP) -> "Altar":
        up_vector = np.array([0.0, 1.0, 0.0])
        return attr.evolve(self, position=food_ground_position.copy() + up_vector * self.get_offset())

    def get_food_height(self, food: Food) -> float:
        return self.size[1] + food.get_offset()

    def get_offset(self) -> float:
        return self.size[1] / 2.0

    def get_food_locations(
        self,
        rand: np.random.Generator,
        center: Point3DNP,
        count: int,
        map: HeightMap,
        scales: Point3DListNP,
        min_radius: float,
        max_radius: float,
        offset: float,
        island_mask: MapBoolNP,
    ) -> Point3DListNP:
        points = get_random_ground_points(
            rand, center, count, map, scales, min_radius, max_radius, offset, island_mask
        )
        points[:, 1] = self.size[1] + offset
        return points


def get_random_ground_points(
    rand: np.random.Generator,
    center: Point3DNP,
    count: int,
    map: HeightMap,
    scales: Point3DListNP,
    min_radius: float,
    max_radius: float,
    offset: Union[float, np.ndarray],
    island_mask: MapBoolNP,
) -> Point3DListNP:
    assert count > 0
    acceptable_points = None
    for i in range(10):
        radius = rand.uniform(min_radius, max_radius, size=count * 2 + 5)
        angle = rand.uniform(0, np.pi * 2, size=count * 2 + 5)
        x = np.cos(angle) * radius + center[0]
        y = np.sin(angle) * radius + center[2]
        points = np.stack([x, y], axis=1)
        points = map.restrict_points_to_region(points)
        idx_x, idx_y = map.points_to_indices(points)
        points = points[island_mask[idx_x, idx_y]]
        heights = map.get_heights(points)
        points_3d = np.stack([points[:, 0], heights, points[:, 1]], axis=1)
        if acceptable_points is None:
            acceptable_points = points_3d
        else:
            acceptable_points = np.concatenate([acceptable_points, points_3d], axis=0)
        if acceptable_points.shape[0] >= count:
            acceptable_points = acceptable_points[:count, :]
            acceptable_points[:, 1] += scales[:, 1] * offset
            return acceptable_points
    raise ImpossibleWorldError("Unable to create enough random points. Likely in the water.")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DoorLock(Entity):
    def get_additional_door_body_nodes(
        self, scene: ImprovedGodotScene, body_size: np.ndarray, body_offset: float = 0
    ) -> List[GDNode]:
        # TODO: body_offset is a hack to deal with hinge/sliding doors having different origin points :/
        return []


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DoorOpenButton(DoorLock):
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = IDENTITY_BASIS

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        OPEN_BUTTON = "open_button"
        button_child_nodes = make_block_child_nodes(
            scene,
            "res://shaders/SphereNoise.material",
            make_mesh=False,
            position=Vector3(0, 0, 0),
            size=Vector3(*self.size),
        )
        mesh_base_size = np.array([0.2, 0.2, 0.2])
        mesh_resource = scene.add_ext_resource("res://entities/doors/open_button.tscn", "PackedScene")
        scale_factors = self.size / mesh_base_size
        button_mesh_node = GDNode(
            "mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        button_child_nodes.append(button_mesh_node)

        button_node = GDNode(
            f"{OPEN_BUTTON}_{self.entity_id}",
            type="RigidBody",
            properties={
                "script": scene.add_ext_resource("res://entities/doors/open_button.gd", "Script").reference,
                "entity_id": self.entity_id,
                "transform": make_transform(position=self.position, rotation=self.rotation),
            },
        )
        for child_node in button_child_nodes:
            button_node.add_child(child_node)
        return button_node


class HingeSide(Enum):
    RIGHT = "right"
    LEFT = "left"


class MountSlot(Enum):
    TOP = "top"
    BOTTOM = "bottom"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RotatingBar(DoorLock):
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    rotation_axis: Axis = Axis.Z
    anchor_side: HingeSide = HingeSide.RIGHT
    unlatch_angle: float = 10.0  # degrees

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        ROTATING_BAR = "rotating_bar"
        ROTATING_BAR_BODY = "bar_body"
        ROTATING_BAR_ANCHOR = "anchor"

        anchor_collision_mesh_node = GDNode(
            "collision_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource("SphereShape", radius=0).reference,
                "disabled": True,
            },
        )
        anchor_node = GDNode(ROTATING_BAR_ANCHOR, "StaticBody")
        anchor_node.add_child(anchor_collision_mesh_node)

        joint_lower_angle = 0 if self.anchor_side == HingeSide.RIGHT else -180
        joint_upper_angle = -180 if self.anchor_side == HingeSide.RIGHT else 0
        hinge_joint_node = GDNode(
            "hinge_joint",
            "HingeJoint",
            properties={
                "nodes/node_a": NodePath(f"../{ROTATING_BAR_ANCHOR}"),
                "nodes/node_b": NodePath(f"../{ROTATING_BAR_BODY}"),
                "angular_limit/enable": True,
                "angular_limit/lower": joint_lower_angle,
                "angular_limit/upper": joint_upper_angle,
                "angular_limit/relaxation": 0.25,
            },
        )

        bar_offset = Vector3(self.size[0] / 2, 0, 0)
        bar_body_child_nodes = make_block_child_nodes(
            scene,
            "res://shaders/BasicColorUnshaded.material",
            position=bar_offset,
            size=Vector3(*self.size),
            make_mesh=False,
        )
        mesh_base_size = np.array([1.2, 0.2, 0.2])
        mesh_resource = scene.add_ext_resource("res://entities/doors/rotating_bar.tscn", "PackedScene")
        scale_factors = self.size / mesh_base_size
        bar_body_mesh_node = GDNode(
            "mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    position=bar_offset,
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        bar_body_child_nodes.append(bar_body_mesh_node)

        body_script = scene.add_ext_resource("res://entities/doors/bar_body.gd", "Script")
        bar_body_rotation_degrees = -180 if self.anchor_side == HingeSide.RIGHT else 0
        bar_body_rotation = (
            Rotation.from_euler(Axis.Z.value, bar_body_rotation_degrees, degrees=True).as_matrix().flatten()
        )
        body_node = GDNode(
            ROTATING_BAR_BODY,
            type="RigidBody",
            properties={
                "transform": make_transform(rotation=bar_body_rotation),
                "mass": 10.0,
                "script": body_script.reference,
                "entity_id": 0,
            },
        )
        for child_node in bar_body_child_nodes:
            body_node.add_child(child_node)

        # todo: extract these all out somewhere
        support_mesh_base_size = np.array([0.087, 0.18, 0.3])
        support_width, support_height, support_thickness = 0.025, 0.1, 0.29
        support_size = np.array([support_width, support_height, support_thickness])
        bar_support_x = -self.size[0] * 0.8
        if self.anchor_side == HingeSide.RIGHT:
            bar_support_x = -bar_support_x
        bar_support_position = (bar_support_x, 0, -support_thickness / 2.5)
        bar_support_mesh_resource = scene.add_ext_resource("res://entities/doors/bar_support.tscn", "PackedScene")
        scale_factors = support_size / support_mesh_base_size
        bar_support_child_nodes = make_block_child_nodes(
            scene,
            "res://shaders/BasicColorUnshaded.material",
            position=Vector3(0, 0, 0),
            size=Vector3(*support_size),
            make_mesh=False,
            collision_shape_disabled=True,
        )
        bar_support_mesh_node = GDNode(
            "mesh",
            instance=bar_support_mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        bar_support_child_nodes.append(bar_support_mesh_node)
        support_spatial_node = GDNode(
            f"bar_support",
            "StaticBody",
            properties={"transform": make_transform(position=bar_support_position)},
        )
        for child_node in bar_support_child_nodes:
            support_spatial_node.add_child(child_node)

        bar_script = scene.add_ext_resource("res://entities/doors/rotating_bar.gd", "Script")
        group_spatial_node = GDNode(
            f"{ROTATING_BAR}_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(position=self.position, rotation=self.rotation),
                "script": bar_script.reference,
                "entity_id": self.entity_id,
                "rotation_axis": self.rotation_axis.value,
                "anchor_side": self.anchor_side.value,
                "unlatch_angle": self.unlatch_angle,
            },
        )
        group_spatial_node.add_child(body_node)
        group_spatial_node.add_child(anchor_node)
        group_spatial_node.add_child(hinge_joint_node)
        group_spatial_node.add_child(support_spatial_node)
        return group_spatial_node

    def get_additional_door_body_nodes(
        self, scene: ImprovedGodotScene, body_size: np.ndarray, body_x_offset: float = 0
    ) -> List[GDNode]:
        hook_width, hook_height, hook_thickness = 0.025, 0.1, 0.2
        hook_size = np.array([hook_width, hook_height, hook_thickness])
        hook_nodes = []
        bar_x, bar_y, bar_z = self.position
        body_width, body_height, body_thickness = body_size
        hook_span = body_width * 0.4
        hook_distance = hook_span / 5
        hook_x_start = 0 if self.anchor_side == HingeSide.RIGHT else -hook_span
        hook_x_positions = np.arange(hook_x_start, hook_x_start + hook_span + hook_distance, hook_distance)
        hook_mesh_base_size = np.array([0.087, 0.18, 0.3])
        for i, hook_x in enumerate(hook_x_positions):
            bar_hook_offset = Vector3(hook_x, 0, 0)
            bar_hook_mesh_resource = scene.add_ext_resource("res://entities/doors/bar_support.tscn", "PackedScene")
            scale_factors = hook_size / hook_mesh_base_size
            bar_hook_child_nodes = make_block_child_nodes(
                scene,
                "res://shaders/BasicColorUnshaded.material",
                position=bar_hook_offset,
                size=Vector3(*hook_size),
                make_mesh=False,
                collision_shape_name=f"collision_shape_{i}",
                collision_shape_disabled=True,
            )
            bar_hook_mesh_node = GDNode(
                f"mesh_{i}",
                instance=bar_hook_mesh_resource.reference.id,
                properties={
                    "transform": make_transform(
                        position=bar_hook_offset,
                        rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                    )
                },
            )
            hook_nodes.extend(bar_hook_child_nodes)
            hook_nodes.append(bar_hook_mesh_node)

        bar_hook_position = body_x_offset, bar_y * 0.98, hook_thickness * 0.6
        hook_spatial_node = GDNode(
            f"bar_hooks",
            "StaticBody",
            properties={"transform": make_transform(position=bar_hook_position)},
        )
        for hook_node in hook_nodes:
            hook_spatial_node.add_child(hook_node)
        return [hook_spatial_node]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SlidingBar(DoorLock):
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    sliding_axis: Axis = Axis.Y
    door_face_axis: Axis = Axis.Z
    mount_slot: MountSlot = MountSlot.BOTTOM
    mount_side: HingeSide = HingeSide.LEFT
    proportion_to_unlock: float = 0.25

    @property
    def latch_size(self):
        bar_width, bar_height, bar_thickness = self.size
        latch_width, latch_height, latch_thickness = bar_width * 1.6, bar_width / 2, bar_thickness * 1.05
        return latch_width, latch_height, latch_thickness

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        # todo: consolidate common bits with sliding door eww; add higher-level godot parser wrapper?
        SLIDING_BAR = "sliding_bar"
        SLIDING_BAR_BODY = "bar_body"
        SLIDING_BAR_ANCHOR = "anchor"

        anchor_collision_mesh_node = GDNode(
            "collision_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource("SphereShape", radius=0).reference,
                "disabled": True,
            },
        )
        anchor_node = GDNode(SLIDING_BAR_ANCHOR, "StaticBody")
        anchor_node.add_child(anchor_collision_mesh_node)

        bar_width, bar_height, bar_thickness = self.size
        # latch_width, latch_height, latch_thickness = 0.08, 0.17, 0.29
        leeway_multiplier = 1.2
        if self.mount_slot == MountSlot.BOTTOM:
            lower_distance = 0
            upper_distance = bar_height * self.proportion_to_unlock * leeway_multiplier
        else:
            lower_distance = -bar_height * self.proportion_to_unlock * leeway_multiplier
            upper_distance = 0

        slider_joint_node = GDNode(
            "slider_joint",
            "SliderJoint",
            properties={
                "transform": make_transform(rotation=Rotation.from_euler("z", 90, degrees=True).as_matrix().flatten()),
                "nodes/node_a": NodePath(f"../{SLIDING_BAR_ANCHOR}"),
                "nodes/node_b": NodePath(f"../{SLIDING_BAR_BODY}"),
                "linear_limit/lower_distance": lower_distance,
                "linear_limit/upper_distance": upper_distance,
                "linear_limit/softness": 0.5,
            },
        )

        x_multiplier = -1 if self.mount_side == HingeSide.LEFT else 1

        latch_width, _latch_height, _latch_thickness = self.latch_size
        # Main body of the bar
        bar_body_offset = np.array([-x_multiplier * latch_width / 2, 0, bar_thickness / 2])
        bar_body_child_nodes = make_block_child_nodes(
            scene,
            "res://shaders/BasicColorUnshaded.material",
            position=Vector3(*bar_body_offset),
            size=Vector3(*self.size),
            make_mesh=False,
            collision_shape_name="body_collision_shape",
        )
        mesh_base_size = np.array([0.1, 0.6, 0.1])
        mesh_resource = scene.add_ext_resource("res://entities/doors/sliding_bar.tscn", "PackedScene")
        scale_factors = self.size / mesh_base_size
        bar_body_mesh_node = GDNode(
            "mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    position=bar_body_offset,
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        bar_body_child_nodes.append(bar_body_mesh_node)

        # Knob
        y_multiplier = 1 if self.mount_slot == MountSlot.BOTTOM else -1
        knob_thickness = 0.1
        knob_size = np.array([0.1, 0.1, knob_thickness])
        knob_offset = bar_body_offset + np.array(
            [0, y_multiplier * bar_height / 2.2, bar_thickness / 2 + knob_thickness / 2]
        )
        knob_child_nodes = make_block_child_nodes(
            scene,
            "res://shaders/BasicColorUnshaded.material",
            position=Vector3(*knob_offset),
            size=Vector3(*knob_size),
            make_mesh=False,
            collision_shape_name="knob_collision_shape",
        )
        # knob_mesh_base_size = np.array([0.1, 0.6, 0.1])
        knob_mesh_resource = scene.add_ext_resource("res://entities/doors/bar_knob.tscn", "PackedScene")
        scale_factors = np.array([1, 1, 1])
        knob_mesh_node = GDNode(
            "knob_mesh",
            instance=knob_mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    position=knob_offset,
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        knob_child_nodes.append(knob_mesh_node)
        bar_body_child_nodes.extend(knob_child_nodes)

        # Guiderail on the side
        rail_height = bar_height * (1 + self.proportion_to_unlock)
        rail_width = bar_width * 1.5
        rail_thickness = bar_width * 1.5
        # The rail is normally horizontal, so we define its size as if it was going to be placed horizontal
        # and then rotate it
        rail_size = Vector3(rail_height, rail_width, rail_thickness)
        rail_position = (
            x_multiplier * rail_width / 2,
            0.5 * self.proportion_to_unlock * bar_height,
            bar_thickness / 1.8,
        )

        mesh_base_size = np.array([2, 0.135, 0.1])
        mesh_resource = scene.add_ext_resource("res://entities/doors/rail.tscn", "PackedScene")
        scale_factors = np.array([rail_size.x, rail_size.y, rail_size.z]) / mesh_base_size
        # todo: decide based on side
        rotation_degrees = 90
        rail_mesh_node = GDNode(
            "mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    rotation=scale_basis(Rotation.identity().as_matrix().flatten(), scale_factors),
                )
            },
        )

        rail_collision_mesh_node = GDNode(
            "collision_shape",
            "CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape",
                    extents=godot_parser.Vector3(rail_size[0] / 2, rail_size[1] / 2, rail_size[2] / 2),
                ).reference,
                "disabled": True,
            },
        )
        rail_node = GDNode(
            "rail",
            "StaticBody",
            properties={
                "transform": make_transform(
                    position=rail_position,
                    rotation=Rotation.from_euler("z", rotation_degrees, degrees=True).as_matrix().flatten(),
                ),
            },
        )
        rail_node.add_child(rail_mesh_node)
        rail_node.add_child(rail_collision_mesh_node)

        # Guiderail hinges
        hinge_span = (bar_height * (1 - self.proportion_to_unlock)) * 0.7  # 70% of the part of bar above the slot
        hinge_width = bar_width * 2.1
        hinge_height = bar_width
        hinge_thickness = 0.05
        offset_from_bottom = (self.proportion_to_unlock * bar_height) / 2
        # The hinge is normally vertical, so we define its size as if it was going to be placed vertically, then rotate
        hinge_size = np.array([hinge_height, hinge_width, hinge_thickness])
        for i, y_offset in enumerate([-hinge_span / 2 + offset_from_bottom, hinge_span / 2 + offset_from_bottom]):
            hinge_position = x_multiplier * rail_width / 2, y_offset, bar_thickness / 1.8
            hinge_mesh_base_size = np.array([0.12, 0.32, 0.05])
            hinge_mesh_resource = scene.add_ext_resource("res://entities/doors/rail_hinge.tscn", "PackedScene")
            scale_factors = hinge_size / hinge_mesh_base_size
            rail_hinge_mesh_node = GDNode(
                "hinge_mesh",
                instance=hinge_mesh_resource.reference.id,
                properties={
                    "transform": make_transform(
                        rotation=scale_basis(Rotation.identity().as_matrix().flatten(), scale_factors),
                    )
                },
            )
            rail_collision_mesh_node = GDNode(
                "collision_shape",
                "CollisionShape",
                properties={
                    "shape": scene.add_sub_resource(
                        "BoxShape",
                        extents=godot_parser.Vector3(hinge_size[0] / 2, hinge_size[1] / 2, hinge_size[2] / 2),
                    ).reference,
                    "disabled": True,
                },
            )
            y_rotation = 0 if self.mount_side == HingeSide.LEFT else 180
            rail_hinge_node = GDNode(
                f"rail_hinge_{i}",
                "StaticBody",
                properties={
                    "transform": make_transform(
                        position=hinge_position,
                        rotation=Rotation.from_euler("zy", (90, y_rotation), degrees=True).as_matrix().flatten(),
                    ),
                },
            )
            rail_hinge_node.add_child(rail_hinge_mesh_node)
            rail_hinge_node.add_child(rail_collision_mesh_node)
            bar_body_child_nodes.append(rail_hinge_node)

        body_script = scene.add_ext_resource("res://entities/doors/vertical_bar_body.gd", "Script")
        body_node = GDNode(
            SLIDING_BAR_BODY,
            type="RigidBody",
            properties={
                "mass": 10.0,
                "script": body_script.reference,
                "entity_id": 0,
            },
        )
        for child_node in bar_body_child_nodes:
            body_node.add_child(child_node)

        bar_script = scene.add_ext_resource("res://entities/doors/sliding_bar.gd", "Script")
        group_spatial_node = GDNode(
            f"{SLIDING_BAR}_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(position=self.position, rotation=self.rotation),
                "script": bar_script.reference,
                "sliding_axis": self.sliding_axis.value,
                "proportion_to_unlock": self.proportion_to_unlock,
                "entity_id": self.entity_id,
            },
        )
        group_spatial_node.add_child(body_node)
        group_spatial_node.add_child(anchor_node)
        group_spatial_node.add_child(slider_joint_node)
        group_spatial_node.add_child(rail_node)
        # for latch_spatial_node in latch_spatial_nodes:
        #     group_spatial_node.add_child(latch_spatial_node)
        return group_spatial_node

    def get_additional_door_body_nodes(
        self, scene: ImprovedGodotScene, body_size: np.ndarray, body_x_offset: float = 0
    ) -> List[GDNode]:
        latch_mesh_base_size = np.array([0.15, 0.09, 0.181])
        bar_width, bar_height, bar_thickness = self.size
        door_body_width, door_body_height, door_body_thickness = body_size
        # latch_width, latch_height, latch_thickness = 0.08, 0.17, 0.29
        latch_width, latch_height, latch_thickness = self.latch_size
        latch_size = np.array([latch_width, latch_height, latch_thickness])
        x_multiplier = -1 if self.mount_side == HingeSide.LEFT else 1
        y_multiplier = -1 if self.mount_slot == MountSlot.BOTTOM else 1
        bar_y_position = self.position[1]

        bar_latch_positions = [
            (
                body_x_offset + x_multiplier * (door_body_width / 2 - latch_width / 2),
                bar_y_position + y_multiplier * bar_height * self.proportion_to_unlock,
                door_body_thickness / 2 + bar_thickness / 2,
            ),
        ]
        scale_factors = latch_size / latch_mesh_base_size
        latch_spatial_nodes = []

        for i, bar_latch_position in enumerate(bar_latch_positions):
            bar_latch_mesh_resource = scene.add_ext_resource("res://entities/doors/bar_latch.tscn", "PackedScene")
            bar_latch_child_nodes = make_block_child_nodes(
                scene,
                "res://shaders/BasicColorUnshaded.material",
                position=Vector3(0, 0, 0),
                size=Vector3(*latch_size),
                make_mesh=False,
                collision_shape_disabled=True,
            )
            bar_latch_mesh_node = GDNode(
                "mesh",
                instance=bar_latch_mesh_resource.reference.id,
                properties={
                    "transform": make_transform(
                        rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                    )
                },
            )
            bar_latch_child_nodes.append(bar_latch_mesh_node)
            latch_spatial_node = GDNode(
                f"bar_latch_{self.entity_id}_{i}",
                "StaticBody",
                properties={"transform": make_transform(position=bar_latch_position)},
            )
            for child_node in bar_latch_child_nodes:
                latch_spatial_node.add_child(child_node)
            latch_spatial_nodes.append(latch_spatial_node)
            return latch_spatial_nodes


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Door(Item):
    # for doors, position is the door's centroid - the middle point of the plane we're trying to close using the door
    is_dynamic: bool = True
    size: np.ndarray = np.array([1, DEFAULT_STORY_HEIGHT, 0.1])
    rotation: np.ndarray = IDENTITY_BASIS
    locks: List[DoorLock] = attr.field(default=attr.Factory(list))


class LatchingMechanics(Enum):
    NO_LATCH = 0
    LATCH_ONCE = 1
    AUTO_LATCH = 2


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class SlidingDoor(Door):
    # todo: deduplication between sliding/hinge door
    face_axis: Axis = Axis.X
    slide_right: bool = True
    handle_size_proportion: float = 0.2
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        DOOR_BODY = "body"

        body_nodes = self._get_body_nodes(scene, DOOR_BODY)
        rail_nodes = self._get_rail_nodes(scene, DOOR_BODY)

        door_script = scene.add_ext_resource("res://entities/doors/sliding_door.gd", "Script")
        group_spatial_node = GDNode(
            f"sliding_door_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(self.rotation, self.position),
                "script": door_script.reference,
                "entity_id": self.entity_id,
                "slide_axis": self.face_axis.value,
                "is_latched": self.latching_mechanics != LatchingMechanics.NO_LATCH,
            },
        )
        for body_node in body_nodes:
            group_spatial_node.add_child(body_node)
        for rail_node in rail_nodes:
            group_spatial_node.add_child(rail_node)

        body_node = only(body_nodes)
        for i, lock in enumerate(self.locks):
            # if we have two locks of the same kind, their instance names will otherwise craash
            lock_with_id = attr.evolve(lock, entity_id=i)
            group_spatial_node.add_child(lock_with_id.get_node(scene))
            for body_addition in lock_with_id.get_additional_door_body_nodes(scene, self.size):
                body_node.add_child(body_addition)
        return group_spatial_node

    def _get_body_nodes(self, scene: ImprovedGodotScene, door_body_node_name: str) -> List[GDNode]:
        # The door body here also includes the handle, since PinJoints suck in Godot, so we can't have them separate :/
        body_children_nodes = make_block_child_nodes(
            scene,
            "res://shaders/BasicColor.material",
            position=Vector3(0, 0, 0),
            size=Vector3(*self.size),
            make_mesh=False,
            collision_shape_name=f"{door_body_node_name}_collision_shape",
        )
        mesh_base_size = np.array([1, 2.7, 0.1])
        mesh_resource = scene.add_ext_resource("res://entities/doors/door_body.tscn", "PackedScene")
        scale_factors = self.size / mesh_base_size
        body_mesh_node = GDNode(
            "body_mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors)
                )
            },
        )
        body_children_nodes.append(body_mesh_node)
        handle_nodes = self._get_handle_nodes(scene)

        door_width, door_height, door_thickness = self.size
        rail_hinge_distance = door_width * 0.5
        rail_hinge_width, rail_hinge_height, rail_hinge_thickness = 0.12, 0.32, 0.05
        rail_hinge_offset = (door_thickness - rail_hinge_thickness) / 2

        left_hinge_mesh_base_size = np.array([0.12, 0.32, 0.05])
        left_hinge_mesh_resource = scene.add_ext_resource("res://entities/doors/rail_hinge.tscn", "PackedScene")
        scale_factors = np.array([1, 1, 1])  # np.array([rail_size.x, rail_size.y, rail_size.z]) / mesh_base_size
        left_hinge_mesh_node = GDNode(
            "left_hinge_mesh",
            instance=left_hinge_mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    position=(-rail_hinge_distance / 2, door_height / 2, rail_hinge_offset),
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        body_children_nodes.append(left_hinge_mesh_node)

        right_hinge_mesh_base_size = np.array([0.12, 0.32, 0.05])
        right_hinge_mesh_resource = scene.add_ext_resource("res://entities/doors/rail_hinge.tscn", "PackedScene")
        scale_factors = np.array([1, 1, 1])  # np.array([rail_size.x, rail_size.y, rail_size.z]) / mesh_base_size
        right_hinge_mesh_node = GDNode(
            "right_hinge_mesh",
            instance=right_hinge_mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    position=(rail_hinge_distance / 2, door_height / 2, rail_hinge_offset),
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        body_children_nodes.append(right_hinge_mesh_node)

        body_script = scene.add_ext_resource("res://entities/doors/door_body.gd", "Script")
        body_node = GDNode(
            door_body_node_name,
            type="RigidBody",
            properties={
                "transform": make_transform(),
                "mass": 100.0,
                "script": body_script.reference,
                "entity_id": 0,
                "is_auto_latching": self.latching_mechanics == LatchingMechanics.AUTO_LATCH,
            },
        )
        for node in body_children_nodes:
            body_node.add_child(node)
        for node in handle_nodes:
            body_node.add_child(node)
        return [body_node]

    def _get_handle_nodes(self, scene: ImprovedGodotScene) -> List[GDNode]:
        DOOR_HANDLE = "handle"

        door_width, door_height, door_thickness = self.size
        handle_width = door_width * (self.handle_size_proportion / 2)
        handle_height = door_height * self.handle_size_proportion
        handle_thickness = door_thickness * 0.75

        # We cap the handle height if there are locks that would get obstructed by it
        leeway = 0.05
        bar_locks = [lock for lock in self.locks if isinstance(lock, (RotatingBar, SlidingBar))]
        if len(bar_locks) > 0:
            max_y = np.inf
            min_y = -np.inf
            for lock in bar_locks:
                if not isinstance(lock, (RotatingBar, SlidingBar)):
                    continue
                if lock.position[1] > 0 and (lock_bottom := lock.position[1] - lock.size[1] / 2) < max_y:
                    max_y = lock_bottom
                elif lock.position[1] < 0 and (lock_top := lock.position[1] + lock.size[1] / 2) > min_y:
                    min_y = lock_top
            max_handle_height = min(abs(max_y), abs(min_y)) * 2 - leeway
            if handle_height > max_handle_height:
                handle_height = max_handle_height
        handle_size = Vector3(handle_width, handle_height, handle_thickness)
        handle_margin_skew = 25  # left margin / right margin
        handle_side_margin = (door_width - handle_width) / (1 + handle_margin_skew)
        if not self.slide_right:
            handle_side_margin *= handle_margin_skew

        handle_offsets = 1, -1
        handle_nodes = []
        for i, handle_offset in enumerate(handle_offsets):
            handle_position = Vector3(
                -door_width / 2 + handle_side_margin + handle_width / 2,
                0,
                handle_offset * (handle_thickness / 2 + door_thickness / 2),
            )
            handle_name = f"{DOOR_HANDLE}_{i}"
            handle_nodes.extend(
                make_block_child_nodes(
                    scene,
                    "res://shaders/BasicColor.material",
                    handle_position,
                    handle_size,
                    make_mesh=False,
                    collision_shape_name=f"{handle_name}_collision_shape",
                )
            )
            handle_mesh_base_size = np.array([0.05, 0.75, 0.075])
            mesh_resource = scene.add_ext_resource("res://entities/doors/door_handle_vertical.tscn", "PackedScene")
            scale_factors = np.array([handle_size.x, handle_size.y, handle_size.z]) / handle_mesh_base_size
            handle_mesh_node = GDNode(
                f"{handle_name}_mesh",
                instance=mesh_resource.reference.id,
                properties={
                    "transform": make_transform(
                        position=handle_position,
                        rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                    )
                },
            )
            handle_nodes.append(handle_mesh_node)
            # todo: place inside handle as well (need to increase wall gap)
            break
        return handle_nodes

    def _get_rail_nodes(self, scene: ImprovedGodotScene, door_body_node_name: str) -> List[GDNode]:
        DOOR_RAIL = "rail"

        door_width, door_height, door_thickness = self.size
        rail_height = door_height * 0.05
        rail_size = Vector3(door_width * 2, rail_height, door_thickness * 1.1)
        direction_multiplier = 1 if self.slide_right else -1
        rail_position = direction_multiplier * door_width / 2, door_height / 2 + rail_height / 2, 0

        mesh_base_size = np.array([2, 0.135, 0.1])
        mesh_resource = scene.add_ext_resource("res://entities/doors/rail.tscn", "PackedScene")
        scale_factors = np.array([rail_size.x, rail_size.y, rail_size.z]) / mesh_base_size
        rail_mesh_node = GDNode(
            "mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )

        rail_collision_mesh_node = GDNode(
            "collision_shape",
            "CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape",
                    extents=godot_parser.Vector3(rail_size[0] / 2, rail_size[1] / 2, rail_size[2] / 2),
                ).reference,
                "disabled": True,
            },
        )
        rail_node = GDNode(
            DOOR_RAIL,
            "StaticBody",
            properties={
                "transform": make_transform(position=rail_position),
            },
        )
        rail_node.add_child(rail_mesh_node)
        rail_node.add_child(rail_collision_mesh_node)

        slider_joint_position = direction_multiplier * door_width, door_height / 2, 0
        slider_joint_node = GDNode(
            "slider_joint",
            "SliderJoint",
            properties={
                "transform": make_transform(position=slider_joint_position),
                "nodes/node_a": NodePath(f"../{DOOR_RAIL}"),
                "nodes/node_b": NodePath(f"../{door_body_node_name}"),
                "linear_limit/lower_distance": 0 if self.slide_right else -door_width,
                "linear_limit/upper_distance": door_width if self.slide_right else 0,
                "linear_limit/softness": 0.5,
            },
        )
        return [rail_node, slider_joint_node]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HingeDoor(Door):
    hinge_side: HingeSide = HingeSide.LEFT
    hinge_radius: float = 0.05
    face_axis: Axis = Axis.X
    max_inwards_angle: float = 90.0
    max_outwards_angle: float = 90.0
    handle_size_proportion: float = 0.2
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH

    def get_node(self, scene: ImprovedGodotScene) -> GDNode:
        DOOR_BODY = "body"

        body_nodes = self._get_body_nodes(scene, DOOR_BODY)
        hinge_nodes = self._get_hinge_nodes(scene, DOOR_BODY)

        door_script = scene.add_ext_resource("res://entities/doors/hinge_door.gd", "Script")
        group_spatial_node = GDNode(
            f"hinge_door_{self.entity_id}",
            "Spatial",
            properties={
                "transform": make_transform(self.rotation, self.position),
                "script": door_script.reference,
                "entity_id": self.entity_id,
                "is_latched": self.latching_mechanics != LatchingMechanics.NO_LATCH,
            },
        )
        for body_node in body_nodes:
            group_spatial_node.add_child(body_node)
        for hinge_node in hinge_nodes:
            group_spatial_node.add_child(hinge_node)
        body_node = only(body_nodes)
        door_body_width = self.size[0]  #  - self.hinge_radius * 2
        door_body_size = np.array([door_body_width, self.size[1], self.size[2]])
        # TODO: fix this hack :/
        body_offset = -door_body_width / 2 if self.hinge_side == HingeSide.RIGHT else door_body_width / 2
        for i, lock in enumerate(self.locks):
            # if we have two locks of the same kind, their instance names will otherwise crash
            lock_with_id = attr.evolve(lock, entity_id=i)
            group_spatial_node.add_child(lock_with_id.get_node(scene))
            for body_addition in lock_with_id.get_additional_door_body_nodes(scene, door_body_size, body_offset):
                body_node.add_child(body_addition)
        return group_spatial_node

    def _get_body_nodes(self, scene, door_body_node_name):
        # The door body here also includes the handle, since PinJoints suck in Godot, so we can't have them separate :/
        door_width, door_height, door_thickness = self.size

        if self.hinge_side == HingeSide.LEFT:
            body_offset = Vector3(door_width / 2, 0, 0)
            body_position = Vector3(-door_width / 2 + self.hinge_radius, 0, 0)
        else:
            body_offset = Vector3(-door_width / 2, 0, 0)
            body_position = Vector3(door_width / 2 - self.hinge_radius, 0, 0)

        body_children_nodes = make_block_child_nodes(
            scene,
            "res://shaders/BasicColor.material",
            body_offset,
            Vector3(*self.size),
            collision_shape_name=f"{door_body_node_name}_collision_shape",
            make_mesh=False,
        )
        mesh_base_size = np.array([1, 2.7, 0.1])
        mesh_resource = scene.add_ext_resource("res://entities/doors/door_body.tscn", "PackedScene")
        scale_factors = self.size / mesh_base_size
        body_mesh_node = GDNode(
            "body_mesh",
            instance=mesh_resource.reference.id,
            properties={
                "transform": make_transform(
                    position=body_offset,
                    rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                )
            },
        )
        body_children_nodes.append(body_mesh_node)
        handle_nodes = self._get_handle_nodes(scene)

        body_script = scene.add_ext_resource("res://entities/doors/door_body.gd", "Script")
        body_node = GDNode(
            door_body_node_name,
            type="RigidBody",
            properties={
                "transform": make_transform(position=body_position),
                "mass": 100.0,
                "script": body_script.reference,
                "is_auto_latching": self.latching_mechanics == LatchingMechanics.AUTO_LATCH,
                "entity_id": 0,
            },
        )
        for node in body_children_nodes:
            body_node.add_child(node)
        for node in handle_nodes:
            body_node.add_child(node)

        return [body_node]

    def _get_handle_nodes(self, scene):
        DOOR_HANDLE = "handle"

        door_width, door_height, door_thickness = self.size
        handle_width = door_width * self.handle_size_proportion
        handle_height = door_height * (self.handle_size_proportion / 2)
        handle_thickness = 0.15

        # We cap the handle height if there are locks that would get obstructed by it
        leeway = 0.05
        bar_locks = [lock for lock in self.locks if isinstance(lock, (RotatingBar, SlidingBar))]
        if len(bar_locks) > 0:
            max_y = np.inf
            min_y = -np.inf
            for lock in bar_locks:
                if not isinstance(lock, (RotatingBar, SlidingBar)):
                    continue
                if lock.position[1] > 0 and (lock_bottom := lock.position[1] - lock.size[1] / 2) < max_y:
                    max_y = lock_bottom
                elif lock.position[1] < 0 and (lock_top := lock.position[1] + lock.size[1] / 2) > min_y:
                    min_y = lock_top
            max_handle_height = min(abs(max_y), abs(min_y)) * 2 - leeway
            if handle_height > max_handle_height:
                handle_height = max_handle_height

        # We cap the handle width to ensure it doesn't prevent the door from opening inwards
        max_width = math.sqrt(
            (door_width - leeway) ** 2 - handle_thickness ** 2
        )  # diagonal of handle can't exceed door width
        handle_width = min(max_width, handle_width)
        handle_size = Vector3(handle_width, handle_height, handle_thickness)

        handle_margin_skew = 25  # how many right-margins fit in the left margin
        handle_side_margin = (door_width - handle_width) / (1 + handle_margin_skew) * handle_margin_skew
        if self.hinge_side == HingeSide.LEFT:
            handle_x = handle_side_margin + handle_width / 2
        else:
            handle_x = -(handle_side_margin + handle_width / 2)

        handle_offsets = 1, -1
        handle_nodes = []
        for i, handle_offset in enumerate(handle_offsets):
            handle_position = Vector3(
                handle_x,
                0,
                handle_offset * (handle_thickness / 2 + door_thickness / 2),
            )
            handle_name = f"{DOOR_HANDLE}_{i}"
            handle_nodes.extend(
                make_block_child_nodes(
                    scene,
                    "res://shaders/BasicColor.material",
                    handle_position,
                    handle_size,
                    collision_shape_name=f"{handle_name}_collision_shape",
                    make_mesh=False,
                )
            )
            handle_mesh_base_size = np.array([0.14, 0.25, 0.05])
            mesh_resource = scene.add_ext_resource("res://entities/doors/door_handle_loop.tscn", "PackedScene")
            scale_factors = np.array([handle_size.x, handle_size.y, handle_size.z]) / handle_mesh_base_size
            handle_mesh_node = GDNode(
                f"{handle_name}_mesh",
                instance=mesh_resource.reference.id,
                properties={
                    "transform": make_transform(
                        position=handle_position,
                        rotation=scale_basis(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float64), scale_factors),
                    )
                },
            )
            handle_nodes.append(handle_mesh_node)
        return handle_nodes

    def _get_hinge_nodes(self, scene, door_body_node_name):
        DOOR_HINGE = "hinge"

        door_width, door_height, door_thickness = self.size
        frame_width = door_width + 2 * self.hinge_radius
        hinge_height = door_height
        body_left_side = -frame_width / 2 + 2 * self.hinge_radius
        body_right_side = frame_width / 2 - 2 * self.hinge_radius

        if self.hinge_side == HingeSide.LEFT:
            hinge_x = -frame_width / 2 + self.hinge_radius
        else:
            hinge_x = frame_width / 2 - self.hinge_radius
        hinge_position = Vector3(hinge_x, 0, 0)
        hinge_material = scene.add_sub_resource(
            "SpatialMaterial", albedo_color=Color(0.223529, 0.196078, 0.141176, 1), roughness=0.2, metallic=0.5
        )
        hinge_mesh_node = GDNode(
            "mesh",
            "MeshInstance",
            properties={
                "transform": make_transform(),
                "mesh": scene.add_sub_resource(
                    "CylinderMesh",
                    material=hinge_material.reference,
                    top_radius=self.hinge_radius,
                    bottom_radius=self.hinge_radius,
                    height=hinge_height,
                ).reference,
            },
        )
        hinge_collision_mesh_node = GDNode(
            "collision_shape",
            "CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "CylinderShape", radius=self.hinge_radius, height=hinge_height
                ).reference,
                "disabled": True,
            },
        )
        hinge_node = GDNode(
            DOOR_HINGE,
            "StaticBody",
            properties={
                "transform": make_transform(position=hinge_position),
            },
        )
        hinge_node.add_child(hinge_mesh_node)
        hinge_node.add_child(hinge_collision_mesh_node)

        hinge_rotation_degrees = 90 if self.hinge_side == HingeSide.LEFT else -90
        hinge_joint_position = Vector3(body_left_side if self.hinge_side == HingeSide.LEFT else body_right_side, 0, 0)
        hinge_rotation = Rotation.from_euler("x", hinge_rotation_degrees, degrees=True).as_matrix().flatten()
        hinge_joint_node = GDNode(
            "hinge_joint",
            "HingeJoint",
            properties={
                "transform": make_transform(position=hinge_joint_position, rotation=hinge_rotation),
                "nodes/node_a": NodePath(f"../{DOOR_HINGE}"),
                "nodes/node_b": NodePath(f"../{door_body_node_name}"),
                "params/bias": 0.99,
                "angular_limit/enable": True,
                "angular_limit/lower": -self.max_outwards_angle,
                "angular_limit/upper": self.max_inwards_angle,
                "angular_limit/bias": 0.99,
                "angular_limit/softness": 0.01,
                "angular_limit/relaxation": 0.01,
            },
        )
        return [hinge_node, hinge_joint_node]


ALL_DOOR_CLASSES: Final[List[Type[Door]]] = [HingeDoor, SlidingDoor]
ALL_LOCK_CLASSES: Final[List[Type[DoorLock]]] = [DoorOpenButton, SlidingBar, RotatingBar]
