from typing import ClassVar
from typing import Final
from typing import List
from typing import Optional
from typing import Type

import attr
from godot_parser import Node as GDNode

from avalon.datagen.world_creation.entities.constants import LARGEST_ANIMAL_SIZE
from avalon.datagen.world_creation.entities.item import InstancedDynamicItem
from avalon.datagen.world_creation.types import GodotScene


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Animal(InstancedDynamicItem):
    is_grounded: bool = False
    is_flying: bool = False
    is_able_to_climb: bool = False

    detection_radius_override: Optional[float] = None

    def get_offset(self):
        return LARGEST_ANIMAL_SIZE / 2

    def get_node(self, scene: GodotScene) -> GDNode:
        node = super().get_node(scene)
        if self.detection_radius_override is not None:
            node.properties["player_detection_radius"] = self.detection_radius_override
        return node


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
