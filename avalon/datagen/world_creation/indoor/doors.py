from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Type
from typing import cast

import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation

from avalon.common.errors import SwitchError
from avalon.common.utils import only
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.entities.doors.door import Door
from avalon.datagen.world_creation.entities.doors.hinge_door import HingeDoor
from avalon.datagen.world_creation.entities.doors.locks.door_open_button import DoorOpenButton
from avalon.datagen.world_creation.entities.doors.locks.rotating_bar import RotatingBar
from avalon.datagen.world_creation.entities.doors.locks.sliding_bar import SlidingBar
from avalon.datagen.world_creation.entities.doors.sliding_door import SlidingDoor
from avalon.datagen.world_creation.entities.doors.types import HingeSide
from avalon.datagen.world_creation.entities.doors.types import LatchingMechanics
from avalon.datagen.world_creation.entities.doors.types import MountSlot
from avalon.datagen.world_creation.geometry import Axis
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.indoor.components import Entrance
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import TILE_SIZE
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.types import Basis3DNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import decompose_weighted_mean

DoorParams = Tuple[Type[Door], Dict, LatchingMechanics, Tuple[Tuple[Callable, ...]]]


def get_door_params_from_difficulty(rand: np.random.Generator, difficulty: float) -> DoorParams:
    difficulty_weights = np.array([4, 0.5, 1])
    difficulties = decompose_weighted_mean(difficulty, difficulty_weights, rand=rand)
    lock_difficulty, latching_difficulty, door_mechanics_difficulty = difficulties

    hinge_side = rand.choice([HingeSide.LEFT, HingeSide.RIGHT])  # type: ignore[arg-type]
    door_mechanics_by_difficulty = {
        0.25: (HingeDoor, dict(is_pushable=True, is_pullable=True, hinge_side=hinge_side)),
        0.5: (HingeDoor, dict(is_pushable=True, is_pullable=False, hinge_side=hinge_side)),
        0.75: (HingeDoor, dict(is_pushable=False, is_pullable=True, hinge_side=hinge_side)),
        1: (SlidingDoor, dict(slide_right=rand.choice([True, False]))),
    }
    door_mechanics_difficulty_distribution = stats.norm(door_mechanics_difficulty, 0.125)
    difficulty_weights = np.array(
        [door_mechanics_difficulty_distribution.pdf(x) for x in door_mechanics_by_difficulty.keys()]
    )
    difficulty_weights /= difficulty_weights.sum()
    door_type, door_mechanics_params = tuple(
        rand.choice(list(door_mechanics_by_difficulty.values()), p=difficulty_weights)  # type: ignore[arg-type]
    )

    variants_by_difficulty = {
        0.125: [],
        0.250: [(make_open_button,)],
        0.675: [(make_rotating_bar,)],
        0.725: [(make_rotating_bar,), (make_open_button,)],
    }
    if door_type != HingeDoor or door_mechanics_params["is_pullable"] == False:
        # Sliding bars cannot be used with pullable doors
        variants_by_difficulty.update(
            {
                0.375: [(make_sliding_bar, dict(slot=MountSlot.BOTTOM))],
                0.500: [(make_sliding_bar, dict(slot=MountSlot.TOP))],
                0.750: [
                    (make_sliding_bar, dict(slot=MountSlot.BOTTOM)),
                    (make_sliding_bar, dict(slot=MountSlot.TOP)),
                ],
                0.800: [(make_rotating_bar,), (make_sliding_bar, dict(slot=MountSlot.BOTTOM))],
                0.925: [
                    (make_rotating_bar,),
                    (make_sliding_bar, dict(slot=MountSlot.BOTTOM)),
                    (make_open_button,),
                ],
            }
        )

    lock_difficulty_distribution = stats.norm(difficulty, 0.125)
    difficulty_weights = np.array([lock_difficulty_distribution.pdf(x) for x in variants_by_difficulty.keys()])
    difficulty_weights /= difficulty_weights.sum()
    variant_indices = range(len(list(variants_by_difficulty.items())))
    variant_index = rand.choice(variant_indices, p=difficulty_weights)
    _variant_difficulty, variant_locks = list(variants_by_difficulty.items())[variant_index]

    latching_mechanics_by_difficulty = {
        0.5: LatchingMechanics.LATCH_ONCE,
        0.75: LatchingMechanics.AUTO_LATCH,
    }
    if len(variant_locks) == 0:
        # Only unlocked doors may be non-latched
        latching_mechanics_by_difficulty[0.25] = LatchingMechanics.NO_LATCH

    latching_mechanics_difficulty_distribution = stats.norm(latching_difficulty, 0.125)
    difficulty_weights = np.array(
        [latching_mechanics_difficulty_distribution.pdf(x) for x in latching_mechanics_by_difficulty.keys()]
    )
    difficulty_weights /= difficulty_weights.sum()
    latching_mechanics = rand.choice(list(latching_mechanics_by_difficulty.values()), p=difficulty_weights)  # type: ignore[arg-type]

    return door_type, door_mechanics_params, latching_mechanics, variant_locks


def get_side_tiles(tile: BuildingTile, azimuth: Azimuth) -> Tuple[BuildingTile, BuildingTile]:
    if azimuth == Azimuth.NORTH:
        return BuildingTile(tile.x - 1, tile.z), BuildingTile(tile.x + 1, tile.z)
    elif azimuth == Azimuth.EAST:
        return BuildingTile(tile.x, tile.z - 1), BuildingTile(tile.x + 1, tile.z + 1)
    elif azimuth == Azimuth.SOUTH:
        return BuildingTile(tile.x + 1, tile.z), BuildingTile(tile.x - 1, tile.z)
    elif azimuth == Azimuth.WEST:
        return BuildingTile(tile.x, tile.z + 1), BuildingTile(tile.x, tile.z - 1)
    else:
        raise SwitchError(azimuth)


def make_sliding_door(
    story: Story,
    door_tile: BuildingTile,
    door_azimuth: Azimuth,
    door_face_axis: Axis,
    slide_right: bool = True,
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH,
) -> SlidingDoor:
    tile_width = 1
    door_width = 1
    positive_floor_depth = story.floor_heightmap[door_tile.z, door_tile.x]
    extra_positive_floor_depth = positive_floor_depth - DEFAULT_FLOOR_THICKNESS
    door_height = (story.inner_height - extra_positive_floor_depth) / 1.1
    door_thickness = 0.1
    door_wall_gap = 0.0

    if door_azimuth == Azimuth.NORTH:
        door_centroid_2d = (door_tile.x + door_width / 2, door_tile.z + door_thickness / 2 + door_wall_gap)
        door_rotation_degrees = 0
    elif door_azimuth == Azimuth.EAST:
        door_centroid_2d = (
            door_tile.x + tile_width - door_thickness / 2 - door_wall_gap,
            door_tile.z + door_width / 2,
        )
        door_rotation_degrees = -90
    elif door_azimuth == Azimuth.SOUTH:
        door_centroid_2d = (
            door_tile.x + door_width / 2,
            door_tile.z + tile_width - door_thickness / 2 - door_wall_gap,
        )
        door_rotation_degrees = 180
    elif door_azimuth == Azimuth.WEST:
        door_centroid_2d = (door_tile.x + door_thickness / 2 + door_wall_gap, door_tile.z + door_width / 2)
        door_rotation_degrees = 90
    else:
        raise SwitchError(door_azimuth)

    side_tiles = get_side_tiles(door_tile, door_azimuth)
    free_tile_on_left = story.get_room_at_point(side_tiles[0]) is not None
    free_tile_on_right = story.get_room_at_point(side_tiles[1]) is not None
    if not free_tile_on_right and not free_tile_on_left:
        raise ImpossibleWorldError("Can't place sliding door - no free space on either side")

    if slide_right and not free_tile_on_right:
        slide_right = False
    elif not free_tile_on_left:
        slide_right = True

    door_floor_gap = 0.1
    door_location = np.array(
        [
            door_centroid_2d[0],
            story.floor_negative_depth + door_height / 2 + door_floor_gap,
            door_centroid_2d[1],
        ]
    )
    door_rotation = cast(
        Basis3DNP, Rotation.from_euler("y", door_rotation_degrees, degrees=True).as_matrix().flatten()
    )
    return SlidingDoor(
        position=door_location,
        size=np.array([door_width, door_height, door_thickness]),
        rotation=door_rotation,
        face_axis=door_face_axis,
        slide_right=slide_right,
        latching_mechanics=latching_mechanics,
    )


def _get_entrance_door_centroid(
    story: Story,
    entrance: Entrance,
    door_size: Point3DNP,
    door_wall_gap: float,
    door_floor_gap: float,
    is_inside_tile: bool,
) -> Point3DNP:
    door_width, door_height, door_thickness = door_size
    outline_position = only(entrance.get_points_in_story_outline(story))
    outline_tile = BuildingTile(z=outline_position[0], x=outline_position[1])

    inset_multiplier = 1 if is_inside_tile else -1

    if entrance.azimuth == Azimuth.NORTH:
        door_centroid_2d = (
            outline_tile.x + door_width / 2,
            outline_tile.z + inset_multiplier * (door_thickness / 2 + door_wall_gap),
        )
    elif entrance.azimuth == Azimuth.EAST:
        door_centroid_2d = (
            outline_tile.x + TILE_SIZE - inset_multiplier * (door_thickness / 2 + door_wall_gap),
            outline_tile.z + door_width / 2,
        )
    elif entrance.azimuth == Azimuth.SOUTH:
        door_centroid_2d = (
            outline_tile.x + door_width / 2,
            outline_tile.z + TILE_SIZE - inset_multiplier * (door_thickness / 2 + door_wall_gap),
        )
    elif entrance.azimuth == Azimuth.WEST:
        door_centroid_2d = (
            outline_tile.x + inset_multiplier * (door_thickness / 2 - door_wall_gap),
            outline_tile.z + door_width / 2,
        )
    else:
        raise SwitchError(entrance.azimuth)
    return np.array(
        [
            door_centroid_2d[0],
            door_height / 2 + door_floor_gap,
            door_centroid_2d[1],
        ]
    )


def _get_entrance_door_rotation(entrance: Entrance) -> Basis3DNP:
    if entrance.azimuth == Azimuth.NORTH:
        rotation_degrees = 180
    elif entrance.azimuth == Azimuth.EAST:
        rotation_degrees = 90
    elif entrance.azimuth == Azimuth.SOUTH:
        rotation_degrees = 0
    elif entrance.azimuth == Azimuth.WEST:
        rotation_degrees = -90
    else:
        raise SwitchError(entrance.azimuth)
    return cast(Basis3DNP, Rotation.from_euler("y", rotation_degrees, degrees=True).as_matrix().flatten())


def make_entrance_sliding_door(
    story: Story,
    entrance: Entrance,
    slide_right: bool = True,
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH,
) -> SlidingDoor:
    door_width = 1
    door_height = entrance.height - 1
    door_thickness = 0.1
    door_wall_gap = 0.0
    door_floor_gap = 0.25

    door_face_axis = entrance.azimuth.aligned_axis
    door_size = np.array([door_width, door_height, door_thickness])
    door_centroid = _get_entrance_door_centroid(
        story, entrance, door_size, door_wall_gap, door_floor_gap, is_inside_tile=False
    )
    door_rotation = _get_entrance_door_rotation(entrance)
    return SlidingDoor(
        position=door_centroid,
        size=door_size,
        rotation=door_rotation,
        face_axis=door_face_axis,
        slide_right=slide_right,
        latching_mechanics=latching_mechanics,
    )


def make_hinge_door(
    story: Story,
    door_tile: BuildingTile,
    door_azimuth: Azimuth,
    door_face_axis: Axis,
    hinge_side: HingeSide,
    is_pushable: bool = True,
    is_pullable: bool = True,
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH,
) -> HingeDoor:
    hinge_radius = 0.05
    tile_size = 1
    wall_gap = 0.025
    door_width = tile_size - hinge_radius * 2 - wall_gap
    door_vertical_gap = 0.1
    extra_positive_floor_depth = story.floor_heightmap[door_tile.z, door_tile.x] - DEFAULT_FLOOR_THICKNESS
    door_height = story.inner_height - extra_positive_floor_depth - 2 * door_vertical_gap
    door_thickness = 0.075

    # Align the centroid such that the outer frame of the door aligns with the walls
    door_centroid_2d = [float(door_tile.x), float(door_tile.z)]
    if door_azimuth == Azimuth.NORTH:
        door_rotation_degrees = 0
        door_centroid_2d[0] += tile_size / 2
        door_centroid_2d[1] -= door_thickness / 2
    elif door_azimuth == Azimuth.EAST:
        door_rotation_degrees = -90
        door_centroid_2d[0] += tile_size + door_thickness / 2
        door_centroid_2d[1] += tile_size / 2
    elif door_azimuth == Azimuth.SOUTH:
        door_rotation_degrees = 180
        door_centroid_2d[0] += tile_size / 2
        door_centroid_2d[1] += tile_size + door_thickness / 2
    elif door_azimuth == Azimuth.WEST:
        door_rotation_degrees = 90
        door_centroid_2d[0] -= door_thickness / 2
        door_centroid_2d[1] += tile_size / 2
    else:
        raise SwitchError(door_azimuth)

    door_position = np.array(
        [
            door_centroid_2d[0],
            story.floor_negative_depth + door_height / 2 + door_vertical_gap,
            door_centroid_2d[1],
        ]
    )
    door_rotation = Rotation.from_euler("y", door_rotation_degrees, degrees=True).as_matrix().flatten()
    return HingeDoor(
        position=door_position,
        size=np.array([door_width, door_height, door_thickness]),
        rotation=door_rotation,
        hinge_side=hinge_side,
        hinge_radius=hinge_radius,
        face_axis=door_face_axis,
        latching_mechanics=latching_mechanics,
        max_inwards_angle=90 if is_pushable else 0,
        max_outwards_angle=90 if is_pullable else 0,
    )


def make_entrance_hinge_door(
    story: Story,
    entrance: Entrance,
    hinge_side: HingeSide,
    is_pushable: bool = True,
    is_pullable: bool = True,
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH,
) -> HingeDoor:
    door_wall_gap = 0.0
    door_floor_gap = 0.25

    hinge_radius = 0.05
    wall_gap = 0.025
    door_width = TILE_SIZE - hinge_radius * 2 - wall_gap
    door_height = entrance.height - 1
    door_thickness = 0.075

    door_face_axis = entrance.azimuth.aligned_axis
    door_size = np.array([door_width, door_height, door_thickness])
    door_slot_size = door_size.copy()
    door_slot_size[0] = TILE_SIZE
    door_centroid = _get_entrance_door_centroid(
        story, entrance, door_slot_size, door_wall_gap, door_floor_gap, is_inside_tile=True
    )
    door_rotation = _get_entrance_door_rotation(entrance)
    return HingeDoor(
        door_centroid,
        size=door_size,
        rotation=door_rotation,
        hinge_side=hinge_side,
        hinge_radius=hinge_radius,
        face_axis=door_face_axis,
        latching_mechanics=latching_mechanics,
        max_inwards_angle=90 if is_pushable else 0,
        max_outwards_angle=90 if is_pullable else 0,
    )


def make_open_button(door: Door) -> DoorOpenButton:
    door_width, door_height, door_thickness = door.size
    if isinstance(door, HingeDoor):
        door_width += door.hinge_radius * 2
        multiplier = 1 if door.hinge_side == HingeSide.LEFT else -1
    elif isinstance(door, SlidingDoor):
        multiplier = -1 if door.slide_right else 1
    else:
        raise NotImplementedError(type(door))
    button_width = button_height = 0.2 * door_width
    button_thickness = 0.25
    offset_from_door = 0.25
    button_size = np.array([button_width, button_height, button_thickness])
    button_position = np.array(
        [multiplier * (door_width / 2 + button_width / 2 + offset_from_door), 0, button_thickness / 2]
    )
    return DoorOpenButton(is_dynamic=True, position=button_position, size=button_size)


def make_rotating_bar(door: Door) -> RotatingBar:
    door_width, door_height, door_thickness = door.size
    unlatch_angle = 10
    if isinstance(door, HingeDoor):
        door_width += door.hinge_radius * 2
        if door.max_outwards_angle > 0 and door.max_inwards_angle == 0:
            unlatch_angle = 75
        anchor_side = HingeSide.RIGHT if door.hinge_side == HingeSide.LEFT else HingeSide.LEFT
    elif isinstance(door, SlidingDoor):
        anchor_side = HingeSide.LEFT if door.slide_right else HingeSide.RIGHT
    else:
        raise SwitchError(f"Unknown door type: {door.__class__}")
    bar_width = door_width * 0.75
    bar_height = door_height * 0.0375
    bar_thickness = 0.25
    bar_size = np.array([bar_width, bar_height, bar_thickness])
    bar_position_x = -door_width / 2 - bar_width / 4
    if anchor_side == HingeSide.RIGHT:
        bar_position_x = -bar_position_x
    bar_position = np.array([bar_position_x, door_height / 4, bar_thickness / 2 + door_thickness / 2])
    rotation_axis = Axis.Z if door.face_axis == Axis.X else Axis.X
    return RotatingBar(
        is_dynamic=True,
        position=bar_position,
        size=bar_size,
        rotation_axis=rotation_axis,
        anchor_side=anchor_side,
        unlatch_angle=unlatch_angle,
    )


def make_sliding_bar(door: Door, slot: MountSlot = MountSlot.BOTTOM) -> SlidingBar:
    door_width, door_height, door_thickness = door.size
    if isinstance(door, HingeDoor):
        door_width += door.hinge_radius * 2
        if door.hinge_side == HingeSide.RIGHT:
            x_multiplier = -1
            mount_side = HingeSide.LEFT
        else:
            x_multiplier = 1
            mount_side = HingeSide.RIGHT
    elif isinstance(door, SlidingDoor):
        if door.slide_right:
            x_multiplier = -1
            mount_side = HingeSide.LEFT
        else:
            x_multiplier = 1
            mount_side = HingeSide.RIGHT
    else:
        raise SwitchError(f"Unknown door type {door.__class__}")
    bar_width = door_width * 0.075
    bar_height = door_height * 0.25
    bar_thickness = 0.125
    bar_size = np.array([bar_width, bar_height, bar_thickness])
    if slot == MountSlot.BOTTOM:
        y_multiplier = -1.1
    elif slot == MountSlot.TOP:
        y_multiplier = 1.1 if isinstance(door, HingeDoor) else 1.15  # account for sliding door rail
    else:
        raise SwitchError(slot)

    bar_position = np.array(
        [
            x_multiplier * door_width / 2,
            y_multiplier * (door_height / 2 - bar_height / 2),
            -door_thickness / 2 + bar_thickness / 2,
        ]
    )
    return SlidingBar(
        is_dynamic=True,
        position=bar_position,
        size=bar_size,
        door_face_axis=door.face_axis,
        mount_slot=slot,
        mount_side=mount_side,
    )
