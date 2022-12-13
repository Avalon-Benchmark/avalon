import math
from enum import Enum
from typing import Dict
from typing import Final
from typing import List
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np


class AvalonTask(Enum):
    # ONE
    EAT = "EAT"
    MOVE = "MOVE"
    # TWO
    JUMP = "JUMP"
    EXPLORE = "EXPLORE"
    SCRAMBLE = "SCRAMBLE"
    # THREE
    CLIMB = "CLIMB"
    DESCEND = "DESCEND"
    THROW = "THROW"
    AVOID = "AVOID"
    # FOUR
    HUNT = "HUNT"
    FIGHT = "FIGHT"
    PUSH = "PUSH"
    STACK = "STACK"
    BRIDGE = "BRIDGE"
    OPEN = "OPEN"
    CARRY = "CARRY"
    # COMPOSITIONAL
    NAVIGATE = "NAVIGATE"
    FIND = "FIND"
    GATHER = "GATHER"
    SURVIVE = "SURVIVE"


avalon_task_to_int = {member.value: i for i, member in enumerate(AvalonTask)}
int_to_avalon_task = {i: member.value for i, member in enumerate(AvalonTask)}


class AvalonTaskGroup(Enum):
    ONE = "ONE"
    TWO = "TWO"
    THREE = "THREE"
    FOUR = "FOUR"
    SIMPLE = "SIMPLE"
    EASY = "EASY"
    COMPOSITIONAL = "COMPOSITIONAL"
    ALL = "ALL"
    # ONE
    EAT = "EAT"
    MOVE = "MOVE"
    # TWO
    JUMP = "JUMP"
    EXPLORE = "EXPLORE"
    SCRAMBLE = "SCRAMBLE"
    # THREE
    CLIMB = "CLIMB"
    DESCEND = "DESCEND"
    THROW = "THROW"
    AVOID = "AVOID"
    # FOUR
    HUNT = "HUNT"
    FIGHT = "FIGHT"
    PUSH = "PUSH"
    STACK = "STACK"
    BRIDGE = "BRIDGE"
    OPEN = "OPEN"
    CARRY = "CARRY"
    # COMPOSITIONAL
    NAVIGATE = "NAVIGATE"
    FIND = "FIND"
    GATHER = "GATHER"
    SURVIVE = "SURVIVE"


TRAIN_TASK_GROUPS = (
    AvalonTaskGroup.ONE,
    AvalonTaskGroup.TWO,
    AvalonTaskGroup.THREE,
    AvalonTaskGroup.FOUR,
    AvalonTaskGroup.COMPOSITIONAL,
)

SINGLE_TASK_GROUPS = (
    AvalonTaskGroup.EAT,
    AvalonTaskGroup.MOVE,
    AvalonTaskGroup.JUMP,
    AvalonTaskGroup.CLIMB,
    AvalonTaskGroup.DESCEND,
    AvalonTaskGroup.SCRAMBLE,
    AvalonTaskGroup.STACK,
    AvalonTaskGroup.BRIDGE,
    AvalonTaskGroup.PUSH,
    AvalonTaskGroup.THROW,
    AvalonTaskGroup.HUNT,
    AvalonTaskGroup.FIGHT,
    AvalonTaskGroup.AVOID,
    AvalonTaskGroup.EXPLORE,
    AvalonTaskGroup.OPEN,
    AvalonTaskGroup.CARRY,
    AvalonTaskGroup.NAVIGATE,
    AvalonTaskGroup.FIND,
    AvalonTaskGroup.SURVIVE,
    AvalonTaskGroup.GATHER,
)

TASKS_BY_TASK_GROUP: Dict[AvalonTaskGroup, Tuple[AvalonTask, ...]] = {
    AvalonTaskGroup.ONE: (AvalonTask.EAT, AvalonTask.MOVE),
    AvalonTaskGroup.TWO: (AvalonTask.JUMP, AvalonTask.EXPLORE, AvalonTask.SCRAMBLE),
    AvalonTaskGroup.THREE: (AvalonTask.CLIMB, AvalonTask.DESCEND, AvalonTask.THROW, AvalonTask.AVOID),
    AvalonTaskGroup.FOUR: (
        AvalonTask.HUNT,
        AvalonTask.FIGHT,
        AvalonTask.PUSH,
        AvalonTask.STACK,
        AvalonTask.BRIDGE,
        AvalonTask.OPEN,
        AvalonTask.CARRY,
    ),
    AvalonTaskGroup.COMPOSITIONAL: (AvalonTask.NAVIGATE, AvalonTask.FIND, AvalonTask.GATHER, AvalonTask.SURVIVE),
    AvalonTaskGroup.EASY: (
        AvalonTask.EAT,
        AvalonTask.MOVE,
    ),
    AvalonTaskGroup.SIMPLE: (
        AvalonTask.EAT,
        AvalonTask.MOVE,
        AvalonTask.JUMP,
        AvalonTask.CLIMB,
        AvalonTask.DESCEND,
        AvalonTask.SCRAMBLE,
        AvalonTask.STACK,
        AvalonTask.BRIDGE,
        AvalonTask.PUSH,
        AvalonTask.THROW,
        AvalonTask.HUNT,
        AvalonTask.FIGHT,
        AvalonTask.AVOID,
        AvalonTask.EXPLORE,
        AvalonTask.OPEN,
        AvalonTask.CARRY,
    ),
    AvalonTaskGroup.ALL: (
        AvalonTask.EAT,
        AvalonTask.MOVE,
        AvalonTask.JUMP,
        AvalonTask.CLIMB,
        AvalonTask.DESCEND,
        AvalonTask.SCRAMBLE,
        AvalonTask.STACK,
        AvalonTask.BRIDGE,
        AvalonTask.PUSH,
        AvalonTask.THROW,
        AvalonTask.HUNT,
        AvalonTask.FIGHT,
        AvalonTask.AVOID,
        AvalonTask.EXPLORE,
        AvalonTask.OPEN,
        AvalonTask.CARRY,
        AvalonTask.NAVIGATE,
        AvalonTask.FIND,
        AvalonTask.SURVIVE,
        AvalonTask.GATHER,
    ),
    AvalonTaskGroup.EAT: (AvalonTask.EAT,),
    AvalonTaskGroup.MOVE: (AvalonTask.MOVE,),
    AvalonTaskGroup.JUMP: (AvalonTask.JUMP,),
    AvalonTaskGroup.CLIMB: (AvalonTask.CLIMB,),
    AvalonTaskGroup.DESCEND: (AvalonTask.DESCEND,),
    AvalonTaskGroup.SCRAMBLE: (AvalonTask.SCRAMBLE,),
    AvalonTaskGroup.STACK: (AvalonTask.STACK,),
    AvalonTaskGroup.BRIDGE: (AvalonTask.BRIDGE,),
    AvalonTaskGroup.PUSH: (AvalonTask.PUSH,),
    AvalonTaskGroup.THROW: (AvalonTask.THROW,),
    AvalonTaskGroup.HUNT: (AvalonTask.HUNT,),
    AvalonTaskGroup.FIGHT: (AvalonTask.FIGHT,),
    AvalonTaskGroup.AVOID: (AvalonTask.AVOID,),
    AvalonTaskGroup.EXPLORE: (AvalonTask.EXPLORE,),
    AvalonTaskGroup.OPEN: (AvalonTask.OPEN,),
    AvalonTaskGroup.CARRY: (AvalonTask.CARRY,),
    AvalonTaskGroup.NAVIGATE: (AvalonTask.NAVIGATE,),
    AvalonTaskGroup.FIND: (AvalonTask.FIND,),
    AvalonTaskGroup.SURVIVE: (AvalonTask.SURVIVE,),
    AvalonTaskGroup.GATHER: (AvalonTask.GATHER,),
}

AGENT_HEIGHT = 2.0
HALF_AGENT_HEIGHT_VECTOR = np.array([0, AGENT_HEIGHT / 2.0, 0])

FOOD_HOVER_DIST = 0.5

MAX_FLAT_JUMP_METERS = 3
MAX_JUMP_HEIGHT_METERS = 1.6
MAX_MOVE_DIST = 10
MAX_REACH_DIST = 0.3
STANDING_REACH_HEIGHT = AGENT_HEIGHT + MAX_REACH_DIST

# minimum height that prevents you from escaping without climbinb
JUMPING_REQUIRED_HEIGHT = 2.0
CLIMBING_REQUIRED_HEIGHT = 3.2

ITEM_FLATTEN_RADIUS = 2.0

# how big is the stone for stacking
BOX_HEIGHT = 1.0

# how big is the boulder for pushing
BOULDER_HEIGHT = 1.4
BOULDER_MIN_MASS = 600.0
BOULDER_MAX_MASS = 2400.0

# controls how much space to reserve around the spawn and goal points
DEFAULT_SAFETY_RADIUS = 0.5

# how far you can jump in a totally flat world
MAX_FLAT_JUMP_DIST = 2.3
# how far you can effectively jump given polygonization
MAX_EFFECTIVE_JUMP_DIST = 2.1

MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE = 10.0
MAX_FALL_DISTANCE_TO_DIE = 30.0

# min dist that requires a bridge
MIN_BRIDGE_DIST = MAX_FLAT_JUMP_DIST + 0.25

# length of the bridge object
BRIDGE_LENGTH = 6.4
MAX_BRIDGE_DIST = BRIDGE_LENGTH - 2

# how far off the ground to spawn weapons. Not really sure what this should be, needs to work for stick and rock...
WEAPON_HEIGHT_OFFSET = 1.0

# standard deviation for when we're trying to tightly specify a distance for a goal
TIGHT_DIST_STD_DEV = 0.5


# any tree within this many meters of your line of sight will be deleted
# (only applies to tasks that require being able to see the goal)
METERS_OF_TREE_CLEARANCE_AROUND_LINE_OF_SIGHT = 5.0

# do not spawn ground cover within this radius of important points
FLORA_REMOVAL_METERS = 1.0


def get_all_tasks_for_task_groups(task_groups: Sequence[AvalonTaskGroup]) -> List[AvalonTask]:
    result: Set[AvalonTask] = set()
    for task_group in task_groups:
        result.update(TASKS_BY_TASK_GROUP[task_group])
    return sorted(result, key=lambda x: x.value)


STARTING_HIT_POINTS: float = 1.0

# for performance experiments
IS_ALL_SCENERY_IN_MAIN = False
FLORA_RADIUS_IN_TILES = 1

IS_DEBUGGING_IMPOSSIBLE_WORLDS = False
WATER_LINE = 0.0
UNCLIMBABLE_BIOME_ID = 19
DARK_SHORE = UNCLIMBABLE_BIOME_ID + 1
FRESH_WATER = DARK_SHORE + 1
SWAMP = FRESH_WATER + 1
MIN_CLIFF_TERRAIN_ANGLE = (math.pi / 2) * 0.4
MAX_CLIFF_TERRAIN_ANGLE = (math.pi / 2) * 0.7
WORLD_RAISE_AMOUNT = 10_000
IDENTITY_BASIS: Final = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
UP_VECTOR = np.array([0.0, 1.0, 0.0])
