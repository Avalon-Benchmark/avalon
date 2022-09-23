from enum import Enum


class HingeSide(Enum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"


class MountSlot(Enum):
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class LatchingMechanics(Enum):
    NO_LATCH = "NO_LATCH"
    LATCH_ONCE = "LATCH_ONCE"
    AUTO_LATCH = "AUTO_LATCH"
