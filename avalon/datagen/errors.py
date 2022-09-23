from avalon.datagen.world_creation.constants import AvalonTask


class DatagenConfigError(Exception):
    """
    Raised when datagen configuration values are impossible or incoherent.
    Ideally we keep the usage of this class to a minimum and use the type signatures to verify, but some
    combinations of invalid parameter values are hard to specify via the types.
    """


class GodotError(Exception):
    """
    Raised when the godot binary fails for any reason.
    """


class DesynchronizedTransformError(Exception):
    """
    Raised when your transforms are not perfectly synchronous.
    Is here to help keep you from going insane if you were to accidentally apply different transforms to
    data that really needs to stay synced (eg, depth and rgb)
    """


class ImpossibleWorldError(Exception):
    pass


class WorldTooSmall(ImpossibleWorldError):
    def __init__(self, task: AvalonTask, min_dist: float, available_dist: float, *args: object) -> None:
        super().__init__(f"Small {task} world: needed {min_dist} but only have {available_dist}", *args)
