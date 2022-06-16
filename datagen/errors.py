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
