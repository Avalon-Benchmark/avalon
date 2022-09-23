from avalon.common.errors import SwitchError
from avalon.datagen.world_creation.geometry import BuildingTile


def rotate_position(
    position: BuildingTile, story_width: int, story_length: int, degrees: int, tile_like: bool = True
) -> BuildingTile:
    # tile_like=True behaves like np.rot90; tile_like=False rotates directly
    if degrees == 0:
        new_x = position.x
        new_z = position.z
    elif degrees == 90:
        new_x = story_length - position.z
        if tile_like:
            new_x -= 1
        new_z = position.x
    elif degrees == -90:
        new_x = position.z
        new_z = story_width - position.x
        if tile_like:
            new_z -= 1
    elif degrees == 180:
        new_x = story_width - position.x
        new_z = story_length - position.z
        if tile_like:
            new_x -= 1
            new_z -= 1
    else:
        raise SwitchError(degrees)
    return BuildingTile(new_x, new_z)
