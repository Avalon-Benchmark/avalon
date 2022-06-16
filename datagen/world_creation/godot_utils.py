from pathlib import Path
from typing import Final

import numpy as np
from godot_parser import GDObject

IDENTITY_BASIS: Final = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=float)

GODOT_PATH = (Path(__file__) / "../../godot").resolve().as_posix()


def make_transform(rotation=(1, 0, 0, 0, 1, 0, 0, 0, 1), position=(0, 0, 0)):
    return GDObject("Transform", *rotation, *position)


def scale_basis(basis, scale):
    x, y, z = scale
    scaled = np.array(basis)
    scaled[0:3] *= x
    scaled[3:6] *= y
    scaled[6:9] *= z
    return scaled
