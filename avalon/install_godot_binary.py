#!/usr/bin/env python
import os
import stat
import sys
import urllib.request
from pathlib import Path
from typing import Dict
from typing import Final
from typing import List
from typing import Literal
from typing import Sequence
from typing import Tuple
from typing import cast
from zipfile import ZipFile

from avalon.datagen.godot_env.interactive_godot_process import GODOT_BINARY_PATH

RELEASES: Final = "https://github.com/Avalon-Benchmark/godot/releases/download"

CURRENT_RELEASE: Final = "3.4.4.avalon.0.9.1"


Platform = Literal["linux", "macos", "windows"]

BinaryType = Literal["runner", "editor"]

RUNNER: Final = "runner"
EDITOR: Final = "editor"

CWD: Final = os.getcwd()


def _friendly_path(path: Path) -> str:
    abs_path = str(path.resolve())
    try:
        relative = str(path.relative_to(CWD))
        if len(relative) < len(abs_path):
            return relative
    except ValueError:
        pass
    return abs_path


def get_platform() -> Platform:
    p = sys.platform
    if p == "linux":
        return "linux"
    if p == "darwin":
        return "macos"
    if p in ("win32", "cygwin"):
        return "windows"
    raise ValueError(f"No avalon godot binaries available for {p}")


builds: Final[Dict[Tuple[Platform, BinaryType], str]] = {
    ("linux", EDITOR): "linux-egl-editor.zip",
    ("linux", RUNNER): "linux-egl-runner.zip",
    ("macos", EDITOR): "macos-editor.zip",
}

def available_builds() -> List[BinaryType]:
    this_platform = get_platform()
    return [build for plat, build in builds.keys() if plat == this_platform]




def get_godot_binary_path() -> Path:
    path = Path(GODOT_BINARY_PATH)
    path.parent.mkdir(exist_ok=True)
    return path


def _extract_singular_as(zipped: Path, target: Path, is_symlinked: bool = False):
    with ZipFile(zipped, "r") as archive:
        members = archive.namelist()
        assert len(members) == 1, "Should only contain a single executable binary"
        archive.extract(members[0], target.parent)
        extracted_destination = target.parent / members[0]
    if not is_symlinked:
        os.rename(extracted_destination, target)
        return

    print(f"Linking {_friendly_path(extracted_destination)} to {_friendly_path(target)}")
    if target.exists():
        os.remove(target)
    os.symlink(extracted_destination, target)


def fetch_binary(build: str, target: Path):
    archive_path = target.parent / build
    print(archive_path)
    release = f"{CURRENT_RELEASE}/{build}"
    print(f"Installing {release} into {_friendly_path(archive_path.parent)} from {RELEASES}")
    urllib.request.urlretrieve(f"{RELEASES}/{release}", archive_path)
    _extract_singular_as(archive_path, target, is_symlinked=True)
    os.remove(archive_path)


def ensure_executable(path: Path):
    current_permissions_plus_executable = os.stat(path).st_mode | stat.S_IEXEC
    os.chmod(path, current_permissions_plus_executable)


def install_godot_binary(binary_type: BinaryType):
    binary_path = get_godot_binary_path()
    if binary_path.is_symlink():
        print("replacing current symlink")
        os.remove(binary_path)
    assert not binary_path.exists(), f"Refusing to overwrite existing file {binary_path}"

    key = (get_platform(), binary_type)
    assert key in builds, f"{binary_type} is an invalid binary type for {key[0]} (options: {list(builds.keys())})"

    fetch_binary(builds[key], binary_path)
    ensure_executable(binary_path)


def fallback_interactive(available: Sequence[BinaryType]) -> BinaryType:
    if len(available) == 0:
        print(f"Running on {get_platform()} is not supported")
        exit(1)
    if len(available) == 1:
        prompt = f"Only available binary for your platform is {available[0]}. Continue (y/N)?\n> "
        is_proceeding = input(prompt).lower().startswith("y")
        if not is_proceeding:
            print("Install cancelled.")
            exit(0)
        return available[0]
    else:
        prompt = f"Please specify one of the available binaries for your platform ({' or '.join(available)}):\n> "
        binary_type = input(prompt).lower().strip()
        if binary_type not in available:
            print(f"Install cancelled: {binary_type} is an invalid option")
            exit(0)
        return cast(BinaryType, binary_type)


if __name__ == "__main__":
    binary_type = sys.argv[1] if len(sys.argv) > 1 else ""
    available = available_builds()
    if binary_type not in available:
        binary_type = fallback_interactive(available)
    install_godot_binary(cast(BinaryType, binary_type))
