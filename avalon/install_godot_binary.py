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
from typing import Tuple
from zipfile import ZipFile

from loguru import logger

from avalon.datagen.godot_env.interactive_godot_process import GODOT_BINARY_PATH
from avalon.datagen.godot_env.interactive_godot_process import GODOT_EDITOR_PATH

RELEASES: Final = "https://github.com/Avalon-Benchmark/godot/releases/download"

CURRENT_RELEASE: Final = "3.4.4.avalon.0.9.3"

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
    ("linux", RUNNER): "linux-egl-editor.zip",
    ("linux", EDITOR): "linux-x11-editor.zip",
    ("macos", EDITOR): "macos-editor.zip",
    ("windows", EDITOR): "windows-editor.zip",
}


def available_builds() -> List[BinaryType]:
    this_platform = get_platform()
    return [build for plat, build in builds.keys() if plat == this_platform]


def get_godot_binary_path() -> Path:
    path = Path(GODOT_BINARY_PATH)
    path.parent.mkdir(exist_ok=True)
    return path


def get_godot_editor_path() -> Path:
    path = Path(GODOT_EDITOR_PATH)
    path.parent.mkdir(exist_ok=True)
    return path


def _extract_singular_as(zipped: Path, target: Path) -> None:
    with ZipFile(zipped, "r") as archive:
        members = archive.namelist()
        assert len(members) == 1, "Should only contain a single executable binary"
        archive.extract(members[0], target.parent)
        extracted_destination = target.parent / members[0]
    os.rename(extracted_destination, target)


def fetch_binary(build: str, target: Path) -> None:
    archive_path = target.parent / build
    release = f"{CURRENT_RELEASE}/{build}"
    logger.info(f"Downloading {release} into {_friendly_path(target)}")
    urllib.request.urlretrieve(f"{RELEASES}/{release}", archive_path)
    _extract_singular_as(archive_path, target)
    os.remove(archive_path)


def ensure_executable(path: Path) -> None:
    current_permissions_plus_executable = os.stat(path).st_mode | stat.S_IEXEC
    os.chmod(path, current_permissions_plus_executable)


def handle_overwrite(target_path: Path, is_overwriting: bool) -> None:
    if target_path.exists() or target_path.is_symlink():
        if not is_overwriting:
            logger.error(f"Refusing to overwrite existing file {target_path} without --overwrite")
            exit(1)
        os.remove(target_path)


def install_binary(build: str, target_path: Path, is_overwriting: bool) -> None:
    handle_overwrite(target_path, is_overwriting)
    if target_path.exists():
        if not is_overwriting:
            logger.error(f"Refusing to overwrite existing file {target_path} without --overwrite")
            exit(1)
        os.remove(target_path)
    fetch_binary(build, target_path)
    ensure_executable(target_path)


def install_available_binaries_for_current_platform(is_overwriting: bool) -> None:
    platform = get_platform()
    binary_path = get_godot_binary_path()
    editor_path = get_godot_editor_path()

    editor_build = builds.get((platform, EDITOR), None)
    headless_build = builds.get((platform, RUNNER), None)
    assert editor_build is not None, f"{platform} is currently unsupported"
    logger.info(
        f"Installing editor {'build' if headless_build is None else 'and headless runner builds'} from {RELEASES}"
    )

    install_binary(editor_build, editor_path, is_overwriting)

    if headless_build is not None:
        install_binary(headless_build, binary_path, is_overwriting)
    elif platform == "windows":
        logger.info(
            f"Note: Installed godot editor for windows, but it cannot be used as a runner. "
            f"Consider using WSL for actual training purposes: https://learn.microsoft.com/en-us/windows/wsl/install"
        )
    else:
        logger.info(
            f"Note: No headless runner currently available for {platform}. Symlinking editor build in it's place."
        )
        handle_overwrite(binary_path, is_overwriting)
        os.symlink(editor_path, binary_path, is_overwriting)


if __name__ == "__main__":
    is_overwriting = len(sys.argv) > 1 and sys.argv[1] == "--overwrite"
    install_available_binaries_for_current_platform(is_overwriting)
