import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from contrib.testing_utils import create_temp_file_path
from contrib.testing_utils import temp_file_path_
from contrib.testing_utils import use
from contrib.utils import TESTS_FOLDER

MYPY_CACHE_NAME = ".mypy_cache"


class MYPYFailure(Exception):
    pass


def get_path_to_root(dir: str) -> Path:
    current_dir = Path(os.getcwd())
    while str(current_dir) != "/":
        if os.path.exists(str(current_dir / dir)):
            return current_dir / dir
        current_dir = current_dir.parent
    raise AssertionError("Could not find the computronium directory")


@use(temp_file_path_)
def test_mypy(temp_file_path: Path):
    shared_mypy_cache_path = Path(os.path.join(TESTS_FOLDER, MYPY_CACHE_NAME))
    local_mypy_cache_path = Path(".") / MYPY_CACHE_NAME
    if shared_mypy_cache_path.exists() and not local_mypy_cache_path.exists():
        shutil.copytree(shared_mypy_cache_path, local_mypy_cache_path)

    excluded_path = Path(".") / "excluded.txt"
    excluded_files = set()
    if os.path.exists(str(excluded_path)):
        with open(excluded_path, "r") as f:
            excluded_files = set([line.strip() for line in f.readlines()])

    all_python_files = [
        str(x)
        for x in Path(".").rglob("*.py")
        if str(x) not in excluded_files and "quarantine/" not in str(x) and "contrib/" not in str(x)
    ]
    if not os.path.exists(str(Path(".") / "computronium")):
        all_python_files += [str(x) for x in get_path_to_root("computronium").rglob("*.py")]
    if not os.path.exists(str(Path(".") / "bones")):
        all_python_files += [
            str(x) for x in get_path_to_root("bones").rglob("*.py") if not str(x).endswith("setup.py")
        ]
    if not os.path.exists(str(Path(".") / "science")):
        # note: we copy conftest.py in the root of science/ and the standalone project dir and we only want to include 1
        all_python_files += [
            str(x) for x in get_path_to_root("science").rglob("*.py") if not str(x).endswith("science/conftest.py")
        ]

    items_to_check = " ".join(all_python_files)
    subprocess.run(
        f"bash -c 'bash ./common/run_mypy.sh \"{items_to_check}\" >& {str(temp_file_path)}'",
        shell=True,
        check=True,
    )

    if not shared_mypy_cache_path.exists():
        shared_mypy_cache_path.mkdir(parents=True, exist_ok=True)

    # removes old cache and copies the newly created cache
    shutil.rmtree(shared_mypy_cache_path)
    shutil.copytree(local_mypy_cache_path, shared_mypy_cache_path)
    assert shared_mypy_cache_path.exists()

    with open(str(temp_file_path), "r") as infile:
        lines = infile.readlines()[:-1]
        if len([x for x in lines if x.strip()]) > 0:
            logger.info("Typing errors:")
            logger.info("".join(lines))
            raise MYPYFailure("mypy type checking failed")


if __name__ == "__main__":
    try:
        with create_temp_file_path() as temp_file:
            test_mypy(temp_file)
    except MYPYFailure:
        sys.exit(1)
