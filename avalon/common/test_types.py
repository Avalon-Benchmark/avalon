import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from avalon.contrib.testing_utils import create_temp_file_path
from avalon.contrib.testing_utils import temp_file_path_
from avalon.contrib.testing_utils import use
from avalon.contrib.utils import TESTS_FOLDER

MYPY_CACHE_NAME = ".mypy_cache"


class MYPYFailure(Exception):
    pass


def get_path_to_root(dir: str) -> Path:
    current_dir = Path(os.getcwd())
    while str(current_dir) != "/":
        if os.path.exists(str(current_dir / dir)):
            return current_dir / dir
        current_dir = current_dir.parent
    raise AssertionError(f"Could not find directory that houses {dir}")


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
            for line in f.readlines():
                line = line.strip()
                if "*" in line:
                    excluded_files |= {str(path).strip() for path in Path(".").glob(line)}
                else:
                    excluded_files.add(line)

    covered_paths = ["avalon", "notebooks"]
    all_python_files = []
    for path in covered_paths:
        all_python_files.extend([str(file) for file in Path(path).rglob("*.py") if str(file) not in excluded_files])

    items_to_check = " ".join(all_python_files)
    subprocess.run(
        f"bash -c 'bash ./avalon/common/run_mypy.sh \"{items_to_check}\" >& {str(temp_file_path)}'",
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
