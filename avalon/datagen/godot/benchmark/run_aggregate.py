#!/usr/bin/env python3

import pathlib
import statistics
from typing import List
from typing import Tuple

CONFIGS = ["basic", "fancy"]
DRIVERS = ["GLES2", "GLES3"]
NPROCS = ["1", "2", "3", "4", "5"]
WORLDS = {
    "032": "small",
    "064": "medium",
    "220": "large",
    "440": "huge",
}


def get_fps_lines_from_log(config: str, driver: str, world: str, process_count: int, process_id: int) -> List[int]:
    log_path = pathlib.Path(f"./{config}.{driver}.{process_count}.{world}/log.{process_id}.txt")
    fps_lines = [x for x in log_path.read_text().split("\n") if x.startswith("Project FPS")]
    fps_counts = [int(x.split()[2]) for x in fps_lines]
    return fps_counts


def get_performance(config: str, driver: str, world: str) -> Tuple[int, int, int]:
    single_process_fps = max(get_fps_lines_from_log(config, driver, world, 1, 1))
    multi_process_fps_list = []
    multi_process_num_list = []
    for i in range(2, 6):
        fps_means = []
        for j in range(1, i + 1):
            fps_lines = get_fps_lines_from_log(config, driver, world, i, j)
            fps_lines = fps_lines[2:]
            fps_means.append(statistics.mean(fps_lines))
        multi_process_fps_list.append(int(sum(fps_means)))
        multi_process_num_list.append(i)
    multi_process_fps = max(multi_process_fps_list)
    multi_process_num = multi_process_num_list[multi_process_fps_list.index(multi_process_fps)]
    return single_process_fps, multi_process_fps, multi_process_num


def show(value_id: int) -> None:
    for driver in DRIVERS:
        for config in CONFIGS:
            values = []
            for world_id, world_name in WORLDS.items():
                value = get_performance(config, driver, world_id)[value_id]
                values.append(int(value))
            print(
                f"{driver} ({config}) & {values[0]:8,} & {values[1]:8,} & {values[2]:8,} & {values[3]:8,} \\\\"
            )  # script


def main() -> None:
    print("single:")  # script
    show(0)
    print()  # script
    print("multi:")  # script
    show(1)
    print()  # script
    print("nproc:")  # script
    show(2)


if __name__ == "__main__":
    main()
