import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict
from typing import Final
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from urllib.request import urlretrieve
from zipfile import ZipFile

import fire

import avalon.install_godot_binary as binary
from avalon.common.utils import AVALON_PACKAGE_DIR
from avalon.contrib.utils import run_local_command
from avalon.datagen.generate_worlds import generate_worlds
from avalon.datagen.godot_env.interactive_godot_process import GODOT_BINARY_PATH
from avalon.datagen.godot_env.interactive_godot_process import GODOT_EDITOR_PATH
from avalon.datagen.godot_env.interactive_godot_process import GODOT_PROJECT_FILE_PATH
from avalon.datagen.world_creation.constants import AvalonTask

# Must be created with `adb shell mkdir -p /sdcard/Android/data/com.godotengine.datagen`
MOCK_OCULUS_EXTERNAL_STORAGE: Final = "/sdcard/Android/data/com.godotengine.datagen/files"

GODOT_SOURCE_PATH: Final = Path(AVALON_PACKAGE_DIR) / "datagen/godot"
ANDROID_CONFIG_PATH: Final = GODOT_SOURCE_PATH / "config.json"
GENERATED_WORLD_PATH: Final = GODOT_SOURCE_PATH / "worlds"
OPENXR_PLUGIN_RELEASE: Final = "https://github.com/GodotVR/godot_openxr/releases/download/1.3.0/godot-openxr.zip"
OPENXR_PLUGIN_PATH: Final = GODOT_SOURCE_PATH / "addons/godot-openxr"


def _regenerate_avalon_words(
    target_path: Path,
    tasks: List[AvalonTask],
    is_generated_as_practice: bool = True,
    is_generating_for_human: bool = True,
    start_seed: int = 10_000,
    number_of_worlds_per_task: int = 10,
    is_verbose: bool = False,
):
    # canonical release num_worlds and seed (50, 0)
    # canonical practice num_worlds and seed (10, 10000)
    return generate_worlds(
        min_difficulty=0.0,
        is_recreating=True,
        num_workers=64,
        base_output_path=target_path,
        tasks=tasks,
        num_worlds_per_task=number_of_worlds_per_task,
        start_seed=start_seed,
        is_practice=is_generated_as_practice,
        is_generating_for_human=is_generating_for_human,
        is_verbose=is_verbose,
    )


def _parse_tasks(task_list: Union[str, Sequence[str]]) -> Iterable[AvalonTask]:
    mapping = {str(task.value).lower(): task for task in AvalonTask}
    if isinstance(task_list, str):
        task_list = task_list.split(",")
    tasks = [t.strip().lower() for t in task_list]

    if "all" in tasks:
        print("Generating all tasks. This might take a bit!")
        yield from [t for t in AvalonTask]
        return

    for task in tasks:
        assert task in mapping, (
            f"Invalid Avalon Task {task}. " f"Allowed options (case insensitive): {', '.join(mapping.keys())}"
        )
        yield mapping[task.lower()]


def _rerun_this_command_with_env(env: Dict[str, str]) -> None:
    _, *args = sys.argv
    new_env = os.environ.copy()
    new_env.update(env)
    os.execve(sys.executable, [sys.executable, "-m", "avalon.for_humans", *args], new_env)


def _edit_project_settings(replacements: Dict[str, str]):
    with open(GODOT_PROJECT_FILE_PATH, "r+") as project_file:
        settings = project_file.read()
        for search_pattern, replacement_pattern in replacements.items():
            settings = re.sub(search_pattern, replacement_pattern, settings, flags=re.MULTILINE)
        project_file.seek(0)
        project_file.write(settings)
        project_file.truncate()


class VRHelperCLI:
    """CLI for human-centric usage of avalon (world generation, editor config, launching the editor)"""

    def available_tasks(self):
        """print available tasks for generation"""
        print("Available Tasks:")
        for task in AvalonTask:
            print(f" * {task.name} ({task})")

    def generate_worlds(
        self,
        tasks: Union[str, Sequence[str]],
        start_seed: int = 10_000,
        worlds_per_task: int = 10,
        delete_existing_worlds: bool = False,
        verbose: bool = False,
    ) -> None:
        """Regenerate canonical avalon worlds and save them to the internal `worlds` directory for inspection & playing by a human.

        Examples:
            generate_worlds all  --delete_existing_worlds
            generate_worlds --tasks="eat,move,navigate"
            generate_worlds EAT,SURVIVE --worlds_per_task=3

        For a complete list of tasks see run `available_tasks`

        :param tasks: List of avalon tasks to generate (case insensitive, comma delimited).
        :param start_seed: Change the world generator random seed.
        :param worlds_per_task: Number of worlds to generate for each task.
        :param delete_existing_worlds: Delete all existing generated worlds before generating new ones.
        :param verbose: Enable verbose logging in world generator.
        """
        hashseed_env = "PYTHONHASHSEED"
        if os.environ.get(hashseed_env, None) is None:
            print(f"Rerunning command with {hashseed_env}=0 so level generation is deterministic\n")
            return _rerun_this_command_with_env({hashseed_env: "0"})

        avalon_tasks = list(_parse_tasks(tasks))

        if delete_existing_worlds:
            shutil.rmtree(GENERATED_WORLD_PATH, ignore_errors=True)

        _regenerate_avalon_words(
            GENERATED_WORLD_PATH,
            avalon_tasks,
            start_seed=start_seed,
            number_of_worlds_per_task=worlds_per_task,
            is_verbose=verbose,
        )

    def print_godot_project_path(self):
        print(Path(GODOT_PROJECT_FILE_PATH).parent)

    def setup_android_export_presets(
        self,
        keystore_path: str,
        keystore_user: str = "androiddebugkey",
        keystore_password: str = "android",
        export_apk_path: Optional[str] = None,
        debug_template_apk_path: Optional[str] = None,
        release_template_apk_path: Optional[str] = None,
        overwrite_apk_templates: bool = False,
    ):
        """Fill in the godot android export_presets.cfg template for running on oculus.

        All APK paths default to avalon/bin/{kind}.apk

        For more details see https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_android.html

        :param keystore_path: Android debug.keystore path
        :param keystore_user: Android debug.keystore user
        :param keystore_password: Android debug.keystore password
        :param export_apk_path: Override export apk path. Default: avalon/bin/avalon.apk
        :param debug_template_apk_path: Override debug apk template path. Default: avalon/bin/debug_template.apk
        :param release_template_apk_path: Override release apk template path. Default: avalon/bin/release_template.apk
        :param overwrite_apk_templates: Overwrite pre-existing APK templates, if any exist
        """
        if export_apk_path is None:
            export_apk_path = f"{AVALON_PACKAGE_DIR}/bin/avalon.apk"
        if debug_template_apk_path is None:
            debug_template_apk_path = f"{AVALON_PACKAGE_DIR}/bin/debug_template.apk"
        if release_template_apk_path is None:
            release_template_apk_path = f"{AVALON_PACKAGE_DIR}/bin/release_template.apk"

        if not Path(debug_template_apk_path).exists() or overwrite_apk_templates:
            print("Downloading Debug APK template")
            urlretrieve("https://mmap.monster/godot/templates/android_debug.apk", debug_template_apk_path)

        if not Path(release_template_apk_path).exists() or overwrite_apk_templates:
            print("Downloading Release APK template")
            urlretrieve("https://mmap.monster/godot/templates/android_release.apk", release_template_apk_path)

        print("Configuring export_presets.cfg")
        project_dir = Path(GODOT_PROJECT_FILE_PATH).parent
        with open(project_dir / "export_presets.template.cfg", "r") as template_file:
            template = template_file.read()
        replacements = {
            "/your/path/avalon.apk": export_apk_path,
            "/your/path/android_debug.apk": debug_template_apk_path,
            "/your/path/android_release.apk": release_template_apk_path,
            "/your/path/android.keystore": keystore_path,
            "yourdebuguser (maybe godot or androiddebugkey)": keystore_user,
            'keystore/debug_password="android"': f'keystore/debug_password="{keystore_password}"',
            'keystore/release_password="android"': f'keystore/release_password="{keystore_password}"',
        }
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)

        with open(project_dir / "export_presets.cfg", "w") as preset_file:
            preset_file.write(template)
        print("Android export configured. You will need to restart the editor to see changes.")

    def remove_openxr_plugin(self):
        """Remove the openxr plugin installed with `install_openxr_plugin` and attempts to clean the project configuration.

        Closing the editor before running ths command is recommended.
        """
        shutil.rmtree(OPENXR_PLUGIN_PATH)
        _edit_project_settings(
            {
                "res://addons/godot-openxr/plugin.cfg": "",
                "res://addons/godot-openxr/config/godot_openxr.gdnlib": "",
            }
        )
        print("OpenXR plugin successfully removed and disabled.")

    def _configure_openxr_plugin(self):
        print(f"Configuring project settings...")
        _find_enabled_plugin_array = r"^(\[editor_plugins\](.|\n)*enabled=PoolStringArray\(.*) \)$"
        _append_openxr_plugin = r'\1"res://addons/godot-openxr/plugin.cfg" )'
        _find_gdnative_singleton_list = r"^(\[gdnative\](.|\n)*singletons=\[.*) \]$"
        _append_openxr_singleton = r'\1"res://addons/godot-openxr/config/godot_openxr.gdnlib" ]'
        _edit_project_settings(
            {
                _find_enabled_plugin_array: _append_openxr_plugin,
                _find_gdnative_singleton_list: _append_openxr_singleton,
            }
        )

    def install_openxr_plugin(self, overwrite: bool = False):
        """Installs and enables Godot's OpenXR plugin for VR support.

        Closing the editor before running ths command is recommended.

        The extension can be toggled in the editor under Project > Project Settings > Plugins,
        and can be removed with `remove_openxr_plugin`

        For more details see the godot docs at https://docs.godotengine.org/en/stable/tutorials/vr/openxr
        and our running in VR guide at https://github.com/Avalon-Benchmark/avalon/docs/running_in_vr.md#installing-openxr

        :param overwrite: Overwrite any existing installation
        """
        if OPENXR_PLUGIN_PATH.exists():
            if not overwrite:
                print(
                    "Cancelling: Already installed. Pass --overwrite if you want to overwrite the existing installation"
                )
                exit(0)
            shutil.rmtree(OPENXR_PLUGIN_PATH)

        print(f"Installing openxr plugin to {OPENXR_PLUGIN_PATH}")
        addon_path = OPENXR_PLUGIN_PATH.parent
        addon_path.mkdir(parents=False, exist_ok=True)
        archive_path = addon_path / "godot-openxr.zip"

        print(f"Downloading {OPENXR_PLUGIN_RELEASE} into addons dir...")
        urlretrieve(OPENXR_PLUGIN_RELEASE, archive_path)

        print(f"Extracting archive...")
        with ZipFile(archive_path, "r") as archive:
            archive.extractall(addon_path)

        print(f"Cleaning up...")
        os.remove(archive_path)
        shutil.move(addon_path / "godot_openxr_1.3.0/addons/godot-openxr", OPENXR_PLUGIN_PATH)
        shutil.rmtree(OPENXR_PLUGIN_PATH / "assets")
        shutil.rmtree(OPENXR_PLUGIN_PATH / "scenes")

        self._configure_openxr_plugin()

        print("OpenXR plugin successfully installed")

    def launch_editor(self, verbose: bool = False):
        """Launch the godot editor installed with `python -m avalon.install_godot_binary`

        :param verbose: Enable godot's verbose mode
        """
        if not os.path.exists(GODOT_EDITOR_PATH):
            assert binary.EDITOR in binary.available_builds(), "No editor available for this platform."
            install = "python -m avalon.install_godot_binary"
            print(f"Warning: no file found in GODOT_EDITOR_PATH. Attempting to run `{install}`")
            run_local_command(install)

        run_local_command(f"{GODOT_EDITOR_PATH} --editor{' --verbose' if verbose else ''} {GODOT_PROJECT_FILE_PATH}")


if __name__ == "__main__":
    fire.Fire(VRHelperCLI)
