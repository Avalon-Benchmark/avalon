import hashlib
import json
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List

from contrib.utils import run_local_command
from datagen.godot_generated_types import AvalonSimSpec

EDITOR_SETTINGS = """
[gd_resource type="EditorSettings" format=2]
[resource]
export/android/android_sdk_path = "/opt/oculus/android-sdk"
export/android/shutdown_adb_on_exit = false
"""


def get_apk_filename(participant_id: str) -> str:
    return f"avalon__{participant_id}.apk"


def generate_apk(
    godot_path: Path,
    worlds_path: Path,
    tmp_path: Path,
    output_path: Path,
    apk_script: str,
    participant_id: str,
    apk_version: str,
    include_worlds: bool = True,
    is_output_traced: bool = False,
):
    # create a dir in which we'll copy all the files in and begin doing the work
    participant_working_path = tmp_path / participant_id
    if participant_working_path.exists():
        shutil.rmtree(participant_working_path)
    participant_working_path.mkdir(parents=True)

    assert godot_path.exists(), f"Godot path not found: {godot_path}"

    participant_godot_path = participant_working_path / "godot"
    # copy godot folder in
    shutil.copytree(godot_path, participant_godot_path)

    (participant_godot_path / "export_presets.template.cfg").unlink()

    # update the user id and app version in the config file
    raw_config_path = participant_godot_path / "android" / "config.json"
    config = AvalonSimSpec.from_dict(json.load(open(raw_config_path, "r")))
    with config.mutable_clone() as new_config:
        new_config.recording_options.user_id = participant_id
        new_config.recording_options.apk_version = apk_version
    serialized_config = new_config.to_dict()
    json.dump(serialized_config, open(raw_config_path, "w"))

    pck_name = get_apk_filename(participant_id).replace(".apk", ".pck")
    apk_name = get_apk_filename(participant_id)
    pck_temp = participant_godot_path / pck_name
    apk_temp = participant_godot_path / apk_name
    apk_file = output_path / apk_name

    self_contained_godot_bin = participant_working_path / "godot_bin"
    godot_bin = Path("/opt/oculus/godot/godot")
    (participant_working_path / "._sc_").touch()
    (participant_working_path / "editor_data").mkdir()
    (participant_working_path / "editor_data" / "editor_settings-3.tres").write_text(EDITOR_SETTINGS)

    pre_bake_command = f"cd {participant_godot_path} && {apk_script}"
    pack_command = f"cd {participant_godot_path} && {self_contained_godot_bin} --verbose --export-pack  Android {pck_temp} | tee ../build.pck.log"
    bake_command = f"cd {participant_godot_path} && {self_contained_godot_bin} --verbose --export-debug Android {apk_temp} | tee ../build.apk.log"

    # make sure build dependencies are installed, clear editor config
    run_local_command(pre_bake_command, trace_output=is_output_traced)
    os.link(godot_bin, self_contained_godot_bin)

    # bake twice, the first build just fixes dependencies/imports
    run_local_command(pack_command, trace_output=is_output_traced)
    if include_worlds:
        assert worlds_path.exists(), f"World path not found: {worlds_path}"
        # link in worlds
        os.symlink(worlds_path, participant_godot_path / "worlds")
    run_local_command(bake_command, trace_output=is_output_traced)

    # move apk to output path
    apk_temp.rename(apk_file)
    pck_temp.unlink()

    # clean up and delete after
    # shutil.rmtree(participant_working_path)

    list_command = f"unzip -v {apk_file} -x 'META-INF/*' -x 'assets/android/config.json' | cut -c 48- | tail -n+2"
    apk_contents = run_local_command(list_command, trace_output=False).stdout
    (participant_working_path / "build.apk.lst").write_bytes(apk_contents)

    return {
        "participant_id": participant_id,
        "apk_file": apk_file,
        "apk_hash": hashlib.sha1(apk_contents).hexdigest(),
    }


def generate_apks(
    godot_path: Path,
    worlds_path: Path,
    tmp_path: Path,
    output_path: Path,
    apk_script: str,
    participant_ids: List[str],
    apk_version: str,
    include_worlds: bool = True,
    is_output_traced: bool = False,
):
    errors = []

    def on_done(result):
        pass

    def on_error(error: BaseException):
        print(error)
        errors.append(error)

    start_time = time.time()

    if not output_path.exists():
        output_path.mkdir(parents=True)

    with Pool(processes=len(participant_ids)) as worker_pool:
        requests = {}
        for participant_id in participant_ids:
            print(f"adding {participant_id} to requests")
            request = worker_pool.apply_async(
                generate_apk,
                kwds={
                    "godot_path": godot_path,
                    "worlds_path": worlds_path,
                    "tmp_path": tmp_path,
                    "output_path": output_path,
                    "apk_script": apk_script,
                    "participant_id": participant_id,
                    "apk_version": apk_version,
                    "include_worlds": include_worlds,
                    "is_output_traced": is_output_traced,
                },
                callback=on_done,
                error_callback=on_error,
            )
            requests[participant_id] = request

        results = []
        for participant_id, request in requests.items():
            request.wait()
            if request.successful():
                results.append(request.get())
            else:
                results.append(None)
                print(f"ERROR: {tmp_path / participant_id}/build.apk.log")

    print(f"finished in {time.time() - start_time} seconds")

    return results
