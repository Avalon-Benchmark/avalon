# World Generation and Runtime

Core world generation and [Godot] runtime code for Avalon.

## Overview of main code paths

Examples using the public interface can be found in the root README and the notebooks it references.
In comparison, this overview is for providing a more detailed explanation of how the code in this module fits together internally.

**When training agents:**

1. A `GodotEnv` is created, starting a background godot process running the game logic in [godot/], written in [gdscript].
2. A pipe-based action & observation message bridge between the python env and the godot process is connected by code in [godot_env.py] and [gym_env_bridge.gd]
3. On `reset`, a world is generated by [world_creation/world_generator.py], resulting in [godot `.tscn` files].
   The difficulty of generated worlds can be set per-task on `env.world_generator`.
4. `env.act` (strongly typed) or `env.step` (gym-compliant) are then used to act in the environment for the duration of the episode.

**During human data collection:**
1. Batches of worlds are generated by [world_creation/world_generator.py].
2. Players run the [godot/] game via the provided `apk`.
3. Worlds are downloaded to devices and recordings are uploaded to the server via communication between [avalon_server/app.py] and the [avalon_client.gd].

## Bridging the `.gd <-> .py` gap

Code generation in [generate_godot_code.py] is used to keep the python codebase in-sync with the gdscript one.

* Python types and constants are generated from [CONST.gd] and [*Spec.gd] files.

* The afformentioned specs are serialzed as json and parsed in [sim_loop.gd],
  enabling environment configuration from python.

* While not currently exposed on the python side,
  almost all godot item attributes are exported and thus overridable per-world,
  which may be used for even richer behavior variation in the future.

See [godot/] for more details on setting up local godot development.


[Godot]: https://godotengine.org/
[world_creation/world_generator.py]: world_creation/world_generator.py
[godot `.tscn` files]: https://docs.godotengine.org/en/stable/development/file_formats/tscn.html
[godot/]: godot/
[gdscript]: https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/
[CONST.gd]: godot/game/utils/caps/CONST.gd
[*Spec.gd]: godot/game/specs
[sim_loop.gd]: godot/game/scene_root/sim_loop.gd
[godot `.tscn` files]: https://docs.godotengine.org/en/stable/development/file_formats/tscn.html
[gym_env_bridge.gd]: godot/game/utils/gym_env_bridge.gd
[godot_env.py]: godot_env.py
[avalon_client.gd]: godot/game/avalon_client.gd
[generate_godot_code.py]: generate_godot_code.py
