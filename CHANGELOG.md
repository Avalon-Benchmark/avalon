# `avalon-rl` Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and `avalon-rl` adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Unreleased changes can be installed via `pip install git+https://github.com/Avalon-Benchmark/avalon.git`.

## [1.0.1]

### Fixed
- Sentry usage is now configurable via `SENTRY_DSN` and the sdk version is bumped to `1.12.1`
- Removed requirement of dm-tree from core avalon code.
- Disallowed broken version of scipy
- Fixes check install script on mac


## [1.0.0]

### Added
- `avalon.for_humans` CLI utility for to assist with human-consumable world generation, inspection, and VR setup.
- [VR guide](./docs/running_in_vr.md).
- Made windows editor binary installable.
- "Inspect Test Worlds" notebook/guide detailing how test worlds are generated and how to use `debug_act`.
- `python -m avalon.for_humans launch_editor` now prints a cautionary note on not editing/breaking stuff in the editor.
- `FixedWorldLoader` factored out from `FixedWorldGenerator`, making distinction between loading and runtime generation more explicit.
- Baseline model determinism tests.
- Improved `GodotError` clarity by extracting the first logged `ERROR`.

### Changed
-  Updated binary installed with `avalon.install_godot_binary` to latest version (`0.9.3`), fixing some editor & player runtime issues on mac.
- `generate_worlds` parameterization is now deterministic, regardless of `is_async` parameter.
- `generate_worlds.generate_worlds` renamed to `generate_worlds.generate_evaluation_worlds` and make it's purpose more internally explicit.
- Internals: make the Godot API slightly more generic using `CombinedInputCollector`, `CombinedAction`, and `ControlledNode`. 
  This is a step towards having multiple agents/actors controlling different in-godot entities.

### Fixed
- Added missing image in VR guide.
- Factored out inlined resources from some scene files.


## [1.0.0rc5] - 2022-10-21

### Changed
-  Updated binary installed with `avalon.install_godot_binary` to latest version.
- `avalon.install_godot_binary` now requires no arguments and installs all available binaries for the current platform.

### Fixed
- Installation script issues.


[1.0.0]: https://pypi.org/project/avalon-rl/1.0.0
[1.0.0rc5]: https://pypi.org/project/avalon-rl/1.0.0rc5
