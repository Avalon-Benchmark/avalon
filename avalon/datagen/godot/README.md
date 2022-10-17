# What this project does

Mostly the game logic for Avalon, implemented in [godot's gdscript](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_basics.html).

# Conventions

All units are metric standard units (distance in meters, time in seconds, etc).

The data generation process is intended to be fully deterministic.
Assuming you use the same world, seed, and action sequence,
you should get exactly the same hash for your generated data between runs, even between machines.

# How to setup for running locally

This is only strictly necessary if you want to run / debug Godot locally, but it's a good idea.

TODO: document where the new godot binaries are

 - Linux: put it under `/usr/local/bin/godot`
 - MacOS: put it under `/usr/local/bin/godot`
 - Windows: just unzip it anywhere (without spaces in the path name)

# How to set up your editor

Install the [GDScript plugin][2] for PyCharm.

**Important**: Go to Settings -> Languages & Frameworks -> GDScript, enable LSP features.

That enables decent code completion _when the Godot editor is running in the background_.

[1]: https://godotengine.org/download
[2]: https://plugins.jetbrains.com/plugin/13107-gdscript

##  GD Formatter

Create a file watcher in PyCharm to automatically reformat godot code.

![Setting up GD format](../../../../../docs/gdformat_configuration.png)

You'll need to create a new scope called `gd_scope` which can be set to `file:*.gd`

# Generating Python Types

To generate python types from godot code run the following:

```
python standalone/avalon/datagen/generate_godot_code.py
```
