# What this project does

It is a set of scripts for generating vidoes and images of simple 3D scenes.
In particular, it is designed to allow us to tightly control the complexity of those scenes.

# Conventions

All units are metric standard units (distance in meters, time in seconds, etc).

All output data is created as a set of numpy arrays in a folder.

The program creates sequences of data (ex: videos, arrays of points, etc) for a fixed set of feature names (ex: rgb, depth, primary_object_position, etc).
You can think of the data as images, but that happens at a higher level.

The data generation process is intended to be fully deterministic.
Assuming you use the same config, you should get exactly the same hash for your generated data between runs, even between machines.

All parameters are expressed via json configuration files.

# How to setup for running locally

This is only strictly necessary if you want to run / debug Godot locally, but it's a good idea.

Download and extract the [Godot editor][1]. (Just the standard, 64-bit, 3.2.3 version.)

 - Linux: put it under `/usr/bin/godot`
 - MacOS: make that a symlink to `/?/Godot.app/Contents/MacOS/Godot`
 - Windows: just unzip it anywhere (without spaces in the path name) 

# How to set up your editor

Install the [GDScript plugin][2] for PyCharm.

**Important**: Go to Settings -> Languages & Frameworks -> GDScript, enable LSP features.

That enables decent code completion _when the Godot editor is running in the background_.

[1]: https://godotengine.org/download
[2]: https://plugins.jetbrains.com/plugin/13107-gdscript

##  GD Formatter

Create a file watcher in PyCharm to automatically reformat godot code.

![Setting up GD format](../../../../docs/gdformat_configuration.png)

You'll need to create a new scope called `gd_scope` which can be set to `file:*.gd`

# Generating Python Types

To generate python types from godot code run the following:

```
python standalone/avalon/datagen/generate_godot_code.py
```
