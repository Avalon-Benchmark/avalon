#!/usr/bin/env -S godot -s
extends SceneTree


func _init():
	OS.window_size = Vector2(0, 0)
	OS.window_borderless = true
	prints(clamp(INF, 0.0, 1.0))
	quit()
