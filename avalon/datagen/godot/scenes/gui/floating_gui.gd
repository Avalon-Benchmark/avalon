extends Spatial

class_name FloatingGUI

var camera
onready var gui = get_child(0)

export var distance = 1.5

var is_opened := false


func _ready():
	visible = false
	camera = get_tree().root.get_camera()


func set_gui_position():
	var relative_distance = Vector3.FORWARD * distance
	gui.global_transform.origin = (
		camera.global_transform.basis * relative_distance
		+ camera.global_transform.origin
	)
	gui.global_transform.basis = camera.global_transform.basis


func set_position():
	if is_opened:
		set_gui_position()


func open():
	is_opened = true
	visible = true
	set_position()


func close():
	is_opened = false
	visible = false
