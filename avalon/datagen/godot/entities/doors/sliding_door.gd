extends Door

class_name SlidingDoor

export var slide_axis := "x"


func _ready():
	HARD.assert(slide_axis in ["x", "y", "z"])
	._ready()


func latch():
	.latch()

	var door_body = find_node("body", true, false)
	door_body.axis_lock_linear_x = true
	door_body.axis_lock_linear_y = true
	door_body.axis_lock_linear_z = true


func unlatch():
	if not .unlatch():
		return false

	var door_body = find_node("body", true, false)
	if slide_axis == "x":
		door_body.axis_lock_linear_x = false
		door_body.axis_lock_linear_y = true
		door_body.axis_lock_linear_z = true
	elif slide_axis == "y":
		door_body.axis_lock_linear_x = true
		door_body.axis_lock_linear_y = false
		door_body.axis_lock_linear_z = true
	elif slide_axis == "z":
		door_body.axis_lock_linear_x = true
		door_body.axis_lock_linear_y = true
		door_body.axis_lock_linear_z = false

	return true
