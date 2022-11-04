extends Door

class_name HingeDoor

export var swing_axis := "y"


func _ready():
	HARD.assert(swing_axis in ["x", "y", "z"])
	._ready()


func latch():
	.latch()
	var door_body = find_node("body", true, false)
	door_body.axis_lock_angular_x = true
	door_body.axis_lock_angular_y = true
	door_body.axis_lock_angular_z = true
	door_body.axis_lock_linear_x = true
	door_body.axis_lock_linear_y = true
	door_body.axis_lock_linear_z = true


func unlatch():
	if not .unlatch():
		return false

	var door_body = find_node("body", true, false)
	if swing_axis == "x":
		door_body.axis_lock_angular_x = true
		door_body.axis_lock_angular_y = false
		door_body.axis_lock_angular_z = false
		door_body.axis_lock_linear_x = true
		door_body.axis_lock_linear_y = false
		door_body.axis_lock_linear_z = false
	elif swing_axis == "y":
		door_body.axis_lock_angular_x = true
		door_body.axis_lock_angular_y = false
		door_body.axis_lock_angular_z = true
		door_body.axis_lock_linear_x = false
		door_body.axis_lock_linear_y = true
		door_body.axis_lock_linear_z = false
	elif swing_axis == "z":
		door_body.axis_lock_angular_x = true
		door_body.axis_lock_angular_y = true
		door_body.axis_lock_angular_z = false
		door_body.axis_lock_linear_x = false
		door_body.axis_lock_linear_y = false
		door_body.axis_lock_linear_z = true
	return true
