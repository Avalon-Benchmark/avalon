extends Entity

class_name RotatingBar

const LEEWAY_FACTOR := 1

export var rotation_axis := "z"
export var anchor_side := "left"
export var proportion_open := 0.0
export var unlatch_angle := 10.0
export var open := false


func _ready():
	HARD.assert(rotation_axis in ["x", "y", "z"])
	HARD.assert(anchor_side in ["right", "left"])
	# The bar's center is it's rotation point, so it does not have to move otherwise
	var bar_body = find_node("bar_body", true, false)
	bar_body.axis_lock_linear_x = true
	bar_body.axis_lock_linear_y = true
	bar_body.axis_lock_linear_z = true

	var adjustment = unlatch_angle * proportion_open
	if anchor_side == "right":
		adjustment *= -1

	if proportion_open == 1:
		open = true
	else:
		var parent = self.get_parent()
		if parent.has_method("lock"):
			parent.lock(get_instance_id())

	if rotation_axis == "x":
		bar_body.axis_lock_angular_x = false
		bar_body.axis_lock_angular_y = true
		bar_body.axis_lock_angular_z = true
		rotate_x(deg2rad(adjustment))
	elif rotation_axis == "y":
		bar_body.axis_lock_angular_x = true
		bar_body.axis_lock_angular_y = false
		bar_body.axis_lock_angular_z = true
		rotate_y(deg2rad(adjustment))
	elif rotation_axis == "z":
		bar_body.axis_lock_angular_x = true
		bar_body.axis_lock_angular_y = true
		bar_body.axis_lock_angular_z = false
		rotate_z(deg2rad(adjustment))


func _physics_process(_delta):
	var parent = self.get_parent()

	var angle = 0
	var bar_body = find_node("bar_body", true, false)
	var rotation = bar_body.get_transform().basis.get_euler()
	# Even though the global rotation axis changes, and we need to know of it to do axis rotation locking,
	# the local rotation always takes place in the z axis, and is what we care about in terms of angles
	angle = rotation.z
	angle = abs(rad2deg(angle))

	var unlatch_angle_exceeded
	if anchor_side == "left":
		unlatch_angle_exceeded = angle >= LEEWAY_FACTOR * unlatch_angle
	else:
		unlatch_angle_exceeded = angle <= 180 - (LEEWAY_FACTOR * unlatch_angle)

	if not open and unlatch_angle_exceeded:
		open = true
		if parent.has_method("unlock"):
			parent.unlock(get_instance_id())
	elif open and not unlatch_angle_exceeded:
		open = false
		if parent.has_method("lock"):
			parent.lock(get_instance_id())


func grab(_hand: RigidBody) -> Node:
	return self


func is_grabbable():
	return true


func release():
	pass
