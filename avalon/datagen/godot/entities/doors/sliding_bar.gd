extends Entity

class_name SlidingBar

export var sliding_axis := "x"
export var proportion_open := 0.0
export var proportion_to_unlock := 0.25
export var open := false
export var size: int


func _ready():
	HARD.assert(sliding_axis in ["x", "y", "z"])
	var bar_body = find_node("bar_body", true, false)
	bar_body.axis_lock_angular_x = true
	bar_body.axis_lock_angular_y = true
	bar_body.axis_lock_angular_z = true
	var aabb = bar_body.get_node("mesh/mesh").get_transformed_aabb()
	if sliding_axis == "x":
		bar_body.axis_lock_linear_x = false
		bar_body.axis_lock_linear_y = true
		bar_body.axis_lock_linear_z = true
		size = aabb.size.x
	elif sliding_axis == "y":
		bar_body.axis_lock_linear_x = true
		bar_body.axis_lock_linear_y = false
		bar_body.axis_lock_linear_z = true
		size = aabb.size.y
	elif sliding_axis == "z":
		bar_body.axis_lock_linear_x = true
		bar_body.axis_lock_linear_y = true
		bar_body.axis_lock_linear_z = false
		size = aabb.size.z

	var adjustment = size * proportion_open
	bar_body.translate(Vector3(adjustment, 0, 0))
	if proportion_open == 1:
		open = true
	else:
		var parent = self.get_parent()
		if parent.has_method("lock"):
			parent.lock(get_instance_id())


func _physics_process(_delta):
	var parent = self.get_parent()
	var bar_body = find_node("bar_body", true, false)
	var offset = abs(bar_body.get_transform().origin.y)
	var offset_to_open = proportion_to_unlock * size
	if not open and offset >= offset_to_open:
		open = true
		if parent.has_method("unlock"):
			parent.unlock(get_instance_id())
	elif open and offset < offset_to_open:
		open = false
		if parent.has_method("lock"):
			parent.lock(get_instance_id())


func grab(_hand: RigidBody) -> Node:
	return self


func is_grabbable():
	return true


func release():
	pass
