extends DynamicEntity

class_name OpenButton


func _ready():
	axis_lock_linear_x = true
	axis_lock_linear_y = true
	axis_lock_linear_z = true
	axis_lock_angular_x = true
	axis_lock_angular_y = true
	axis_lock_angular_z = true

	var parent = self.get_parent()
	if parent.has_method("lock"):
		parent.lock(get_instance_id())


func grab(_hand: RigidBody) -> Node:
	var parent = self.get_parent()
	if parent.has_method("unlock"):
		parent.unlock(get_instance_id())
	find_node("red_mesh", true, false).visible = false
	find_node("green_mesh", true, false).visible = true
	return self


func is_grabbable():
	return true


func release():
	pass
