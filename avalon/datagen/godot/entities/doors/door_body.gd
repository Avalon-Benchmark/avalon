extends DynamicEntity

# Includes the body, the lever and latcing mechanism
class_name DoorBody

export var is_auto_latching := false


func is_grabbable() -> bool:
	var parent = get_parent()
	return not parent.is_locked()


func is_pushable() -> bool:
	var parent = get_parent()
	return not parent.is_locked() and not parent.is_latched


func grab(_hand: RigidBody) -> Node:
	var parent = get_parent()
	if parent.has_method("unlatch"):
		parent.unlatch()
	return self


func release():
	var parent = get_parent()
	if is_auto_latching and parent.has_method("latch"):
		parent.latch()


func push(push_impulse: Vector3, push_offset: Vector3) -> void:
	apply_impulse(push_offset, push_impulse)
