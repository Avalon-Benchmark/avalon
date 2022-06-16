extends Food

class_name Carrot


func _ready():
	is_behaving_like_item = false


func is_edible() -> bool:
	return is_behaving_like_item


func grab(_hand_tracker: RigidBody):
	if is_edible():
		return self
	global_transform = global_transform.translated(Vector3(0, 0.75, 0))
	rotate_z(deg2rad(90))

	$leaves.hide()
	$leaves_collision_shape.disabled = true
	$leaves_collision_shape.queue_free()
	$no_leaves.show()
	mode = MODE_RIGID
	apply_central_impulse(Vector3.UP * 0.25)
	behave_like_item()
