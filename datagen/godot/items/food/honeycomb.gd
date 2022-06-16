extends Food

class_name Honeycomb


func is_edible() -> bool:
	return $clean_mesh.visible


func dirty():
	$clean_mesh.hide()
	$dirty_mesh.show()


func _physics_process(delta):
	._physics_process(delta)
	if ground_contact_count > 0 and is_edible():
		dirty()
