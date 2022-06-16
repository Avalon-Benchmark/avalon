extends Entity

class_name TextGrabbable

var frame := 0
const FRAME_DEBOUNCE = 10


func is_grabbable() -> bool:
	return true


func grab(_hand: RigidBody) -> Node:
	if frame > FRAME_DEBOUNCE:
		get_parent().set_next_text(self)
		frame = 0
	return null


func release():
	pass


func _process(_delta):
	frame += 1
