extends Entity

class_name TeleportButton

signal on_button_pressed(button)


func is_grabbable() -> bool:
	return true


func grab(_hand: RigidBody) -> Node:
	emit_signal("on_button_pressed", self)
	print("pressed teleporter button")
	return null


func release():
	pass
