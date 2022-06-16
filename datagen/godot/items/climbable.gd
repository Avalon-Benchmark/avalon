extends Entity

class_name Climbable


func is_climbable() -> bool:
	return true


func climb(_hand: RigidBody) -> Node:
	return self
