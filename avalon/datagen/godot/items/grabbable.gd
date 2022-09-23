extends Entity

# Useful for anything that's not an item but that you want to be grabbable without any custom behavior
class_name Grabbable


func is_grabbable() -> bool:
	return true


func grab(_hand: RigidBody) -> Node:
	return self


func release():
	pass
