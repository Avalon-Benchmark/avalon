extends DynamicEntity

# Useful for anything that's not an item but that you want to be grabbable without any custom behavior
class_name BarBody

export var MAX_DISTANCE_FROM_HAND := 1.0


func is_grabbable() -> bool:
	return true


func is_pushable() -> bool:
	return false


func grab(_physical_hand: RigidBody) -> Node:
	angular_damp = 0.0
	linear_damp = 0.0

	return self


func hold(physical_hand: RigidBody) -> Node:
	if (
		(global_transform.origin - physical_hand.global_transform.origin).length()
		> MAX_DISTANCE_FROM_HAND
	):
		return null
	return self


func release():
	# prevents the bar from moving a lot when we let go of it
	angular_damp = 1.0
	linear_damp = 1.0
