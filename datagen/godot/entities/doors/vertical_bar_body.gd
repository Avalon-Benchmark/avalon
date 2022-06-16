extends BarBody

class_name VerticalBarBody


func ready():
	.ready()
	axis_lock_linear_y = true


func grab(_physical_hand: RigidBody) -> Node:
	var _bar = .grab(_physical_hand)
	axis_lock_linear_y = false
	return self


func release():
	.release()
	# we don't want gravity to drop it back in its place
	axis_lock_linear_y = true
