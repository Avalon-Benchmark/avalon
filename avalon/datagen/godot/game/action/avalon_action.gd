extends Action

class_name AvalonAction

var is_left_hand_grasping: bool = false
var is_right_hand_grasping: bool = false
var is_jumping: bool = false

var head_delta_position: Vector3 = Vector3.ZERO
var head_delta_rotation: Vector3 = Vector3.ZERO
var left_hand_delta_position: Vector3 = Vector3.ZERO
var left_hand_delta_rotation: Vector3 = Vector3.ZERO
var right_hand_delta_position: Vector3 = Vector3.ZERO
var right_hand_delta_rotation: Vector3 = Vector3.ZERO


func reset() -> void:
	is_left_hand_grasping = false
	is_right_hand_grasping = false
	is_jumping = false

	head_delta_position = Vector3.ZERO
	head_delta_rotation = Vector3.ZERO
	left_hand_delta_position = Vector3.ZERO
	left_hand_delta_rotation = Vector3.ZERO
	right_hand_delta_position = Vector3.ZERO
	right_hand_delta_rotation = Vector3.ZERO


func to_byte_array() -> PoolByteArray:
	# NOTE: the order must be the same as how we send bytes
	return HARD.assert(false, "Not implemented")
