extends AvalonAction

class_name MouseKeyboardAction

var is_left_hand_throwing: bool
var is_right_hand_throwing: bool
var is_eating: bool
var is_crouching: bool


func reset() -> void:
	.reset()
	is_left_hand_throwing = false
	is_right_hand_throwing = false
	is_eating = false
	is_crouching = false


func to_byte_array() -> PoolByteArray:
	# NOTE: the order must be the same as how we send bytes
	var stream = StreamPeerBuffer.new()
	stream.put_float(head_delta_position.x)
	stream.put_float(head_delta_position.z)
	stream.put_float(head_delta_rotation.x)
	stream.put_float(head_delta_rotation.y)
	stream.put_float(is_left_hand_grasping)
	stream.put_float(is_right_hand_grasping)
	stream.put_float(is_left_hand_throwing)
	stream.put_float(is_right_hand_throwing)
	stream.put_float(is_jumping)
	stream.put_float(is_eating)
	stream.put_float(is_crouching)
	return stream.data_array
