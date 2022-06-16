extends AvalonAction

class_name VRAction


func reset() -> void:
	.reset()


func to_byte_array() -> PoolByteArray:
	# NOTE: the order must be the same as how we send bytes
	var stream = StreamPeerBuffer.new()
	stream.put_float(head_delta_position.x)
	stream.put_float(head_delta_position.y)
	stream.put_float(head_delta_position.z)
	stream.put_float(head_delta_rotation.x)
	stream.put_float(head_delta_rotation.y)
	stream.put_float(head_delta_rotation.z)
	stream.put_float(left_hand_delta_position.x)
	stream.put_float(left_hand_delta_position.y)
	stream.put_float(left_hand_delta_position.z)
	stream.put_float(left_hand_delta_rotation.x)
	stream.put_float(left_hand_delta_rotation.y)
	stream.put_float(left_hand_delta_rotation.z)
	stream.put_float(right_hand_delta_position.x)
	stream.put_float(right_hand_delta_position.y)
	stream.put_float(right_hand_delta_position.z)
	stream.put_float(right_hand_delta_rotation.x)
	stream.put_float(right_hand_delta_rotation.y)
	stream.put_float(right_hand_delta_rotation.z)
	stream.put_float(is_left_hand_grasping)
	stream.put_float(is_right_hand_grasping)
	stream.put_float(is_jumping)
	return stream.data_array
