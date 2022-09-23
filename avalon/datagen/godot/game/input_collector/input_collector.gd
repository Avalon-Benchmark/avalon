extends Reference

class_name InputCollector


func reset():
	HARD.assert(false, "Not implemented")


func to_normalized_relative_action(_player):
	return HARD.assert(false, "Not implemented")


func scaled_relative_action_from_normalized_relative_action(_normalized_action, _player):
	return HARD.assert(false, "Not implemented")


func read_input_from_event(_event: InputEvent) -> void:
	HARD.assert(false, "Not implemented")


func read_input_before_physics() -> void:
	pass


func read_input_from_pipe(_action_pipe: StreamPeerBuffer) -> void:
	HARD.assert(false, "Not implemented")


func to_byte_array(_player) -> PoolByteArray:
	return HARD.assert(false, "Not implemented")
