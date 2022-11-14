extends Reference

class_name InputCollector


func reset():
	HARD.assert(false, "Not implemented")


func get_action(_controlled_node):
	HARD.assert(false, "Not implemented")


func read_input_from_event(_event: InputEvent) -> void:
	HARD.assert(false, "Not implemented")


func read_input_before_physics() -> void:
	pass


func read_input_from_pipe(_action_pipe: StreamPeerBuffer) -> void:
	HARD.assert(false, "Not implemented")


func read_input_from_data(_action: PoolRealArray) -> void:
	HARD.assert(false, "Not implemented")


func write_into_stream(_stream: StreamPeerBuffer, _player) -> void:
	HARD.assert(false, "Not implemented")


func to_byte_array(player) -> PoolByteArray:
	var stream = StreamPeerBuffer.new()
	write_into_stream(stream, player)
	return stream.data_array
