extends InputCollector

class_name DebugCameraInputCollector

var action


func to_normalized_relative_action(_player):
	return action


func scaled_relative_action_from_normalized_relative_action(normalized_action, _player):
	return normalized_action


func read_input_from_pipe(action_pipe: StreamPeerBuffer) -> void:
	action = DebugCameraAction.new()
	action.offset = read_vec(action_pipe)
	action.rotation = read_vec(action_pipe)
	action.is_facing_tracked = action_pipe.get_float() == 1.0
	action.is_frame_advanced = action_pipe.get_float() == 1.0
	var remaining_size = action_pipe.get_size() - action_pipe.get_position()
	action.tracked_node = action_pipe.get_utf8_string(remaining_size)


func read_vec(action_pipe: StreamPeerBuffer) -> Vector3:
	var vec = Vector3.ZERO
	vec.x = action_pipe.get_float()
	vec.y = action_pipe.get_float()
	vec.z = action_pipe.get_float()
	return vec


func reset():
	action = null
