extends Reference

class_name Recorder

var current_state := _get_default_state()

const ACTION_KEY = "actions"
const OBSERVATION_KEY = "observations"
const HUMAN_INPUT_KEY = "human_inputs"
const METADATA_KEY = "metadata"
const _METADATA_IGNORE = [ACTION_KEY, OBSERVATION_KEY, HUMAN_INPUT_KEY]

const JSON_FILE = "JSON_FILE"
const BINARY_FILE = "BINARY_FILE"


func _get_default_state() -> Dictionary:
	return {
		ACTION_KEY: [],
		OBSERVATION_KEY: [],
		HUMAN_INPUT_KEY: [],
		"start_time": null,
		"end_time": null,
		"world_id": null,
		"episode_seed": null,
		"apk_version": null,
		"is_success": false,
		"run_id": Tools.uuidv4()
	}


func is_recording_enabled_for_world() -> bool:
	return current_state.get("world_id") != null


func record_result(_frame: int, is_success: bool) -> void:
	current_state["is_success"] = is_success


func record_action(_frame: int, action: PoolByteArray) -> void:
	var stream = StreamPeerBuffer.new()
	stream.put_8(CONST.ACTION_MESSAGE)
	stream.put_32(len(action))
	current_state[ACTION_KEY].append(stream.data_array + action)


func record_human_input(_frame: int, human_input: PoolByteArray) -> void:
	var stream = StreamPeerBuffer.new()
	stream.put_8(CONST.HUMAN_INPUT_MESSAGE)
	stream.put_32(len(human_input))
	current_state[HUMAN_INPUT_KEY].append(stream.data_array + human_input)


func write_value(buffer: StreamPeerBuffer, value, data_type):
	match data_type:
		TYPE_VECTOR2:
			buffer.put_float(value.x)
			buffer.put_float(value.y)
		TYPE_VECTOR3:
			buffer.put_float(value.x)
			buffer.put_float(value.y)
			buffer.put_float(value.z)
		TYPE_QUAT:
			buffer.put_float(value.x)
			buffer.put_float(value.y)
			buffer.put_float(value.z)
			buffer.put_float(value.w)
		TYPE_REAL:
			buffer.put_float(value)
		TYPE_INT:
			buffer.put_32(value)
		TYPE_ARRAY:
			for inner_value in value:
				write_value(buffer, inner_value, typeof(inner_value))
		_:
			HARD.stop("`recorder.write_value` unknown data type: %s", data_type)


func _get_observation_byte_array(data: Dictionary) -> PoolByteArray:
	var stream = StreamPeerBuffer.new()
	for feature_name in data:
		var entry = data[feature_name]
		var value = entry[0]
		var data_type = entry[1]
		write_value(stream, value, data_type)
	return stream.data_array


func record_observation(_frame: int, observation: Dictionary) -> void:
	var byte_array = _get_observation_byte_array(observation)
	current_state[OBSERVATION_KEY].append(byte_array)


func _get_data_by_key_from_state(key: String):
	if key == METADATA_KEY:
		return _get_metadata()
	return current_state[key]


func _get_metadata() -> Dictionary:
	var ret := {}
	for key in current_state:
		if not key in _METADATA_IGNORE:
			ret[key] = current_state[key]

	var world_metadata = WorldTools.get_global_world_metadata(current_state["world_id"])
	for k in world_metadata:
		ret[k] = world_metadata[k]
	return ret


func _get_file_extension_from_file_type(file_type: String) -> String:
	match file_type:
		JSON_FILE:
			return "json"
		BINARY_FILE:
			return "out"
		_:
			return HARD.assert(false)


func _get_filename(key: String, file_type: String) -> String:
	return "%s.%s" % [key, _get_file_extension_from_file_type(file_type)]


func save(_key: String, _file_type: String) -> void:
	HARD.assert(false, "`save` not implemented")


func reset() -> void:
	current_state = _get_default_state()


func finish() -> void:
	print("finished recording for %s" % current_state["world_id"])
	if is_recording_enabled_for_world():
		current_state["end_time"] = OS.get_unix_time()
		save(OBSERVATION_KEY, BINARY_FILE)
		save(ACTION_KEY, BINARY_FILE)
		save(HUMAN_INPUT_KEY, BINARY_FILE)
		save(METADATA_KEY, JSON_FILE)
	reset()


func start(apk_version: String, world_path: String, episode: int) -> void:
	var world_id = WorldTools.get_world_id_from_world_path(world_path)
	print("starting recording for %s (%s)" % [world_id, world_path])
	current_state["start_time"] = OS.get_unix_time()
	current_state["world_id"] = WorldTools.get_world_id_from_world_path(world_path)
	current_state["episode_seed"] = episode
	current_state["apk_version"] = apk_version
