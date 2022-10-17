class_name GymEnvBridge

extends Reference

##
## Pipe-based bridge for communication with an external process.
##

var action_pipe: File
var observation_pipe: File

## Whether an output_pipe_path was provided to enable observations
var is_output_enabled := false

var PERF_SIMPLE_AGENT: bool = ProjectSettings.get_setting("avalon/simple_agent")


func _init(input_pipe_path: String, output_pipe_path: String):
	print(CONST.BRIDGE_LOG_SIGNAL)
	action_pipe = File.new()
	HARD.assert(action_pipe.open(input_pipe_path, File.READ) == OK, "Could not open action pipe.")

	if output_pipe_path == null or output_pipe_path == "":
		return

	is_output_enabled = true
	observation_pipe = File.new()
	HARD.assert(
		observation_pipe.open(output_pipe_path, File.WRITE) == OK,
		"Could not open observation pipe."
	)


# Read a message and any additional data such as action or seed from pipe
func read_message() -> Array:
	if PERF_SIMPLE_AGENT:
		return action_pipe.get_var()

	var message = action_pipe.get_8()
	# this will only happen when debugging!
	if action_pipe.get_error() == ERR_FILE_EOF:
		return []
	match message:
		CONST.RENDER_MESSAGE, CONST.QUERY_AVAILABLE_FEATURES_MESSAGE, CONST.CLOSE_MESSAGE, CONST.SAVE_SNAPSHOT_MESSAGE:
			return [message]
		CONST.SEED_MESSAGE:
			return [message, action_pipe.get_64()]
		CONST.SELECT_FEATURES_MESSAGE:
			var feature_names = []
			var count := action_pipe.get_32()
			for _i in range(count):
				feature_names.append(action_pipe.get_line())
			return [message, feature_names]
		CONST.ACTION_MESSAGE, CONST.DEBUG_CAMERA_ACTION_MESSAGE:
			var count := action_pipe.get_32()
			var data := action_pipe.get_buffer(count)
			return [message, data]
		CONST.LOAD_SNAPSHOT_MESSAGE:
			var snpashot_path := action_pipe.get_line()
			return [message, snpashot_path]
		CONST.RESET_MESSAGE:
			var count := action_pipe.get_32()
			var data := action_pipe.get_buffer(count)
			var episode_seed := action_pipe.get_64()
			var world_id := action_pipe.get_line()
			var starting_hit_points := action_pipe.get_float()
			return [message, data, episode_seed, world_id, starting_hit_points]
		_:
			return []


func write_available_features_response(data: Dictionary):
	assert(
		is_output_enabled,
		"ERROR: Cannot write_available_features_response without an observation_pipe"
	)
	if PERF_SIMPLE_AGENT:
		observation_pipe.store_var(data.keys())
		observation_pipe.flush()
		return

	# write out feature count
	observation_pipe.store_32(len(data))

	for feature_name in data:
		var entry = data[feature_name]
		var value = entry[0]
		var data_type = entry[1]
		var shape = entry[2]

		# write the feature name, length prefixed
		observation_pipe.store_32(len(feature_name))
		observation_pipe.store_string(feature_name)

		# write the data type. Have to figure out the underlying type if this is an array
		var fixed_data_type = data_type
		if data_type == TYPE_ARRAY:
			var inner_value = value
			while typeof(inner_value) == TYPE_ARRAY:
				inner_value = inner_value[0]
			fixed_data_type = typeof(inner_value)
		observation_pipe.store_32(fixed_data_type)

		# write the shape
		observation_pipe.store_32(shape.size())
		for dim in shape:
			observation_pipe.store_32(dim)

	observation_pipe.flush()


func write_step_result_to_pipe(data: Dictionary):
	assert(is_output_enabled, "ERROR: Cannot write_step_result_to_pipe without an observation_pipe")
	if PERF_SIMPLE_AGENT:
		observation_pipe.store_var(data)
		observation_pipe.flush()
		return
	for feature_name in data:
		var entry = data[feature_name]
		var value = entry[0]
		var data_type = entry[1]
		_write_value(value, data_type)
	observation_pipe.flush()


func write_single_value(value):
	assert(is_output_enabled, "ERROR: Cannot write_single_value without an observation_pipe")
	_write_value(value, typeof(value))
	observation_pipe.flush()


func _write_value(value, data_type):
	match data_type:
		CONST.FAKE_TYPE_IMAGE:
			observation_pipe.store_buffer(value.data.data)
		TYPE_VECTOR2:
			observation_pipe.store_float(value.x)
			observation_pipe.store_float(value.y)
		TYPE_VECTOR3:
			observation_pipe.store_float(value.x)
			observation_pipe.store_float(value.y)
			observation_pipe.store_float(value.z)
		TYPE_QUAT:
			observation_pipe.store_float(value.x)
			observation_pipe.store_float(value.y)
			observation_pipe.store_float(value.z)
			observation_pipe.store_float(value.w)
		TYPE_REAL:
			observation_pipe.store_float(value)
		TYPE_INT:
			observation_pipe.store_32(value)
		TYPE_ARRAY:
			for inner_value in value:
				_write_value(inner_value, typeof(inner_value))
		TYPE_STRING:
			observation_pipe.store_line(value)
		_:
			HARD.stop("`gym_env_bridge.write_value` Unknown data type: %s", data_type)


func render_to_pipe(screen: Image):
	assert(is_output_enabled, "ERROR: Cannot render_to_pipe without an observation_pipe")
	screen.flip_y()
	screen.convert(Image.FORMAT_RGB8)
	observation_pipe.store_buffer(screen.data.data)
	observation_pipe.flush()
