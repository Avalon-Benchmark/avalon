extends GameManager

class_name AgentGameManager

var env_bridge: GymEnvBridge
var is_reply_requested := false
var input_pipe_path: String
var output_pipe_path: String

var PERF_SIMPLE_AGENT: bool = ProjectSettings.get_setting("avalon/simple_agent")
var PERF_ACTION_READ: bool = ProjectSettings.get_setting("avalon/action_read")
var PERF_OBSERVATION_READ: bool = ProjectSettings.get_setting("avalon/observation_read")
var PERF_OBSERVATION_WRITE: bool = ProjectSettings.get_setting("avalon/observation_write")
var last_action = null
var last_observation = null


func _init(_root: Node, _avalon_spec, _input_pipe_path: String, _output_pipe_path: String).(
	_root, _avalon_spec
):
	input_pipe_path = _input_pipe_path
	output_pipe_path = _output_pipe_path

	env_bridge = GymEnvBridge.new(input_pipe_path, output_pipe_path)

	world_node.add_child(load(default_scene_path).instance())


func is_safe_to_spawn_on_new_world() -> bool:
	# need to apply one action before spawning so that nodes are in a reasonable place
	return (
		.is_safe_to_spawn_on_new_world()
		and (
			not player.is_human_playback_enabled
			or player.is_human_playback_enabled and frame == 1
		)
	)


func spawn_controlled_nodes() -> void:
	if player.is_human_playback_enabled:
		player.configure_nodes_for_playback(
			input_collector.arvr_camera_transform,
			input_collector.arvr_left_hand_transform,
			input_collector.arvr_right_hand_transform,
			input_collector.arvr_origin_transform,
			input_collector.human_height if input_collector.human_height != 0.0 else 2.0,
			true
		)
		# reset frame counter
		frame = 0

	.spawn_controlled_nodes()


func _reply_with_observation():
	var interactive_observation
	if last_observation == null:
		interactive_observation = observation_handler.get_interactive_observation(
			player, episode, frame, selected_features, true, true
		)
		if not PERF_OBSERVATION_READ:
			last_observation = interactive_observation
	else:
		interactive_observation = last_observation
	if PERF_OBSERVATION_WRITE:
		env_bridge.write_step_result_to_pipe(interactive_observation)


func read_input_from_pipe() -> bool:
	if env_bridge.is_output_enabled and is_reply_requested:
		_reply_with_observation()

	log_debug_info()

	# process messages until we encounter one that requires a tick:
	while true:
		var decoded_message
		if last_action == null:
			decoded_message = env_bridge.read_message()
		else:
			decoded_message = last_action
		match decoded_message:
			[CONST.RESET_MESSAGE, var _data, var episode_seed, var world_path, var starting_hit_points]:  # [tag, null action, int, string + \n, float]
				episode = episode_seed
				player.hit_points = starting_hit_points
				advance_episode(world_path)
				is_reply_requested = true
				# must reset the `input_collector` before we can take the next action
				input_collector.reset()
				return false
			[CONST.RENDER_MESSAGE]:
				env_bridge.render_to_pipe(observation_handler.get_rgbd_data())
			[CONST.SEED_MESSAGE, var new_episode_seed]:
				episode = new_episode_seed
			[CONST.QUERY_AVAILABLE_FEATURES_MESSAGE]:
				var available_features = observation_handler.get_available_features(player)
				env_bridge.write_available_features_response(available_features)
			[CONST.SELECT_FEATURES_MESSAGE, var feature_names]:
				selected_features = {}
				for feature_name in feature_names:
					selected_features[feature_name] = true
			[CONST.ACTION_MESSAGE, var data]:
				if not PERF_ACTION_READ:
					last_action = decoded_message
				if PERF_SIMPLE_AGENT:
					input_collector.read_input_from_data(data)
				else:
					var stream = StreamPeerBuffer.new()
					stream.data_array = data
					input_collector.read_input_from_pipe(stream)
				is_reply_requested = true
				advance_frame()
				return false
			[CONST.DEBUG_CAMERA_ACTION_MESSAGE, var data]:
				if is_debugging_output_requested():
					debug_logger.current_debug_file.flush()
				if camera_controller.debug_view == null:
					camera_controller.add_debug_camera()

				var is_frame_advanced = camera_controller.debug_view.read_and_apply_action(data)
				if not is_frame_advanced:
					_reply_with_observation()
				else:
					is_reply_requested = true
					advance_frame()
					return false
			[CONST.SAVE_SNAPSHOT_MESSAGE]:
				var snapshot_path = SnapshotHandler.save_snapshot(self)
				env_bridge.write_single_value(snapshot_path)
			[CONST.LOAD_SNAPSHOT_MESSAGE, var snapshot_path]:
				SnapshotHandler.load_snapshot(self, snapshot_path)
				_reply_with_observation()
			[CONST.CLOSE_MESSAGE]:
				print("CLOSE_MESSAGE received: exiting")
				return true
			_:
				HARD.stop("Encountered unexpected message %s" % [decoded_message])

	# quit if we ever break out of the above loop
	return true


func _get_references_to_clean() -> Array:
	return ._get_references_to_clean() + [env_bridge]
