extends SceneTree
class_name SimLoop

var sim_builder
var formatted_random_key := ""
var env_bridge
var player
var input_collector
var rng
var recorder

var video := 0  # TODO: rename to something like `episode`
var frame := 0

# TODO: remove
var frame_max := 0
var video_max := 0

var video_total := 0
var frame_total := 0

var camera_controller
# TODO: remove the rgb one, replace the depth one
var rgb_viewport: Viewport
var depth_viewport: Viewport
var top_down_viewport: Viewport
var isometric_viewport: Viewport

var is_initial_reset_done := false
var is_reply_requested := false
var selected_features := {}
var last_action: PoolByteArray
var arvr_interface: ARVRInterface

var current_debug_file: File

var spawn_point: Spatial
var is_spawning_player := false
var last_food_frame := INF
var is_food_gone := false

# TODO clean up scene management
# for the human player
var initial_scene_path = "res://scenes/entry.tscn"
var current_world_path = "res://scenes/entry.tscn"
var gui_manager: FloatingGUIManager
var is_vr_warmed_up := false
var is_player_on_entry_world := false
var teleporter_manager: TeleporterManager
var max_frames_to_complete_level := INF
var client: AvalonClient
var visual_effect_handler: VisualEffectHandler
var is_apk_version_verified: bool = false
var has_checked_apk_version: bool = false

# TODO remove
var generator_root: Node

# TODO remnants of removing clean recorder, can probably be cleaned up / removed / moved
var feature_shape_type: Dictionary = {}

# TODO in lieu
const NUM_FRAMES_BEFORE_VR_ENABLED := 20

# Godot main loop order:
# 1:   Update timers.
# 2:   For each step:
# 2.1: PhysicsServer.flush_queries()
# 2.2: MainLoop.iteration()
# 2.3: PhysicsServer.step() -> [...]._physics_process()
# 3:   MainLoop.idle()
# 4:   VisualServer.sync()
# 5:   VisualServer.draw()
# 6:   MainLoop._post_draw()


func is_recording_rgb() -> bool:
	return sim_builder.recording_options.is_recording_rgb


# TODO f0b4fcc3-5221-48be-8c39-47d1354efd72: enable/disable depth recording
func is_recording_depth() -> bool:
	return sim_builder.recording_options.is_recording_depth


func is_adding_debugging_views() -> bool:
	return sim_builder.recording_options.is_adding_debugging_views


func is_debugging_output_requested() -> bool:
	return sim_builder.recording_options.is_debugging_output_requested


func is_player_using_vr_controls() -> bool:
	HARD.assert(is_instance_valid(player), "Player must exist first")
	return player is VRPlayer


func is_player_using_mouse_keyboard_controls() -> bool:
	HARD.assert(is_instance_valid(player), "Player must exist first")
	return player is MouseKeyboardPlayer


func is_player_human() -> bool:
	return player is VRHumanPlayer or player is MouseKeyboardHumanPlayer


func is_recording_human_actions() -> bool:
	return sim_builder.recording_options.is_recording_human_actions


func is_remote_recording_enabled() -> bool:
	return sim_builder.recording_options.is_remote_recording_enabled


func is_recording_enabled_for_world() -> bool:
	return recorder and recorder.is_recording_enabled_for_world()


func get_user_id() -> String:
	return sim_builder.recording_options.user_id


func is_teleporter_enabled() -> bool:
	return sim_builder.recording_options.is_teleporter_enabled


func apk_version() -> String:
	return sim_builder.recording_options.apk_version


func _initialize():
	rng = CleanRNG.new()
	generator_root = root.get_node("/root/SceneRoot/GeneratorRoot")

	# TODO remove `sim_builder`
	var results = load_sim_builder_from_args()
	sim_builder = results[0]

	is_player_on_entry_world = is_teleporter_enabled()

	# TODO: remove
	frame_max = sim_builder.frame_max
	video_max = sim_builder.video_max

	var input_pipe_path = results[1]
	var output_pipe_path = results[2]

	# TODO to kill `sim_bulder`
	HARD.assert(sim_builder.player != null, "Must provide a player")
	generator_root.add_child(sim_builder.player.build())
	player = generator_root.find_node("player", true, false)

	if player is MouseKeyboardHumanPlayer:
		input_collector = MouseKeyboardHumanInputCollector.new()
	elif player is MouseKeyboardAgentPlayer:
		input_collector = MouseKeyboardAgentInputCollector.new()
	elif player is VRHumanPlayer:
		input_collector = VRHumanInputCollector.new()
	elif player is VRAgentPlayer:
		input_collector = VRAgentInputCollector.new()

	# TODO please print the seed for debugging but I don't know if here is the right spot
	print("seed:  %d" % sim_builder.random_int)

	# TODO sigh ... fix globals
	var _globals = root.get_node("/root/Globals")
	_globals.player = player
	_globals.user_id = get_user_id()
	_globals.is_player_human = is_player_human()
	_globals.is_player_using_vr_controls = is_player_using_vr_controls()
	_globals.is_player_using_mouse_keyboard_controls = is_player_using_mouse_keyboard_controls()

	if not is_player_human():
		env_bridge = GymEnvBridge.new(input_pipe_path, output_pipe_path)

	# TODO blerh ... do this better
	if not is_player_human() or is_player_human() and is_player_using_vr_controls():
		root.find_node("health_bar", true, false).queue_free()

	HARD.assert(sim_builder.recording_options.resolution_x > 0, "Invalid resolution_x")
	HARD.assert(sim_builder.recording_options.resolution_y > 0, "Invalid resolution_y")

	var window_size = Vector2(
		sim_builder.recording_options.resolution_x, sim_builder.recording_options.resolution_y
	)
	var total_window_size: Vector2 = window_size

	if is_player_human() and is_adding_debugging_views():
		var debug_scene = load("res://scenes/debug.tscn").instance()
		generator_root.add_child(debug_scene)

		for viewport_name in [
			"rgb_viewport", "depth_viewport", "top_down_viewport", "isometric_viewport"
		]:
			var _viewport = root.find_node(viewport_name, true, false)
			var _camera = _viewport.get_child(0)
			if is_instance_valid(_camera) and _camera.is_enabled:
				_viewport.size = window_size
			else:
				_viewport.render_target_update_mode = Viewport.UPDATE_DISABLED
				_viewport.size = Vector2.ZERO

		# quick hack to get the health bar the right size when debugging
		root.find_node("health_bar", true, false).anchor_right = 0.5
		total_window_size *= 2

	print(CONST.READY_LOG_SIGNAL)
	print()

	formatted_random_key = sim_builder.read_and_format_config_key(sim_builder.random_key)
	video = 0

	# TODO why do we have this print?
	if is_player_human():
		print("video:  %d" % video)

	set_time_and_seed()

	# configure human recorder
	if is_player_human() and is_recording_human_actions():
		if is_remote_recording_enabled():
			client = AvalonClient.new(
				sim_builder.recording_options.recorder_host,
				sim_builder.recording_options.recorder_port,
				get_user_id()
			)
			upload_log_crash()
			recorder = RemoteRecorder.new(client)
		else:
			recorder = LocalRecorder.new(sim_builder.dir_root)

	# add the initial level
	if is_player_human() and is_teleporter_enabled():
		WorldTools.set_global_world_metadata_from_included_levels()
		is_player_on_entry_world = true
		teleporter_manager = TeleporterManager.new(client, apk_version())
		var initial_scene = load(initial_scene_path).instance()
		generator_root.add_child(initial_scene)
		spawn_point = generator_root.find_node("SpawnPoint", true, false)
		is_spawning_player = is_instance_valid(spawn_point)
		_teleporter_on_scene_enter()
	else:
		generator_root.add_child(sim_builder.build())

	gui_manager = load("res://scenes/gui/floating_gui_manager.tscn").instance()
	generator_root.add_child(gui_manager)

	camera_controller = root.find_node("camera_controller", true, false)
	camera_controller.init()

	if is_player_human() and is_player_using_vr_controls():
		var arvr_viewport: Viewport = root
		arvr_interface = ARVRServer.find_interface("OpenXR")
		if arvr_interface and arvr_interface.initialize():
			# NOTE: really it'll set its own size, but it's this on Valve Index:
			arvr_viewport.arvr = true
			arvr_viewport.fxaa = true
			# shadow_atlas_size = 8192
			# shadow_atlas_quad_0 = 1
			# arvr_viewport.size = Vector2(2740, 2468)
			arvr_viewport.render_target_update_mode = Viewport.UPDATE_ALWAYS
			Engine.target_fps = 120
		else:
			HARD.assert(false, "VR could not initialize")
		visual_effect_handler = VisualEffectHandler.new()

	# this needs to happen at the end otherwise sadness
	OS.window_size = total_window_size


func _input_event(event: InputEvent):
	input_collector.read_input_from_event(event)


func _iteration(delta: float) -> bool:
	if Input.is_action_pressed("ui_cancel"):
		handle_quit()

	if is_player_human():
		_nonblocking_physics_process()

		# if we stop tracking hands -- try and fix the issue once tracking is working again
		if is_player_using_vr_controls() and player.fix_tracking_once_working():
			gui_manager.open_by_name("tracking")
			return false
		else:
			gui_manager.close_by_name("tracking")

	if is_player_human() and is_player_using_vr_controls():
		if not is_vr_warmed_up and frame == NUM_FRAMES_BEFORE_VR_ENABLED:
			is_vr_warmed_up = true

			# TODO maybe a nicer way we can calibrate height or display a message when they're playing
			player.set_human_height()

		# VR needs a "warm-up" period before the controllers start to be tracked
		if not is_vr_warmed_up and frame < NUM_FRAMES_BEFORE_VR_ENABLED:
			# move all the nodes to the correct starting point or else it will think there's velocities to apply
			input_collector.reset()
			# note: this is for VR specifically, we need to set the targets to where the ARVR controllers are
			player.set_target_body_transforms()
			# then move the physical body to where the targets are
			player.apply_action_to_physical_body(null, delta)
			# and finally reset the old transforms
			player.update_previous_transforms_and_velocities(true)
			return false

	if is_spawning_player:
		# spawning must happen before `flush_queries`
		player.set_spawn(spawn_point.global_transform)
		is_spawning_player = false
		input_collector.reset()
		return false

	if is_player_human():
		input_collector.read_input_before_physics()

	var normalized_action = input_collector.to_normalized_relative_action(player)

	if is_player_human() and is_recording_enabled_for_world():
		recorder.record_human_input(frame, input_collector.to_byte_array(player))
		recorder.record_action(frame, normalized_action.to_byte_array())

	var action = input_collector.scaled_relative_action_from_normalized_relative_action(
		normalized_action, player
	)
	player.apply_action(action, delta)
	input_collector.reset()
	return false


func _idle(_delta):
	VisualServer.request_frame_drawn_callback(self, "_post_draw", null)
	if OS.has_feature("zero_delay_physics"):
		PhysicsServer.flush_queries()

	# move tracking camera, after physics has been done
	camera_controller.do_physics(_delta)

	if is_player_human():
		gui_manager.set_positions()

	return false


func _post_draw(_arg):
	if not is_player_human():
		_blocking_physics_process()


func _get_rgbd_data() -> Image:
	var image_data = root.get_texture().get_data()
	#	image_data.flip_y()
	#	image_data.convert(Image.FORMAT_RGB8)
	return image_data


func _get_observation() -> Dictionary:
	if is_player_human() and is_player_on_entry_world:
		return {}

	# add any environment specific observations (usually related to food)
	var observation = player.get_oberservation_and_reward()

	if is_food_present_in_world():
		last_food_frame = frame

	var nearest_food = get_nearest_food()
	if is_instance_valid(nearest_food):
		observation["nearest_food_position"] = nearest_food.global_transform.origin
		observation["nearest_food_id"] = RID(nearest_food).get_id()
	else:
		observation["nearest_food_position"] = Vector3(INF, INF, INF)
		observation["nearest_food_id"] = -1

	observation["is_food_present_in_world"] = int(is_food_present_in_world())
	observation["is_done"] = int(
		(
			(frame - last_food_frame) >= player.num_frames_alive_after_food_is_gone
			or observation["is_dead"]
		)
	)

	return observation


func _blocking_physics_process():
	var current_observation = _get_observation()

	if frame == 0 and is_debugging_output_requested():
		_open_debug_file()

	var interactive_data = create_interactive_data(current_observation, video, frame)
	if env_bridge.is_output_enabled and is_reply_requested:
		var limited_data = limit_to_selected_features(interactive_data)
		env_bridge.write_step_result_to_pipe(limited_data)

	if is_debugging_output_requested():
		_write_debug_output()

	# process messages until we encounter one that requires a tick:
	while true:
		var decoded_message = env_bridge.read_message()
		match decoded_message:
			[CONST.RESET_MESSAGE, var data, var world_path, var starting_hit_points]:
				last_action = data
				if is_initial_reset_done:
					video += 1
					video_total += 1
				else:
					# ensures that video_min is the first video even in interactive mode
					# this should help with allowing us to interact in the same world regardless of
					# whether we are interacting or not
					is_initial_reset_done = true
					video = 0
				player.hit_points = starting_hit_points
				advance_video(world_path)
				is_reply_requested = true
				# must reset the `input_collector` before we can take the next action
				input_collector.reset()
				return
			[CONST.RENDER_MESSAGE]:
				# TODO: reimplement render to pipe
				HARD.stop("Implement render_to_pipe")
				# env_bridge.render_to_pipe(rgb_viewport.get_texture().get_data())
			[CONST.SEED_MESSAGE, var new_seed, var new_video_id]:
				sim_builder.random_int = new_seed
				video = new_video_id
			[CONST.QUERY_AVAILABLE_FEATURES_MESSAGE]:
				env_bridge.write_available_features_response(interactive_data)
			[CONST.SELECT_FEATURES_MESSAGE, var feature_names]:
				selected_features = {}
				for feature_name in feature_names:
					selected_features[feature_name] = true
			[CONST.ACTION_MESSAGE, var data]:
				last_action = data
				var stream = StreamPeerBuffer.new()
				stream.data_array = data
				input_collector.read_input_from_pipe(stream)
				is_reply_requested = true
				advance_frame()
				return
			[CONST.CLOSE_MESSAGE]:
				print("CLOSE_MESSAGE received: exiting")
				handle_quit()
				return
			_:
				HARD.stop("Encountered unexpected message %s" % [decoded_message])


func advance_video(world_path: String):
	current_world_path = world_path
	is_player_on_entry_world = world_path == initial_scene_path

	if is_player_human():
		if not is_player_on_entry_world:
			_teleporter_on_scene_exit()
		if is_recording_human_actions():
			if is_player_on_entry_world:
				recorder.finish()
				delete_log_guard()
			else:
				# start the next recording
				create_log_guard(world_path)
				recorder.start(apk_version(), world_path, sim_builder.random_int, video)

		max_frames_to_complete_level = INF

		if not WorldTools.is_practice_world(world_path) and not is_player_on_entry_world:
			var world_id = WorldTools.get_world_id_from_world_path(world_path)
			max_frames_to_complete_level = ceil(
				(
					WorldTools.get_time_to_complete_level_by_world_id(world_id)
					* ProjectSettings.get_setting("physics/common/physics_fps")
				)
			)

	# TODO player state can be a little unsafe / easy to mess up -- worth taking a look at this later
	# reset internal player state before moving to a new world
	player.reset_on_new_world()

	print("video:  %d" % video)
	frame = 0
	set_time_and_seed()

	var sim_node = generator_root.get_node("sim")
	for child in sim_node.get_children():
		sim_node.remove_child(child)
		child.queue_free()
	var scene: PackedScene = ResourceLoader.load(world_path)
	sim_node.add_child(scene.instance())

	# this needs to happen after the world as been loaded
	if is_player_human() and is_player_on_entry_world:
		_teleporter_on_scene_enter()

	# move the player to spawn in the right place
	spawn_point = sim_node.find_node("SpawnPoint", true, false)
	is_spawning_player = is_instance_valid(spawn_point)
	is_food_gone = false
	last_food_frame = INF
	if visual_effect_handler:
		visual_effect_handler.bind_world(sim_node.find_node("WorldEnvironment", true, false))


func advance_frame():
	frame += 1
	frame_total += 1
	set_time_and_seed()


func limit_to_selected_features(data: Dictionary) -> Dictionary:
	if len(selected_features) == 0:
		return data
	var result := {}
	for feature_name in selected_features:
		result[feature_name] = data[feature_name]
	return result


func _get_teleporters() -> Array:
	return root.get_tree().get_nodes_in_group("teleporters")


func teleport_to_new_world():
	for teleporter in _get_teleporters():
		if teleporter.can_teleport():
			var world_path = teleporter_manager.get_world_path_for_teleporter(teleporter)
			if ResourceLoader.exists(world_path):
				prints("teleporting to", world_path)
				advance_video(world_path)
			else:
				prints("cannot teleport to", world_path)
				teleporter.disable()
				var message: String
				var world_id = WorldTools.get_world_id_from_world_path(world_path)
				if world_id:
					message = "Could not teleport to %s" % world_id
				else:
					message = "Unable to teleport to this level"
				teleporter.set_message(message)


func _teleporter_on_scene_enter() -> void:
	for teleporter in _get_teleporters():
		teleporter_manager.on_scene_enter(teleporter)


func _teleporter_on_scene_exit() -> void:
	for teleporter in _get_teleporters():
		teleporter_manager.on_scene_exit(teleporter)


func _nonblocking_physics_process():
	# disable teleporters if users have the wrong apks
	if client:
		verify_apk_version()

	if is_recording_enabled_for_world():
		var current_observation = _get_observation()
		var interactive_data = create_interactive_data(current_observation, video, frame, false)
		recorder.record_observation(frame, interactive_data)

		if visual_effect_handler:
			visual_effect_handler.react(current_observation)

		if (
			not is_player_on_entry_world
			and (current_observation["is_done"] or max_frames_to_complete_level < frame)
		):
			prints("is done", current_observation["is_done"])
			prints("ran out of time", max_frames_to_complete_level < frame)
			# TODO tell the users they either succeed or failed
			advance_video(initial_scene_path)

	if frame == 0 and is_debugging_output_requested():
		_open_debug_file()

	if is_debugging_output_requested():
		_write_debug_output()

	frame += 1
	frame_total += 1

	# TODO is there a better way to restart?
	#	would love to emit signals but I can't seem to listen to them in MainLoop
	if gui_manager.has("restart"):
		var restart_gui = gui_manager.get_by_name("restart")
		var shortened_apk_version = apk_version().substr(0, 8)

		if restart_gui.is_opened and current_world_path != initial_scene_path:
			var world_id = WorldTools.get_world_id_from_world_path(current_world_path)
			var world_info = WorldTools.get_long_name_from_world_id(world_id)
			restart_gui.set_message(
				"End this task?\n%s, Version: %s" % [world_info, shortened_apk_version]
			)

		if restart_gui.is_opened and current_world_path == initial_scene_path:
			restart_gui.set_message("Reset?\nVersion: %s" % shortened_apk_version)

		if restart_gui.is_restarting:
			print("player requested a restart")
			# log when the user hits the restart button -- this will technically fail a run
			if client and current_world_path != initial_scene_path:
				var world_id = WorldTools.get_world_id_from_world_path(current_world_path)
				var run_id = recorder.current_state["run_id"]
				var _res = client.reset_world(world_id, run_id)

			restart_gui.is_restarting = false
			advance_video(initial_scene_path)
			return

	teleport_to_new_world()

	if not is_teleporter_enabled() and frame_total == 5 and HARD.mode():
		var debug_world_path = "res://main.tscn"
		if ResourceLoader.exists(debug_world_path):
			advance_video(debug_world_path)


func handle_quit():
	print("videos: %11d" % video_total)
	print("frames: %11d" % frame_total)
	print()

	if is_player_human() and is_player_using_vr_controls():
		if arvr_interface:
			arvr_interface.uninitialize()
		arvr_interface = null

	ClassBuilder.clear()

	paused = true
	quit()


func set_time_and_seed():
	rng.set_seed(video, frame, sim_builder.random_int, formatted_random_key)


# TODO: this will need to change slightly, probably don't need to call ClassBuilder
func load_sim_builder_from_args():
	var args := Array(OS.get_cmdline_args())
	var config_paths = []
	if HARD.mode():
		config_paths.append("./config.json")
	elif OS.get_name() == "Android":
		config_paths.append("res://android/config.json")
	var input_pipe_path := ""
	var output_pipe_path := ""
	for arg in args:
		var split_arg := Array(arg.split("=", true, 1))
		match split_arg[0]:
			"--dev":
				pass
			"--config-file":
				var path = split_arg[1]
				config_paths.append(ProjectSettings.globalize_path(path))
			"--input-pipe-path":
				input_pipe_path = split_arg[1]
			"--output-pipe-path":
				output_pipe_path = split_arg[1]
			_:
				HARD.stop("Unknown command line arg: '%s'", [arg])
	HARD.assert(HARD.mode() or (len(config_paths) == 1), "No config file given")

	var json_dict := Tools.path_read_json(config_paths[-1])
	var json_spec = ClassBuilder.get_object_from_json(json_dict, "/")
	return [json_spec, input_pipe_path, output_pipe_path]


func _get_file_path(file_name: String) -> String:
	return "{dir_root}/{video_id}/{file_name}".format(
		{
			"dir_root": sim_builder.dir_root,
			"video_id": "%06d" % video,
			"file_name": file_name,
		}
	)


func _open_debug_file():
	if current_debug_file and current_debug_file.is_open() == true:
		current_debug_file.close()
	var path = Tools.file_create(_get_file_path("debug.json"))
	current_debug_file = File.new()
	var error = current_debug_file.open(path, File.READ_WRITE)
	HARD.assert(error == OK, "IO error: %s for file at path %s" % [error, path])


func _write_debug_output():
	var item_data := _get_dynamic_item_snapshot()
	var items_json := JSON.print({"time": frame, "episode": video, "items": item_data})
	current_debug_file.store_line(items_json)
	var error = current_debug_file.get_error()
	HARD.assert(error == OK, "IO error: %s", current_debug_file.get_path())


func is_food_present_in_world():
	return len(_get_all_food_in_world()) > 0


func get_nearest_food() -> Node:
	var current_position = player.get_current_position()
	var min_distance = INF
	var nearest_food = null
	for food in _get_all_food_in_world():
		var food_distance = (food.global_transform.origin - current_position).length()
		if food_distance < min_distance:
			nearest_food = food
			min_distance = food_distance
	return nearest_food


func _get_all_food_in_world() -> Array:
	var foods = []
	for food in root.get_tree().get_nodes_in_group("food"):
		if food.is_hidden:
			continue
		foods.append(food)
	return foods


func _get_items_recursively(node: Node) -> Array:
	var results = []
	for child in node.get_children():
		if child is Item:
			results.push_back(child.to_dict())
		else:
			results += _get_items_recursively(child)
	return results


func _get_dynamic_item_snapshot() -> Array:
	var dynamic_item_group = root.find_node("dynamic_tracker", true, false)
	var results = []
	if dynamic_item_group:
		results += _get_items_recursively(dynamic_item_group)
	return results


# TODO remnants of removing clean recorder, can probably be cleaned up / removed / moved
func create_interactive_data(
	feature_data: Dictionary, video_id: int, frame_id: int, is_getting_image_data: bool = true
) -> Dictionary:
	if feature_shape_type.empty():
		for feature in feature_data:
			var value = feature_data[feature]
			if feature == CONST.RGB_FEATURE or feature == CONST.DEPTH_FEATURE:
				# always set below because sometimes not in feature data
				pass
			else:
				var data_type = typeof(value)
				var shape = get_shape(value, data_type)
				feature_shape_type[feature] = [data_type, shape]
		feature_shape_type[CONST.RGB_FEATURE] = [
			CONST.FAKE_TYPE_IMAGE,
			[
				sim_builder.recording_options.resolution_y,
				sim_builder.recording_options.resolution_x,
				3
			]
		]
		feature_shape_type[CONST.RGBD_FEATURE] = [
			CONST.FAKE_TYPE_IMAGE,
			[
				sim_builder.recording_options.resolution_y,
				sim_builder.recording_options.resolution_x,
				4
			]
		]
		feature_shape_type[CONST.DEPTH_FEATURE] = [
			CONST.FAKE_TYPE_IMAGE,
			[
				sim_builder.recording_options.resolution_y,
				sim_builder.recording_options.resolution_x,
				3
			]
		]

	var observed_data := {}
	for feature in feature_data:
		var value = feature_data[feature]
		var feature_shape = feature_shape_type[feature]
		var data_type: int = feature_shape[0]
		var shape = feature_shape[1]
		observed_data[feature] = [value, data_type, shape]

	observed_data[CONST.VIDEO_ID_FEATURE] = [video_id, typeof(video_id), [1]]
	observed_data[CONST.FRAME_ID_FEATURE] = [frame_id, typeof(frame_id), [1]]

	var rgbd_data = null
	if (
		is_getting_image_data
		and is_recording_rgb()
		and (len(selected_features) == 0 or CONST.RGBD_FEATURE in selected_features)
	):
		rgbd_data = _get_rgbd_data()

	observed_data[CONST.RGBD_FEATURE] = [
		rgbd_data,
		CONST.FAKE_TYPE_IMAGE,
		[sim_builder.recording_options.resolution_y, sim_builder.recording_options.resolution_x, 4]
	]

	# TODO neither of these make sense now
	observed_data[CONST.RGB_FEATURE] = [
		null,
		CONST.FAKE_TYPE_IMAGE,
		[sim_builder.recording_options.resolution_y, sim_builder.recording_options.resolution_x, 3]
	]

	observed_data[CONST.DEPTH_FEATURE] = [
		null,
		CONST.FAKE_TYPE_IMAGE,
		[sim_builder.recording_options.resolution_y, sim_builder.recording_options.resolution_x, 3]
	]

	return observed_data


# TODO remnants of removing clean recorder, can probably be cleaned up / removed / moved
func get_shape(value, data_type):
	match data_type:
		TYPE_VECTOR2:
			return [2]
		TYPE_VECTOR3:
			return [3]
		TYPE_REAL:
			return [1]
		TYPE_INT:
			return [1]
		TYPE_ARRAY:
			var shape = get_shape(value[0], typeof(value[0]))
			shape.append(value.size())
			return shape
		_:
			print("get_shape")
			HARD.stop("Unknown data type: %s", data_type)


func create_log_guard(world_path: String):
	var world := WorldTools.get_world_id_from_world_path(world_path)
	var guard := File.new()
	var error := guard.open("user://guard.id", File.WRITE)
	if error == OK:
		guard.store_string(world)
		guard.close()
	else:
		prints("cannot create guard:", error)


func delete_log_guard():
	var udata := Directory.new()
	var error := udata.remove("user://guard.id")
	if error != OK:
		prints("cannot delete guard:", error)


func upload_log_crash():
	var guard := File.new()
	var udata := Directory.new()
	var error := OK
	error = guard.open("user://guard.id", File.READ)
	if error != OK:
		return
	error = udata.remove("user://guard.id")
	if error != OK:
		prints("cannot delete guard:", error)
	error = udata.open("user://")
	if error != OK:
		prints("cannot open logs:", error)
		return
	error = udata.list_dir_begin()
	if error != OK:
		prints("cannot list logs:", error)
		return
	var file_name := ""  # example: godot_2022_05_30_19.15.43.log
	var log_found := false
	while true:
		file_name = udata.get_next()
		if file_name == "":
			break
		if file_name.begins_with("godot_"):
			log_found = true
			break
	udata.list_dir_end()
	if log_found == false:
		return
	var log_file := File.new()
	error = log_file.open("user://%s" % file_name, File.READ)
	if error != OK:
		prints("cannot read file:", file_name, error)
		return
	var world_id := guard.get_as_text().strip_edges()
	var run_id := "crash"
	var log_data := log_file.get_buffer(16 * 1024 * 1024)
	var upload_name = file_name.replace("-", "_")
	var _res = client.upload(apk_version(), world_id, run_id, log_data, upload_name)


func verify_apk_version():
	if not has_checked_apk_version:
		var res = client.verify(apk_version())
		is_apk_version_verified = res["is_valid"]
		has_checked_apk_version = true
	if not is_apk_version_verified:
		for teleporter in _get_teleporters():
			teleporter_manager.disable(teleporter)
