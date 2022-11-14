extends GameManager

class_name HumanGameManager

var arvr_interface: ARVRInterface
var recorder: Recorder
var initial_scene_path = "res://scenes/entry.tscn"
var gui_manager_scene_path = "res://scenes/gui/floating_gui_manager.tscn"
var gui_manager: FloatingGUIManager
var is_vr_warmed_up := false
var is_player_on_entry_world := false
var teleporter_manager: TeleporterManager
var max_frames_to_complete_level := INF
var client: AvalonClient
var visual_effect_handler: VisualEffectHandler
var is_apk_version_verified: bool = false
var moved_to_debug_world: bool = false

const ANDROID_LOCAL_RECORDER_PATH := "/sdcard/Android/data/com.godotengine.datagen/files"
# VR takes serveral frames to properly initialize when booting
const NUM_FRAMES_BEFORE_VR_ENABLED := 20


func _init(_root, _avalon_spec).(_root, _avalon_spec):
	gui_manager = load(gui_manager_scene_path).instance()
	scene_root.add_child(gui_manager)

	# configure human recorder
	if is_recording_human_actions():
		if is_remote_recording_enabled():
			client = AvalonClient.new(
				avalon_spec.recording_options.recorder_host,
				avalon_spec.recording_options.recorder_port,
				get_user_id()
			)
			upload_log_crash()
			recorder = RemoteRecorder.new(client)
		else:
			var recoder_output_dir: String
			if OS.get_name() == "Android":
				var _is_request_complete = OS.request_permissions()
				recoder_output_dir = ANDROID_LOCAL_RECORDER_PATH
			else:
				recoder_output_dir = avalon_spec.dir_root
			recorder = LocalRecorder.new(recoder_output_dir)

	# add the initial level
	if is_teleporter_enabled():
		WorldTools.set_global_world_metadata_from_included_levels()
		is_player_on_entry_world = true
		teleporter_manager = TeleporterManager.new(client, apk_version())
		for child in world_node.get_children():
			world_node.remove_child(child)
			child.queue_free()
		var scene: PackedScene = ResourceLoader.load(initial_scene_path)
		world_node.add_child(scene.instance())
		is_spawn_on_new_world_pending = _is_current_world_playable()
		_teleporter_on_scene_enter()
	else:
		_load_default_scene()

	var is_player_using_vr_controls = player is VRPlayer

	if is_player_using_vr_controls:
		var arvr_viewport: Viewport = root
		arvr_interface = ARVRServer.find_interface("OpenXR")
		if arvr_interface and arvr_interface.initialize():
			arvr_viewport.arvr = true
			arvr_viewport.fxaa = true
			arvr_viewport.render_target_update_mode = Viewport.UPDATE_ALWAYS
			Engine.target_fps = 120
		else:
			HARD.assert(false, "VR could not initialize")

	if not is_player_using_vr_controls and is_adding_debugging_views():
		# NOTE: hack to make sure the health bar width is correct when using mouse and keyboard locally
		root.find_node("health_bar", true, false).anchor_right = 0.5

	visual_effect_handler = VisualEffectHandler.new()

	if client:
		verify_apk_version_and_disable_teleporters()

	if player is MouseKeyboardHumanPlayer:
		root.get_viewport().add_child(MouseKeyboardHelpPanel.new())


func _is_current_world_playable() -> bool:
	for _node in controlled_nodes:
		var node: ControlledNode = _node
		if node.get_spawn_point(root) == null:
			return false
	return true


func is_teleporter_enabled() -> bool:
	return avalon_spec.recording_options.is_teleporter_enabled


func get_user_id() -> String:
	return avalon_spec.recording_options.user_id


func is_remote_recording_enabled() -> bool:
	return avalon_spec.recording_options.is_remote_recording_enabled


func is_recording_enabled_for_world() -> bool:
	return recorder and recorder.is_recording_enabled_for_world()


func is_recording_human_actions() -> bool:
	return avalon_spec.recording_options.is_recording_human_actions


func apk_version() -> String:
	return avalon_spec.recording_options.apk_version


func spawn_controlled_nodes() -> void:
	.spawn_controlled_nodes()
	frame = 0
	set_time_and_seed()


func before_physics() -> void:
	var current_observation = observation_handler.get_current_observation(player, frame)

	if is_recording_enabled_for_world():
		var interactive_observation = observation_handler.get_interactive_observation_from_current_observation(
			current_observation,
			episode,
			frame,
			ObservationHandler.SELECTED_FEATURES_FOR_HUMAN_RECORDING,
			true,
			false
		)
		recorder.record_observation(frame, interactive_observation)

	if visual_effect_handler and not is_player_on_entry_world:
		visual_effect_handler.react(current_observation)

	if (
		not is_player_on_entry_world
		and (current_observation["is_done"] or max_frames_to_complete_level < frame)
	):
		var world_id = WorldTools.get_world_id_from_world_path(current_world_path)
		var result = WorldTools.get_task_success_result(
			world_id,
			current_observation["is_done"],
			current_observation["is_dead"],
			max_frames_to_complete_level < frame,
			current_observation["hit_points"] > player.starting_hit_points
		)
		print(
			(
				"finished %s with result: success=%s health=%.2f time=%d/%s"
				% [
					world_id,
					result,
					current_observation["hit_points"],
					frame,
					max_frames_to_complete_level,
				]
			)
		)

		if is_recording_enabled_for_world():
			recorder.record_result(frame, result)

		advance_episode(initial_scene_path)

	log_debug_info()

	advance_frame()

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
			advance_episode(initial_scene_path)
			return

	teleport_to_new_world()

	if not is_teleporter_enabled() and not moved_to_debug_world and frame == 5 and HARD.mode():
		var debug_world_path = "res://main.tscn"
		if ResourceLoader.exists(debug_world_path):
			advance_episode(debug_world_path)
			moved_to_debug_world = true


func advance_episode(world_path: String) -> void:
	if camera_controller.debug_view != null:
		camera_controller.remove_debug_camera()

	current_world_path = world_path
	is_player_on_entry_world = world_path == initial_scene_path

	if is_teleporter_enabled():
		if not is_player_on_entry_world:
			_teleporter_on_scene_exit()

		if is_recording_human_actions():
			if is_player_on_entry_world:
				recorder.finish()
				delete_log_guard()
			else:
				# start the next recording
				create_log_guard(world_path)
				recorder.start(apk_version(), world_path, episode)

		max_frames_to_complete_level = INF

		if not WorldTools.is_practice_world(world_path) and not is_player_on_entry_world:
			var world_id = WorldTools.get_world_id_from_world_path(world_path)
			max_frames_to_complete_level = ceil(
				(
					WorldTools.get_time_to_complete_level_by_world_id(world_id)
					* ProjectSettings.get_setting("physics/common/physics_fps")
				)
			)

	.advance_episode(world_path)

	# this needs to happen after the world as been loaded
	if is_player_on_entry_world:
		_teleporter_on_scene_enter()

	visual_effect_handler.bind_world(world_node.find_node("WorldEnvironment", true, false))

	if client:
		verify_apk_version_and_disable_teleporters()


func idle(delta: float) -> void:
	.idle(delta)
	gui_manager.set_positions()


func read_input_before_physics() -> void:
	.read_input_before_physics()

	input_collector.read_input_before_physics()

	if is_recording_enabled_for_world():
		recorder.record_human_input(frame, input_collector.to_byte_array(controlled_nodes))


func apply_collected_action(delta: float) -> Array:
	var actions = .apply_collected_actions(delta)
	var action: PlayerAction = actions[0]
	if is_recording_enabled_for_world():
		recorder.record_action(frame, action.normalized.to_byte_array())

	return actions


func is_warming_up(delta: float) -> bool:
	var is_exiting_early = player.is_warming_up(delta, warmup_frame)
	warmup_frame += 1

	if is_exiting_early:
		return true

	# if we stop tracking hands -- try and fix the issue once tracking is working again
	if player.fix_tracking_once_working():
		gui_manager.open_by_name("tracking")
		return true
	else:
		gui_manager.close_by_name("tracking")

	return false


func _get_references_to_clean() -> Array:
	return (
		._get_references_to_clean()
		+ [
			client,
			recorder,
			visual_effect_handler,
		]
	)


func quit() -> void:
	if arvr_interface:
		arvr_interface.uninitialize()
	arvr_interface = null

	.quit()


func _get_teleporters() -> Array:
	return root.get_tree().get_nodes_in_group("teleporters")


func teleport_to_new_world():
	for teleporter in _get_teleporters():
		if teleporter.can_teleport():
			var world_path = teleporter_manager.get_world_path_for_teleporter(teleporter)
			if ResourceLoader.exists(world_path):
				prints("teleporting to", world_path)
				advance_episode(world_path)
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


func verify_apk_version_and_disable_teleporters():
	var res = client.verify(apk_version())
	is_apk_version_verified = res["is_valid"]
	if not is_apk_version_verified:
		for teleporter in _get_teleporters():
			teleporter_manager.disable(teleporter)


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
