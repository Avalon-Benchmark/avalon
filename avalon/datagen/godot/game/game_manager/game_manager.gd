extends Reference

class_name GameManager

var root: Node
var avalon_spec: SimSpec
var rng: CleanRNG
var player: Player
var input_collector: InputCollector
var camera_controller: TrackingCameraController
var observation_handler: ObservationHandler
var debug_logger: DebugLogger

var episode := 0
var frame := 0
var warmup_frame := 0

var scene_root: Node
var world_node: Node
var spawn_point: Spatial
var is_spawning_player := false

var default_scene_path := "res://scenes/empty.tscn"
var selected_features := {}


func _init(_root: Node, _avalon_spec: SimSpec):
	avalon_spec = _avalon_spec
	root = _root

	if avalon_spec.episode_seed != null:
		episode = avalon_spec.episode_seed

	rng = CleanRNG.new()
	scene_root = root.get_node("/root/scene_root")
	world_node = scene_root.get_node("world")

	if is_debugging_output_requested():
		debug_logger = DebugLogger.new(root)

	scene_root.add_child(avalon_spec.player.get_node())
	player = scene_root.find_node("player", true, false)

	if player is MouseKeyboardHumanPlayer:
		input_collector = MouseKeyboardHumanInputCollector.new()
	elif player is MouseKeyboardAgentPlayer:
		input_collector = MouseKeyboardAgentInputCollector.new()
	elif player is VRHumanPlayer:
		input_collector = VRHumanInputCollector.new()
	elif player is VRAgentPlayer:
		input_collector = VRAgentInputCollector.new()

	camera_controller = TrackingCameraController.new(
		root, avalon_spec.get_resolution(), "physical_head", is_adding_debugging_views()
	)

	observation_handler = ObservationHandler.new(root, camera_controller)


func _load_default_scene() -> void:
	scene_root.add_child(load(default_scene_path).instance())


func is_adding_debugging_views() -> bool:
	return avalon_spec.recording_options.is_adding_debugging_views


func is_debugging_output_requested() -> bool:
	return avalon_spec.recording_options.is_debugging_output_requested


func read_input_from_pipe() -> bool:
	return false


func before_physics() -> void:
	pass


func set_time_and_seed() -> void:
	rng.set_seed(episode, frame)


func advance_frame() -> void:
	frame += 1
	set_time_and_seed()


func advance_episode(world_path: String) -> void:
	if camera_controller.debug_view != null:
		camera_controller.remove_debug_camera()

	# reset internal player state before moving to a new world
	player.reset_on_new_world()

	frame = 0
	set_time_and_seed()

	for child in world_node.get_children():
		world_node.remove_child(child)
		child.queue_free()

	var scene: PackedScene = ResourceLoader.load(world_path)
	world_node.add_child(scene.instance())

	# move the player to spawn in the right place
	spawn_point = world_node.find_node("SpawnPoint", true, false)
	is_spawning_player = is_instance_valid(spawn_point)

	observation_handler.reset()


func idle(delta: float) -> void:
	# move tracking camera, after physics has been done
	camera_controller.do_physics(delta)


func spawn() -> void:
	# spawning must happen before `flush_queries`
	player.set_spawn(spawn_point.global_transform)
	is_spawning_player = false
	input_collector.reset()


func get_action() -> Dictionary:
	var normalized_action = input_collector.to_normalized_relative_action(player)
	var scaled_action = input_collector.scaled_relative_action_from_normalized_relative_action(
		normalized_action, player
	)
	return {"scaled_action": scaled_action, "normalized_action": normalized_action}


func is_warming_up(_delta: float) -> bool:
	return false


func should_try_to_spawn_player() -> bool:
	return is_spawning_player


func do_physics(delta: float) -> void:
	# TODO: ideally this would be after spawn but playback is finnicky
	before_physics()

	var is_exiting_early = is_warming_up(delta)
	if is_exiting_early:
		input_collector.reset()
		return

	read_input_before_physics()

	if should_try_to_spawn_player():
		spawn()
		return

	var actions = get_action()
	player.apply_action(actions.scaled_action, delta)
	input_collector.reset()


func _get_references_to_clean() -> Array:
	return [input_collector, rng, observation_handler, debug_logger, camera_controller]


func quit() -> void:
	var possibly_dangling_references = _get_references_to_clean()
	for ref in possibly_dangling_references:
		if ref:
			ref.unreference()
			ref = null


func read_input_before_physics() -> void:
	pass


func read_input_from_event(event: InputEvent) -> void:
	input_collector.read_input_from_event(event)


func log_debug_info() -> void:
	if frame == 0 and is_debugging_output_requested():
		debug_logger.open_debug_file(avalon_spec.dir_root, episode)

	if is_debugging_output_requested():
		debug_logger.write_debug_output(episode, frame)
