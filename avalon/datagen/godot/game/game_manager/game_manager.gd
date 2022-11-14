extends Reference

class_name GameManager

var root: Node
var avalon_spec: SimSpec
var rng: CleanRNG
var player: Player
var input_collector: CombinedInputCollector
var camera_controller: TrackingCameraController
var observation_handler: ObservationHandler
var debug_logger: DebugLogger

var episode := 0
var frame := 0
var warmup_frame := 0

var scene_root: Node
var world_node: Node
var current_world_path := "res://scenes/entry.tscn"

var is_spawn_on_new_world_pending := false

var default_scene_path := "res://scenes/empty.tscn"
var selected_features := {}

var controlled_nodes: Array


func _init(_root: Node, _avalon_spec: SimSpec):
	avalon_spec = _avalon_spec
	root = _root

	if avalon_spec.episode_seed != null:
		episode = avalon_spec.episode_seed

	rng = CleanRNG.new()
	scene_root = root.get_node(CONST.SCENE_ROOT_NODE_PATH)
	world_node = root.get_node(CONST.WORLD_NODE_PATH)

	if is_debugging_output_requested():
		debug_logger = DebugLogger.new(root)

	_configure_controlled_nodes_and_collector(avalon_spec.get_controlled_node_specs())
	for node in controlled_nodes:
		scene_root.add_child(node)

	camera_controller = TrackingCameraController.new(
		root, avalon_spec.get_resolution(), "physical_head", is_adding_debugging_views()
	)

	observation_handler = ObservationHandler.new(root, camera_controller)


func _configure_controlled_nodes_and_collector(controlled_node_specs: Array) -> void:
	controlled_nodes = []
	var collectors = []
	for spec in controlled_node_specs:
		HARD.assert(
			spec is ControlledNodeSpec,
			"get_controlled_node_specs should return all ControlledNodeSpecs, but got %s" % spec
		)
		var collector = spec.get_input_collector()
		var node = spec.get_node()
		HARD.assert(collector is InputCollector and node is ControlledNode)
		HARD.assert(node.spawn_point_name != "", "Failed to set spawn_point_name")
		controlled_nodes.append(node)
		collectors.append(collector)

		if node is Player:
			if player != null:
				print("Warning: 2+ player nodes configured. Only the first will be the POV player")
				continue
			player = node
	HARD.assert(player != null, "get_controlled_node_specs did not return a player")
	input_collector = CombinedInputCollector.new(collectors)


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


func swap_in_scene(scene_instance: Node) -> void:
	if camera_controller.debug_view != null:
		camera_controller.remove_debug_camera()

	for child in world_node.get_children():
		world_node.remove_child(child)
		child.queue_free()

	world_node.add_child(scene_instance)


func advance_episode(world_path: String) -> void:
	current_world_path = world_path

	# reset internal player states before moving to a new world
	for controlled in controlled_nodes:
		controlled.reset_on_new_world()

	frame = 0
	set_time_and_seed()

	if HARD.mode():
		print("Loading world %s with episode seed %s" % [world_path, episode])
	var scene: PackedScene = ResourceLoader.load(world_path)
	swap_in_scene(scene.instance())

	is_spawn_on_new_world_pending = true

	observation_handler.reset()


func idle(delta: float) -> void:
	# move tracking camera, after physics has been done
	camera_controller.do_physics(delta)


func spawn_controlled_nodes() -> void:
	# spawning must happen before `flush_queries`
	for controlled_node in controlled_nodes:
		controlled_node.spawn_into(root)
	is_spawn_on_new_world_pending = false
	input_collector.reset()


func is_warming_up(_delta: float) -> bool:
	return false


func is_safe_to_spawn_on_new_world() -> bool:
	return true


func apply_collected_actions(delta: float) -> Array:
	var actions = input_collector.get_actions(controlled_nodes)
	for index in len(actions):
		var cn: ControlledNode = controlled_nodes[index]
		cn.apply_action(actions[index], delta)
	return actions


func do_physics(delta: float) -> void:
	# TODO: ideally this would be after spawn but playback is finnicky
	before_physics()

	var is_exiting_early = is_warming_up(delta)
	if is_exiting_early:
		input_collector.reset()
		return

	read_input_before_physics()

	if is_spawn_on_new_world_pending and is_safe_to_spawn_on_new_world():
		spawn_controlled_nodes()
		return

	var _actions = apply_collected_actions(delta)
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
