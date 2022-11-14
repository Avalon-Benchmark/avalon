extends Reference

class_name SnapshotHandler

# binary .scn is faster/higher precision than .tscn,
# but has a bug where empty arrays in player.gd are conflated and somehow become identical.
const _SNAPSHOT_FORMAT := "tscn"


static func get_snapshot_path(dir_root: String) -> String:
	return Tools.update_file_path("%s/%s" % [dir_root, CONST.SNAPSHOT_SUBPATH])


static func snapshot_node(node: Node, snapshot_path: String) -> void:
	Tools.set_owner_of_subtree(node, node.get_children())
	var packed_scene = PackedScene.new()
	packed_scene.pack(node)
	var file_name = "%s/%s.%s" % [snapshot_path, node.name, _SNAPSHOT_FORMAT]
	HARD.assert(OK == ResourceSaver.save(file_name, packed_scene))


static func _get_current_frame_snapshot_path(game_manager: GameManager) -> String:
	var file_parts = game_manager.current_world_path.split("/")
	var world_id = file_parts[len(file_parts) - 2]
	var uuid_or_static_dir = file_parts[len(file_parts) - 3]
	var snapshot_path = get_snapshot_path(game_manager.avalon_spec.dir_root)
	var world_snapshots_path = "%s/%s/%s" % [snapshot_path, uuid_or_static_dir, world_id]
	return "%s/ep_%s_frame_%s" % [world_snapshots_path, game_manager.episode, game_manager.frame]


static func _write_context_json(snapshot_path: String, game_manager: GameManager) -> void:
	var current_world_path = game_manager.current_world_path
	var episode = game_manager.episode
	var frame = game_manager.frame
	var controlled_node_names = []
	for node in game_manager.controlled_nodes:
		controlled_node_names.append(node.name)
	HARD.assert(OK == Directory.new().make_dir_recursive(snapshot_path))
	Tools.write_json_to_path(
		"%s/%s" % [snapshot_path, CONST.SNAPSHOT_JSON],
		{
			"world_path": current_world_path,
			"episode_seed": episode,
			"frame": frame,
			"controlled_node_names": controlled_node_names
		}
	)
	# TODO copy and reload debug_logger.current_debug_file

	var path_details = "world_path=%s, snapshot_path=%s" % [current_world_path, snapshot_path]
	print("Saving snapshot of episode %s frame %s (%s)" % [episode, frame, path_details])


static func save_snapshot(game_manager: GameManager) -> String:
	var snapshot_path = _get_current_frame_snapshot_path(game_manager)
	_write_context_json(snapshot_path, game_manager)
	# TODO copy and reload debug_logger.current_debug_file

	for controlled_node in game_manager.controlled_nodes:
		snapshot_node(controlled_node, snapshot_path)

	var dynamic_tracker = game_manager.root.get_node(CONST.DYNAMIC_TRACKER_NODE_PATH)
	for item in dynamic_tracker.get_children():
		if item is Animal:
			item.persist_behaviors()
	snapshot_node(dynamic_tracker, snapshot_path)
	return snapshot_path


static func get_snapshot_node_instance(snapshot_path: String, node_name: String) -> Node:
	var snapshot: PackedScene = load("%s/%s.%s" % [snapshot_path, node_name, _SNAPSHOT_FORMAT])
	HARD.assert(snapshot != null, "failed to load snapshot scene from %s" % snapshot_path)
	return snapshot.instance()


static func swap_node_with_snapshot(node: Node, snapshot_path: String) -> Node:
	var instance = get_snapshot_node_instance(snapshot_path, node.name)
	var parent = node.get_parent()
	parent.remove_child(node)
	node.queue_free()
	parent.add_child(instance)
	Tools.set_owner_of_subtree(instance, instance.get_children())
	return instance


static func load_snapshot(game_manager: GameManager, snapshot_path: String):
	var snapshot_context = Tools.read_json_from_path("%s/%s" % [snapshot_path, CONST.SNAPSHOT_JSON])
	var is_snapshot_of_current_world = (
		game_manager.current_world_path
		== snapshot_context["world_path"]
	)

	game_manager.current_world_path = snapshot_context["world_path"]
	game_manager.episode = snapshot_context["episode_seed"]
	game_manager.frame = snapshot_context["frame"]
	game_manager.set_time_and_seed()

	var current_world_path = game_manager.current_world_path
	var episode = game_manager.episode
	var frame = game_manager.frame

	var path_info = "world_path=%s, snapshot_path=%s" % [current_world_path, snapshot_path]
	var swap_info = " of current world" if is_snapshot_of_current_world else " of swapped in world"
	print("Loading snapshot %s: episode %s frame %s (%s)" % [swap_info, episode, frame, path_info])

	if game_manager.camera_controller.debug_view != null:
		game_manager.camera_controller.remove_debug_camera()

	if not is_snapshot_of_current_world:
		game_manager.swap_in_scene(load(current_world_path).instance())

	var controlled_names: Array = snapshot_context.get("controlled_node_names", ["player"])
	HARD.assert(
		len(game_manager.controlled_nodes) == len(controlled_names),
		"Different number of configured and snapshotted controlled nodes"
	)
	game_manager.player = null
	for i in len(game_manager.controlled_nodes):
		var current: ControlledNode = game_manager.controlled_nodes[i]
		HARD.assert(
			controlled_names.has(current.name), "missing expected controlled node %s" % current.name
		)
		var new_node = swap_node_with_snapshot(current, snapshot_path)
		game_manager.controlled_nodes[i] = new_node

	game_manager.player = game_manager.scene_root.find_node("player", true, false)
	HARD.assert(game_manager.player != null, "Failed to load player from snapshot")

	var dynamic_tracker = game_manager.root.get_node(CONST.DYNAMIC_TRACKER_NODE_PATH)
	var _new_node = swap_node_with_snapshot(dynamic_tracker, snapshot_path)

	game_manager.camera_controller.setup()
	game_manager.observation_handler.reset()
