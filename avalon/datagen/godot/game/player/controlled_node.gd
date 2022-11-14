extends Node

class_name ControlledNode

export var spawn_point_name: String


func apply_action(_action, _delta: float) -> void:
	HARD.assert(false, "not implemented")


func reset_on_new_world() -> void:
	HARD.assert(false, "not implemented")


func get_spawn_point(root: Viewport) -> Spatial:
	var expected_path = "%s/%s" % [CONST.DYNAMIC_TRACKER_NODE_PATH, spawn_point_name]
	var spawn_point: Spatial = root.get_node_or_null(expected_path)
	if spawn_point:
		return spawn_point
	spawn_point = root.get_node(CONST.WORLD_NODE_PATH).find_node(spawn_point_name, true, false)
	return spawn_point


func set_spawn(_spawn_transform: Transform) -> void:
	HARD.assert(false, "ControlledNodes must implement set_spawn")


func spawn_into(root: Viewport) -> void:
	var spawn_point = get_spawn_point(root)
	HARD.assert(spawn_point != null, "%s's spawn point '%s' not found" % [self, spawn_point_name])
	set_spawn(spawn_point.global_transform)
