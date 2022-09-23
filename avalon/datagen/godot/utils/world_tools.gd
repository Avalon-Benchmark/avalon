extends Reference

class_name WorldTools

const GLOBAL_WORLD_METADATA := {}


static func has_world(world_id: String) -> bool:
	return GLOBAL_WORLD_METADATA.get(world_id) != null


static func add_global_world_metadata(world_id: String, metadata: Dictionary):
	GLOBAL_WORLD_METADATA[world_id] = metadata


static func get_global_world_metadata(world_id: String):
	return GLOBAL_WORLD_METADATA.get(world_id)


static func set_global_world_metadata_from_included_levels():
	var path = "res://worlds"
	var dir = Directory.new()
	if dir.open(path) == OK:
		dir.list_dir_begin(true, true)
		var world_id = dir.get_next()
		while world_id != "":
			GLOBAL_WORLD_METADATA[world_id] = read_world_metadata_from_world_id(world_id)
			world_id = dir.get_next()
		dir.list_dir_end()
	else:
		print("An error occurred when trying to access the path when reading metadata.")


static func get_world_id_from_world_path(world_path: String) -> String:
	var file_parts = world_path.split("/")
	return file_parts[len(file_parts) - 2]


static func get_world_path_from_world_id(world_id: String) -> String:
	return "res://worlds/%s/main.tscn" % world_id


static func read_world_metadata_from_world_id(world_id: String) -> Dictionary:
	var metadata_path = "res://worlds/%s/%s" % [world_id, "meta.json"]
	return _read_world_metadata(metadata_path)


static func _read_world_metadata(metadata_path: String) -> Dictionary:
	var directory = Directory.new()
	if directory.file_exists(metadata_path):
		var file = File.new()
		file.open(metadata_path, File.READ)
		var content = file.get_as_text()
		return JSON.parse(content).result
	else:
		return HARD.assert(false, "could not find %s" % metadata_path)


static func get_time_to_complete_level_by_world_id(world_id: String) -> float:
	var parsed_world_id = get_parsed_world_id(world_id)
	var task = parsed_world_id["task"]
	if task in ["find", "gather", "navigate"]:
		return 15.0 * 60.0
	elif task in ["survive", "stack", "carry", "explore"]:
		return 10.0 * 60.0
	else:
		return 5.0 * 60.0


static func get_task_success_result(
	world_id: String,
	is_done: bool,
	is_dead: bool,
	is_timeout: bool,
	has_more_hit_points_than_start: bool
) -> bool:
	var parsed_world_id = get_parsed_world_id(world_id)
	var task = parsed_world_id["task"]
	if task in ["survive", "gather"]:
		return not is_dead and has_more_hit_points_than_start
	else:
		return is_done and not is_dead and not is_timeout


static func is_practice_world(world: String) -> bool:
	return "practice" in world


static func get_parsed_world_id(world_id: String) -> Dictionary:
	var is_practice = is_practice_world(world_id)
	var start = 0
	if is_practice:
		start += 1
	var parts = world_id.split("__")
	var task_name = parts[0 + start]
	var world_seed = parts[1 + start]
	var difficulty = float(parts[2 + start].replace("_", "."))
	return {
		"world_id": world_id,
		"task": task_name,
		"seed": world_seed,
		"difficulty": difficulty,
		"is_practice": is_practice
	}


static func get_long_name_from_world_id(world_id: String) -> String:
	var parsed_world_id = get_parsed_world_id(world_id)
	return (
		"task: %s, seed: %s, difficulty: %s"
		% [parsed_world_id["task"], parsed_world_id["seed"], parsed_world_id["difficulty"]]
	)


static func get_short_name_from_world_id(world_id: String) -> String:
	var parsed_world_id = get_parsed_world_id(world_id)
	# TODO: sigh this function is pointless now
	var _seed = parsed_world_id["seed"]
	return "%s, seed: %s/%s" % [parsed_world_id["task"], _seed, parsed_world_id["difficulty"]]
