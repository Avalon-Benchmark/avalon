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


static func read_world_metadata_from_world_path(world_path: String) -> Dictionary:
	return _read_world_metadata(world_path.replace("main.tscn", "meta.json"))


static func is_metadatda_for_world_present(world_path: String) -> bool:
	var metadata_path = world_path.replace("main.tscn", "meta.json")
	var directory = Directory.new()
	return directory.file_exists(metadata_path)


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


static func get_time_to_complete_level(data: Dictionary) -> float:
	var size_in_meters: float = data["size_in_meters"]
	# var food_count: int = data["food_count"]
	var total_distance: float = data["total_distance"]
	var is_visibility_required: bool = data["is_visibility_required"]

	# always give at least this much time
	var min_seconds = 120.0

	var seconds = 0.0
	if is_visibility_required:
		# 200 meters should take about 600 seconds
		var seconds_per_meter = 600.0 / 200.0
		seconds = seconds_per_meter * total_distance
	else:
		# time to explore is related to the area of the world
		var world_area = size_in_meters * size_in_meters
		# and for a 50m x 50m we probably want to give you about 100 seconds to explore
		var seconds_per_meter_squared = 100 / (50 * 50)
		seconds = world_area * seconds_per_meter_squared

	if seconds < min_seconds:
		seconds = min_seconds

	return seconds


static func get_time_to_complete_level_by_world_id(world_id: String) -> float:
	var parsed_world_id = get_parsed_world_id(world_id)
	var task = parsed_world_id["task"]
	if task in ["survive", "find", "gather", "navigate"]:
		return 15.0 * 60.0
	elif task in ["stack", "carry", "explore"]:
		return 10.0 * 60.0
	else:
		return 5.0 * 60.0


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
	var _seed = parsed_world_id["seed"].substr(
		len(parsed_world_id["seed"]) - 5, len(parsed_world_id["seed"])
	)
	return "%s, seed: %s/%s" % [parsed_world_id["task"], _seed, parsed_world_id["difficulty"]]
