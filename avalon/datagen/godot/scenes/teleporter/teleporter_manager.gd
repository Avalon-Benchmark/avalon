extends Node

class_name TeleporterManager

var client: AvalonClient
var apk_version: String
var task_teleporter_world_id := ""
var last_teleporter_world_id := ""
var practice_teleporter_state := {}


func _init(_client: AvalonClient, _apk_version: String):
	client = _client
	apk_version = _apk_version


func on_scene_exit(teleporter: Teleporter) -> void:
	if teleporter is PracticeTeleporter:
		practice_teleporter_state = teleporter.get_option_state()
	if teleporter is TaskTeleporter:
		last_teleporter_world_id = task_teleporter_world_id


func on_scene_enter(teleporter: Teleporter) -> void:
	if teleporter is PracticeTeleporter:
		teleporter.from_option_state(practice_teleporter_state)

	if teleporter is TaskTeleporter:
		var last_world_id := ""
		var world_id := ""
		var metadata := {}
		if client:
			var response = client.get_world_info(apk_version)
			last_world_id = response["last_world_id"]
			world_id = response["world_id"]
			metadata = response.get("world_metadata")
			if not WorldTools.has_world(world_id):
				var world_zip_path = client.download(apk_version, world_id)
				var world_is_ready = ProjectSettings.load_resource_pack(world_zip_path)
				HARD.assert(world_is_ready, "can't load world from file: %s", world_zip_path)
				WorldTools.add_global_world_metadata(world_id, metadata)
		else:
			var worlds = WorldTools.GLOBAL_WORLD_METADATA.keys()
			if worlds:
				worlds.shuffle()
				last_world_id = last_teleporter_world_id
				world_id = worlds[0]

		if not world_id:
			teleporter.set_message("No more worlds to conquer.", 0, "")
			return

		task_teleporter_world_id = world_id
		metadata = WorldTools.get_global_world_metadata(world_id)

		var time_to_complete_in_minutes = ceil(
			WorldTools.get_time_to_complete_level_by_world_id(world_id) / 60
		)
		var readable_next_world = WorldTools.get_short_name_from_world_id(world_id)
		var readable_last_world = ""
		if last_world_id:
			readable_last_world = (
				"Last world: %s"
				% WorldTools.get_short_name_from_world_id(last_world_id)
			)

		teleporter.set_message(
			"Up next: %s" % readable_next_world, time_to_complete_in_minutes, readable_last_world
		)


func disable(teleporter: Teleporter) -> void:
	teleporter.is_disabled = true
	if teleporter is PracticeTeleporter:
		teleporter.set_message("Disabled: Invalid APK Version")
	if teleporter is TaskTeleporter:
		teleporter.set_message("Disabled: Invalid APK Version", 0, "")


func get_world_path_for_teleporter(teleporter: Teleporter) -> String:
	if teleporter is PracticeTeleporter:
		var world_id = teleporter.get_world_id()
		return WorldTools.get_world_path_from_world_id(world_id)
	if teleporter is TaskTeleporter:
		return WorldTools.get_world_path_from_world_id(task_teleporter_world_id)
	return HARD.assert(false)
