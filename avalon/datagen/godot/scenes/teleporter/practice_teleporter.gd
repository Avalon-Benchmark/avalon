extends Teleporter

class_name PracticeTeleporter

var task_select: TextSelect
var world_select: TextSelect
var title: LabelGUI

var parsed_worlds_by_task_name = {}


class DifficultySorter:
	static func sort_by_difficulty(a, b):
		if a["difficulty"] < b["difficulty"]:
			return true
		return false


func _ready():
	title = get_node("teleporter_gui/title/text_panel")
	task_select = get_node("teleporter_gui/tasks")
	world_select = get_node("teleporter_gui/world")

	for world_id in WorldTools.GLOBAL_WORLD_METADATA.keys():
		if WorldTools.is_practice_world(world_id):
			var parsed_world_id = WorldTools.get_parsed_world_id(world_id)
			var task_name = parsed_world_id["task"]
			if task_name in parsed_worlds_by_task_name:
				parsed_worlds_by_task_name[task_name].append(parsed_world_id)
			else:
				parsed_worlds_by_task_name[task_name] = [parsed_world_id]

	var num_worlds_per_tasks = 0
	for key in parsed_worlds_by_task_name:
		var value = parsed_worlds_by_task_name[key]
		value.sort_custom(DifficultySorter, "sort_by_difficulty")
		parsed_worlds_by_task_name[key] = value
		num_worlds_per_tasks = len(value)

	var task_options = Array(parsed_worlds_by_task_name.keys())
	task_options.sort()
	task_select.options = task_options
	world_select.options = Array(range(1, num_worlds_per_tasks + 1))


func _get_world_id_from_options() -> String:
	var task_name = task_select.get_current_option()
	var world = int(world_select.get_current_option())

	if task_name in parsed_worlds_by_task_name:
		var index = world - 1
		var worlds = parsed_worlds_by_task_name[task_name]
		if index >= len(worlds):
			return ""
		else:
			var parsed_world = worlds[index]
			return parsed_world["world_id"]

	print("Could not find world for (%s, %s)" % [task_name, world])
	return ""


func set_message(text: String):
	title.text = text


func get_selected_options() -> Dictionary:
	return {
		"task": task_select.get_current_option(),
		"world": int(world_select.get_current_option()) - 1,
	}


func get_option_state() -> Dictionary:
	return {"task": task_select.curr_option_index, "world": world_select.curr_option_index}


func from_option_state(state: Dictionary) -> void:
	if len(state):
		task_select.curr_option_index = state["task"]
		world_select.curr_option_index = state["world"]


func get_world_id() -> String:
	return _get_world_id_from_options()
