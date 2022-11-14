extends Reference

class_name DebugLogger

var root: Node
var current_debug_file: File


func _init(_root: Node):
	root = _root


func _get_items_recursively(node: Node) -> Array:
	var results = []
	for child in node.get_children():
		if child is Item:
			results.push_back(child.to_dict())
		else:
			results += _get_items_recursively(child)
	return results


func _get_dynamic_item_snapshot() -> Array:
	var dynamic_item_group = root.get_node_or_null(CONST.DYNAMIC_TRACKER_NODE_PATH)
	var results = []
	if dynamic_item_group:
		results += _get_items_recursively(dynamic_item_group)
	return results


func _get_file_path(dir_root: String, episode: int, file_name: String) -> String:
	return "{dir_root}/{episode_id}/{file_name}".format(
		{
			"dir_root": dir_root,
			"episode_id": "%06d" % episode,
			"file_name": file_name,
		}
	)


func open_debug_file(dir_root: String, episode: int, is_truncated_on_open: bool = true) -> void:
	if current_debug_file and current_debug_file.is_open() == true:
		current_debug_file.close()
	var path = Tools.file_create(_get_file_path(dir_root, episode, "debug.json"))
	current_debug_file = File.new()
	var mode = File.WRITE_READ if is_truncated_on_open else File.READ_WRITE
	var error = current_debug_file.open(path, mode)
	HARD.assert(error == OK, "IO error: %s for file at path %s" % [error, path])
	if HARD.mode():
		print("Logging debug output to %s" % path)


func write_debug_output(episode: int, frame: int) -> void:
	var item_data := _get_dynamic_item_snapshot()
	var items_json := JSON.print({"time": frame, "episode": episode, "items": item_data})
	current_debug_file.store_line(items_json)
	var error = current_debug_file.get_error()
	HARD.assert(error == OK, "IO error: %s", current_debug_file.get_path())
	if HARD.mode():
		current_debug_file.flush()
