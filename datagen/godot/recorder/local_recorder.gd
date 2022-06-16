extends Recorder

class_name LocalRecorder

var dir_root: String


func _init(_dir_root):
	dir_root = _dir_root


func _open_file(path: String) -> File:
	var actual_path = Tools.file_create(path)
	var file = File.new()
	var error = file.open(actual_path, File.READ_WRITE)
	HARD.assert(error == OK, "IO error: %s for file at path %s" % [error, path])
	return file


func _get_file_path(filename: String) -> String:
	return "%s/%s/%s/%s" % [dir_root, current_state["world_id"], current_state["run_id"], filename]


func save(key: String, file_type: String) -> void:
	var filename = _get_filename(key, file_type)
	var data = _get_data_by_key_from_state(key)
	var path = _get_file_path(filename)
	var file = _open_file(path)
	print("saving to %s" % file.get_path())
	match file_type:
		JSON_FILE:
			var line := JSON.print(data)
			file.store_line(line)
		BINARY_FILE:
			for item in data:
				file.store_buffer(item)
	file.close()
