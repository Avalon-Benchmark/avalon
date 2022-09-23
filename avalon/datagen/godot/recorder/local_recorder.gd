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
	return (
		"%s/%s/%s/%s.gz"
		% [dir_root, current_state["world_id"], current_state["run_id"], filename]
	)


func save(key: String, file_type: String) -> void:
	var filename = _get_filename(key, file_type)
	var data = _get_data_by_key_from_state(key)
	var path = _get_file_path(filename)
	var file = _open_file(path)
	print("saving to %s" % file.get_path())
	var binary_data: PoolByteArray = []
	match file_type:
		JSON_FILE:
			binary_data = JSON.print(data).to_ascii()
		BINARY_FILE:
			for byte_array in data:
				binary_data.append_array(byte_array)
	file.store_buffer(binary_data.compress(File.COMPRESSION_GZIP))
	file.close()
