extends Recorder

class_name RemoteRecorder

var client: AvalonClient


func _init(_client):
	client = _client


func save(key: String, file_type: String) -> void:
	var filename = _get_filename(key, file_type)
	var data = _get_data_by_key_from_state(key)
	print("uploading to %s" % filename)
	var binary_data: PoolByteArray = []
	match file_type:
		JSON_FILE:
			binary_data = JSON.print(data).to_ascii()
		BINARY_FILE:
			for byte_array in data:
				binary_data.append_array(byte_array)
	var _res = client.upload(
		current_state["apk_version"],
		current_state["world_id"],
		current_state["run_id"],
		binary_data,
		filename
	)
	print("upload succeeded with response %s" % _res)


func start(apk_version: String, world_path: String, episode: int) -> void:
	.start(apk_version, world_path, episode)

	var _res = client.start_world(current_state["world_id"], current_state["run_id"])
	print("start succeeded with response %s" % _res)
