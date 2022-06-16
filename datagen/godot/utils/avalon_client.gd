extends Reference

class_name AvalonClient

const VERIFY_APK_VERSION_URL = "/verify/{apk_version}/"
const GET_WORLD_URL = "/get_world/{apk_version}/{user_id}/"
const START_WORLD_URL = "/start_world/{world_id}/{user_id}/{run_id}/"
const RESET_WORLD_URL = "/reset_world/{world_id}/{user_id}/{run_id}/"
const UPLOAD_URL = "/upload/{apk_version}/{world_id}/{user_id}/{run_id}/{filename}"
const DOWNLOAD_URL = "/download/{apk_version}/{world_id}"

var api: RequestHelper
var user_id: String


func _init(host: String, port: int, user: String):
	api = RequestHelper.new(host, port)
	user_id = user


func verify(apk_version: String) -> Dictionary:
	var path = VERIFY_APK_VERSION_URL.format({"apk_version": apk_version})
	return api.GET_JSON(path)


func get_world_info(apk_version: String) -> Dictionary:
	var path = GET_WORLD_URL.format({"apk_version": apk_version, "user_id": user_id})
	return api.GET_JSON(path)


func start_world(world_id: String, run_id: String) -> Dictionary:
	var path = START_WORLD_URL.format({"world_id": world_id, "user_id": user_id, "run_id": run_id})
	return api.GET_JSON(path)


func reset_world(world_id: String, run_id: String) -> Dictionary:
	var path = RESET_WORLD_URL.format({"world_id": world_id, "user_id": user_id, "run_id": run_id})
	return api.GET_JSON(path)


func upload(
	apk_version: String, world_id: String, run_id: String, data: PoolByteArray, filename: String
) -> Dictionary:
	var path = UPLOAD_URL.format(
		{
			"apk_version": apk_version,
			"world_id": world_id,
			"user_id": user_id,
			"run_id": run_id,
			"filename": filename
		}
	)
	return api.POST_JSON(path, data)


func download(apk_version: String, world_id: String) -> String:
	var path = DOWNLOAD_URL.format(
		{
			"apk_version": apk_version,
			"world_id": world_id,
		}
	)
	var world_path = "user://world__%s.zip" % world_id
	var world_data = api.GET_FILE(path)
	var world_file = File.new()
	HARD.assert(world_file.open(world_path, File.WRITE) == OK)
	world_file.store_buffer(world_data)
	world_file.close()
	return world_path
