extends Reference

class_name RequestHelper

var HTTP_HOST: String
var HTTP_PORT: int


class AvalonRequest:
	extends Reference

	var host: String
	var port: int
	var path: String

	var time_start: int
	var time_abort: int

	var head := PoolStringArray(["Accept: application/json"])

	var http: HTTPClient
	var error := 0
	var state := 0

	func _init(http_host: String, http_port: int, http_path: String):
		host = http_host
		port = http_port
		path = http_path
		http = HTTPClient.new()
		http.read_chunk_size = 1024 * 1024

	func request(data = null, max_time := 60):
		prints("starting request:", host, path, len(data) if data else 0)
		time_start = OS.get_ticks_msec()
		time_abort = time_start + 1000 * max_time
		if http.connect_to_host(host, port) != OK:
			return null
		if not await(HTTPClient.STATUS_CONNECTED):
			return null
		if data == null:
			if http.request(HTTPClient.METHOD_GET, path, head) != OK:
				return null
		else:
			if http.request_raw(HTTPClient.METHOD_POST, path, head, data) != OK:
				return null
		if not await(HTTPClient.STATUS_BODY):
			return null
		var response = await_response()
		prints("finished request:", host, path, len(response) if response else 0)
		return response

	func await(state_done: int) -> bool:
		while timed_poll():
			if state == state_done:
				return true
			OS.delay_msec(1)
		return false

	func await_response():
		var response = PoolByteArray()
		var expected_size = http.get_response_body_length()
		var expected_hash = http.get_response_headers_as_dictionary().get("X-MD5")
		while state == HTTPClient.STATUS_BODY:
			response.append_array(http.read_response_body_chunk())
			if not timed_wait():
				break
		http.close()
		if (expected_size > 0) and (expected_size != len(response)):
			prints("response size mismatch:", host, path, expected_size, len(response))
			return null
		elif (expected_hash != null) and (expected_hash != bytes_hash(response)):
			prints("response hash mismatch:", host, path)
			return null
		else:
			return response

	func timed_poll() -> bool:
		error = http.poll()
		state = http.get_status()
		if error != OK:
			prints("request error:", host, path, error, state)
			return false
		elif time_abort < OS.get_ticks_msec():
			prints("request timed out:", host, path)
			http.close()
			return false
		return true

	func timed_wait() -> bool:
		error = http.poll()
		state = http.get_status()
		if time_abort < OS.get_ticks_msec():
			prints("request timed out:", host, path)
			return false
		else:
			return true

	static func bytes_hash(bytes: PoolByteArray) -> String:
		var context := HashingContext.new()
		var _error = 0
		_error = context.start(HashingContext.HASH_MD5)
		_error = context.update(bytes)
		return context.finish().hex_encode()


func _init(http_host: String, http_port: int):
	HTTP_HOST = http_host
	HTTP_PORT = http_port


func GET_JSON(http_path: String, max_tries := 3, max_time := 30) -> Dictionary:
	var request: AvalonRequest
	var response = null
	for _try in range(max_tries):
		request = AvalonRequest.new(HTTP_HOST, HTTP_PORT, http_path)
		response = request.request(null, max_time)
		if response != null:
			return bytes_json(response)
	return HARD.stop("request failed: %s", http_path)


func GET_FILE(http_path: String, max_tries := 3, max_time := 90) -> PoolByteArray:
	var request: AvalonRequest
	var response = null
	for _try in range(max_tries):
		request = AvalonRequest.new(HTTP_HOST, HTTP_PORT, http_path)
		response = request.request(null, max_time)
		if response != null:
			return response
	return HARD.stop("request failed: %s", http_path)


func POST_JSON(http_path: String, post_data: PoolByteArray, max_tries := 5, max_time := 30) -> Dictionary:
	post_data = post_data.compress(File.COMPRESSION_GZIP)
	var request: AvalonRequest
	var response = null
	for _try in range(max_tries):
		request = AvalonRequest.new(HTTP_HOST, HTTP_PORT, http_path)
		response = request.request(post_data, max_time)
		if response != null:
			return bytes_json(response)
	return HARD.stop("request failed: %s", http_path)


static func bytes_json(bytes: PoolByteArray) -> Dictionary:
	var utf8 := bytes.get_string_from_utf8()
	var json := JSON.parse(utf8)
	if json.error == OK:
		return json.result
	else:
		return HARD.stop("json parse error:", json.error_string)
