extends AvalonAction

class_name DebugCameraAction

var offset := Vector3.ZERO
var rotation := Vector3.ZERO

var is_facing_tracked := false
var is_frame_advanced := true

var tracked_node: String


func reset() -> void:
	.reset()


func to_byte_array() -> PoolByteArray:
	# NOTE: the order must be the same as how we send bytes
	var stream = StreamPeerBuffer.new()
	for vec in [offset, rotation]:
		stream.put_float(vec.x)
		stream.put_float(vec.y)
		stream.put_float(vec.z)
	stream.put_float(1.0 if is_facing_tracked else 0.0)
	stream.put_float(1.0 if is_frame_advanced else 0.0)
	stream.put_utf8_string(tracked_node)
	return stream.data_array


func get_transform() -> Transform:
	return Transform(Basis(rotation), offset)
