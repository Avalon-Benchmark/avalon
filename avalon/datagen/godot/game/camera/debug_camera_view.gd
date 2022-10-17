extends Viewport

class_name DebugCameraView

var input_collector: DebugCameraInputCollector
var debug_camera_node: Node

var _cached_transform


func _init():
	input_collector = DebugCameraInputCollector.new()


func insert_as_overlay(camera_controller, root: Node, size: Vector2) -> void:
	self.size = size
	var camera = camera_controller.get_primary_camera()
	var debug_camera = TrackingCamera.new()
	debug_camera.transform = camera.transform
	debug_camera.cull_mask = camera.cull_mask
	debug_camera.near = camera.near
	debug_camera.far = camera.far
	debug_camera.keep_aspect = camera.keep_aspect
	debug_camera.is_enabled = camera.is_enabled
	debug_camera.is_tracking_relative_to_initial_transform = camera.is_tracking_relative_to_initial_transform
	debug_camera.is_automatically_looking_at_tracked = camera.is_automatically_looking_at_tracked
	debug_camera.name = "debug_camera"

	add_child(debug_camera)
	debug_camera.make_current()

	var container = ViewportContainer.new()
	container.show_on_top = true
	container.set_anchors_preset(Control.PRESET_WIDE)
	container.add_child(self)

	debug_camera_node = Node.new()
	debug_camera_node.name = "debug_camera"
	root.add_child(debug_camera_node)

	debug_camera_node.add_child(container)
	camera_controller.cameras.push_front(debug_camera)


func remove_from(camera_controller):
	HARD.assert(
		camera_controller.get_primary_camera() == get_camera(),
		"attempted to remove debug camera when not debugging"
	)

	# TODO with direct viewport rendering, is this necessary?
	var vpc = get_parent()
	debug_camera_node.remove_child(vpc)
	vpc.queue_free()
	camera_controller.cameras.pop_front()
	camera_controller.debug_view = null


func apply_action(action: DebugCameraAction):
	var camera = get_camera()
	if not is_already_tracking(action.tracked_node):
		var tracked = get_tree().root.find_node(action.tracked_node, true, false)
		if tracked == null:
			print("DEBUG_WARNING: Could not find node: '%s'" % action.tracked_node)
			return
		if not tracked is Spatial:
			print("DEBUG_WARNING: tracked_node %s must be a Spatial node" % action.tracked_node)
			return
		print("debug_camera now tracking %s" % tracked)
		camera.tracked = tracked

	if action.rotation != Vector3.ZERO or action.offset != Vector3.ZERO:
		camera.is_tracking_relative_to_initial_transform = true
		_cached_transform = camera.initial_transform
		camera.initial_transform = action.get_transform()
	elif _cached_transform != null:
		camera.is_tracking_relative_to_initial_transform = false
		camera.initial_transform = _cached_transform
		_cached_transform = null

	camera.is_automatically_looking_at_tracked = action.is_facing_tracked


func is_already_tracking(tracked_node: String) -> bool:
	var camera = get_camera()
	if camera.tracked == null:
		return false
	var full_path = camera.tracked.get_path()
	return ("%s" % full_path).ends_with(tracked_node)


func read_and_apply_action(data: PoolByteArray) -> bool:
	var stream = StreamPeerBuffer.new()
	stream.data_array = data
	input_collector.read_input_from_pipe(stream)
	apply_action(input_collector.action)
	return input_collector.action.is_frame_advanced


# TODO not sure if the A is actually depth here post-conversion but we ignore usually anyways
func get_rgbx_data():
	var image_data = get_texture().get_data()
	image_data.convert(Image.FORMAT_RGBA8)
	return image_data
