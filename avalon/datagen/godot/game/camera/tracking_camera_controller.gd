extends Reference

class_name TrackingCameraController

var tracked_node
var tracked: Spatial
var cameras = []
var root: Node
var resolution: Vector2
var rgb_viewport: Viewport
var top_down_viewport: Viewport
var isometric_viewport: Viewport
var debug_view: DebugCameraView
var are_debug_views_enabled := false

const DEFAULT_CAMERA_NAME = "camera"
const DEBUG_VIEWPORTS = [
	"rgb_viewport", "depth_viewport", "isometric_viewport", "top_down_viewport"
]
const DEBUG_CAMERAS_SCENE_PATH = "res://scenes/debug.tscn"


func _init(_root: Node, _resolution: Vector2, _tracked_node: String, _are_debug_views_enabled: bool):
	root = _root
	resolution = _resolution
	tracked_node = _tracked_node
	are_debug_views_enabled = _are_debug_views_enabled

	setup()


func setup() -> void:
	tracked = root.find_node(tracked_node, true, false)
	HARD.assert(tracked != null, "Could not find node %s" % tracked_node)
	HARD.assert(tracked is Spatial, "Track node %s must be a Spatial node" % tracked_node)

	cameras = []

	if are_debug_views_enabled:
		# delete any existing camera as we won't be using them
		root.find_node(DEFAULT_CAMERA_NAME, true, false).queue_free()

		var debug_scene = load(DEBUG_CAMERAS_SCENE_PATH).instance()
		root.add_child(debug_scene)

		for viewport_name in DEBUG_VIEWPORTS:
			var _viewport = root.find_node(viewport_name, true, false)
			var _camera = _viewport.get_child(0)
			if is_instance_valid(_camera) and _camera.is_enabled:
				_viewport.size = resolution
				cameras.append(_camera)
			else:
				_viewport.render_target_update_mode = Viewport.UPDATE_DISABLED
				_viewport.size = Vector2.ZERO

		rgb_viewport = root.find_node("rgb_viewport", true, false)
		top_down_viewport = root.find_node("top_down_viewport", true, false)
		isometric_viewport = root.find_node("isometric_viewport", true, false)
	else:
		var node = root.find_node(DEFAULT_CAMERA_NAME, true, false)
		if is_instance_valid(node):
			cameras = [node]


func do_physics(_delta: float) -> void:
	HARD.assert(tracked != null, "Must set tracked node before calling `do_physics`")

	for camera in cameras:
		if not is_instance_valid(camera) or not camera.is_enabled:
			continue
		var _tracked = camera.tracked if camera.tracked != null else tracked
		if camera.is_tracking_relative_to_initial_transform:
			camera.global_transform.origin = (
				_tracked.global_transform.origin
				+ camera.initial_transform.origin
			)
			camera.global_transform.basis = camera.initial_transform.basis
		else:
			camera.global_transform = _tracked.global_transform

		if camera.is_automatically_looking_at_tracked:
			camera.look_at(_tracked.transform.origin, Vector3.UP)


func get_primary_camera() -> TrackingCamera:
	return cameras[0]


func add_debug_camera() -> void:
	debug_view = DebugCameraView.new()
	debug_view.insert_as_overlay(self, root, resolution)


func remove_debug_camera() -> void:
	debug_view.remove_from(self)


func get_rgbd_data() -> Image:
	if debug_view != null:
		return debug_view.get_rgbx_data()
	if are_debug_views_enabled:
		return rgb_viewport.get_texture().get_data()
	else:
		return root.get_texture().get_data()


func get_top_down_rgbd_data() -> Image:
	HARD.assert(
		top_down_viewport != null and are_debug_views_enabled,
		"Trying to get top down rgbd but viewport is not defined"
	)
	return top_down_viewport.get_texture().get_data()


func get_isometric_rgbd_data() -> Image:
	HARD.assert(
		isometric_viewport != null and are_debug_views_enabled,
		"Trying to get isometric rgbd but viewport is not defined"
	)
	return isometric_viewport.get_texture().get_data()
