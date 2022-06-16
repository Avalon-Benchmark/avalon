extends Node

class_name NewTrackingCameraController

export(String) var tracked_node
var node_paths = ["camera", "rgb_camera", "depth_camera", "top_down_camera", "isometric_camera"]

var tracked: Spatial
var cameras = []


func init():
	var root = get_tree().root

	# TODO: this should get more information from SimLoop, I think?
	for node_path in node_paths:
		var node = root.find_node(node_path, true, false)
		if is_instance_valid(node):
			cameras.append(node)

	tracked = get_tree().root.find_node(tracked_node, true, false)
	HARD.assert(tracked != null, "Could not find node %s" % tracked_node)
	HARD.assert(tracked is Spatial, "Track node %s must be a Spatial node" % tracked_node)


func do_physics(_delta: float) -> void:
	HARD.assert(tracked != null, "Must set tracked node before calling `do_physics`")

	for camera in cameras:
		if camera.is_tracking_relative_to_initial_transform:
			camera.global_transform.origin = (
				tracked.global_transform.origin
				+ camera.initial_transform.origin
			)
			camera.global_transform.basis = camera.initial_transform.basis
		else:
			camera.global_transform = tracked.global_transform
