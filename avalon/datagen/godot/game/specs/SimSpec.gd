extends SpecBase

class_name SimSpec

var episode_seed: int  #optional
var dir_root: String
var recording_options: RecordingOptionsSpec
var player: PlayerSpec


func get_controlled_node_specs() -> Array:
	return [player]


func get_resolution() -> Vector2:
	return Vector2(recording_options.resolution_x, recording_options.resolution_y)
