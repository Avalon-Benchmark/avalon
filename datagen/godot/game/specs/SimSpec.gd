extends SpecBase

class_name SimSpec

# TODO these aren't related to RL envs, we'll leave them for now
var dataset_id: int
var label: int

var video_min: int
var video_max: int
var frame_max: int

var random_int: int
var random_key: String

var dir_root: String
var output_file_name_format: String
var scene_path: String

var is_using_shared_caches: bool

# TODO this isn't related to RL envs, we'll leave them for now
var is_generating_paired_videos: bool

var recording_options: RecordingOptionsSpec
var player: PlayerSpec  #optional


# TODO: clean this up and move it somewhere sane
func read_and_format_config_key(
	format_str: String, video := 0, feature := "ERROR", type_and_dim := "OH_NO", check := true
) -> String:
	var result = format_str.format(
		{
			"dir_root": dir_root,
			"video_id": "%06d" % video,
			"feature": feature,
			"type_and_dim": type_and_dim,
		}
	)
	if check:
		for test in ["{", "}", "%"]:
			HARD.assert(result.find(test) == -1, 'cannot format path: "%s"', format_str)
	return result


func build(_is_registered: bool = true) -> Object:
	var sim = ClassBuilder.get_class_resource(self).new(self)
	sim.name = "sim"
	return sim
