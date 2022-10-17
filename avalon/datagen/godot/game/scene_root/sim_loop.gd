extends SceneTree
class_name SimLoop

var avalon_spec: SimSpec
var game_manager: GameManager
var _is_quitting := false
var _has_done_physics := false

# Godot main loop order:
# 1:   Update timers.
# 2:   For each step:
# 2.1: PhysicsServer.flush_queries()
# 2.2: MainLoop.iteration()
# 2.3: PhysicsServer.step() -> [...]._physics_process()
# 3:   MainLoop.idle()
# 4:   VisualServer.sync()
# 5:   VisualServer.draw()
# 6:   MainLoop._post_draw()


func _initialize() -> void:
	var results = load_avalon_spec_from_args()
	avalon_spec = results[0]
	var input_pipe_path = results[1]
	var output_pipe_path = results[2]

	HARD.print_debug_info()

	if avalon_spec.player is HumanPlayerSpec and not avalon_spec.player.is_human_playback_enabled:
		game_manager = HumanGameManager.new(root, avalon_spec)
	elif avalon_spec.player is AgentPlayerSpec or avalon_spec.player.is_human_playback_enabled:
		game_manager = AgentGameManager.new(root, avalon_spec, input_pipe_path, output_pipe_path)

	print(CONST.READY_LOG_SIGNAL)
	print()


func _input_event(event: InputEvent) -> void:
	if _is_quitting:
		return
	if avalon_spec.player is HumanPlayerSpec and not avalon_spec.player.is_human_playback_enabled:
		game_manager.read_input_from_event(event)


func _iteration(delta: float) -> bool:
	if _is_quitting:
		return true

	if Input.is_action_pressed("ui_cancel"):
		handle_quit()
		return true

	_has_done_physics = true
	game_manager.do_physics(delta)

	return false


func _idle(delta: float) -> bool:
	VisualServer.request_frame_drawn_callback(self, "_post_draw", null)
	if OS.has_feature("zero_delay_physics") and _has_done_physics:
		PhysicsServer.flush_queries()

	if _is_quitting:
		return true

	game_manager.idle(delta)

	_has_done_physics = false

	return false


func _post_draw(_arg) -> void:
	if _is_quitting:
		return

	var is_quitting = game_manager.read_input_from_pipe()
	if is_quitting:
		handle_quit()


func handle_quit():
	_is_quitting = true
	ClassBuilder.clear()
	paused = true
	game_manager.quit()
	var possibly_dangling_references = [game_manager, avalon_spec]
	for ref in possibly_dangling_references:
		if ref:
			ref.unreference()
			ref = null
	quit()


func load_avalon_spec_from_args():
	var args := Array(OS.get_cmdline_args())
	var config_paths = []
	if HARD.mode():
		config_paths.append("./config.json")
	elif OS.get_name() == "Android":
		config_paths.append("res://android/config.json")
	var input_pipe_path := ""
	var output_pipe_path := ""
	for arg in args:
		var split_arg := Array(arg.split("=", true, 1))
		match split_arg[0]:
			"--dev":
				pass
			"--config-file":
				var path = split_arg[1]
				config_paths.append(ProjectSettings.globalize_path(path))
			"--input-pipe-path":
				input_pipe_path = split_arg[1]
			"--output-pipe-path":
				output_pipe_path = split_arg[1]
			_:
				HARD.stop("Unknown command line arg: '%s'", [arg])
	HARD.assert(HARD.mode() or (len(config_paths) == 1), "No config file given")

	var json_dict := Tools.read_json_from_path(config_paths[-1])
	var json_spec = ClassBuilder.get_object_from_json(json_dict, "/")
	return [json_spec, input_pipe_path, output_pipe_path]
