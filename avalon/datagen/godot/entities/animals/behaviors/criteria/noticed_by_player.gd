extends BehaviorCriteria

class_name NoticedByPlayer

var switch_after_steps: int
var switch_step := 0

var visibility_notifier: VisibilityNotifier
var is_visible_to_player := false
var is_noticed := false

var _player_eyes: Camera


func _init(_visiblity_notifier: VisibilityNotifier, _switch_after_steps: int):
	_player_eyes = Globals.get_player().eyes
	visibility_notifier = _visiblity_notifier
	HARD.assert(
		OK == visibility_notifier.connect("camera_entered", self, "_on_camera_entered"),
		"Failed to connect signal"
	)
	HARD.assert(
		OK == visibility_notifier.connect("camera_exited", self, "_on_camera_exited"),
		"Failed to connect signal"
	)
	switch_after_steps = _switch_after_steps


func is_matched_by(animal) -> bool:
	if is_visible_to_player == is_noticed:
		if HARD.mode() and switch_step != 0:
			var visibility = "re-entered camera" if is_visible_to_player else "re-exited camera"
			var steps = "step %s/%s" % [switch_step, switch_after_steps]
			print("%s %s before notice switched: %s" % [animal, visibility, steps])
		switch_step = 0
		return is_noticed

	switch_step = (switch_step + 1) % switch_after_steps
	var is_ready_to_switch = switch_step == 0
	if is_ready_to_switch:
		is_noticed = is_visible_to_player

	if HARD.mode() and switch_step == 1:
		if is_visible_to_player:
			print("%s now visible, will be noticed in %s steps" % [animal, switch_after_steps])
		else:
			print("%s no longer visible, deactivation in %s steps" % [animal, switch_after_steps])

	return is_noticed


func _on_camera_entered(camera: Camera):
	if camera == _player_eyes:
		is_visible_to_player = true


func _on_camera_exited(camera: Camera):
	if camera == _player_eyes:
		is_visible_to_player = false


func reset():
	switch_step = 0
