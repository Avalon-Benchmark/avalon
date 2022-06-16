extends BehaviorCriteria

class_name NoticedByPlayer

var switch_after_steps: int
var switch_step := 0

var visibility_notifier: VisibilityNotifier
var is_visible_to_player := false
var is_noticed := false


func _init(_visiblity_notifier: VisibilityNotifier, _switch_after_steps: int):
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


func is_matched_by(_animal) -> bool:
	if is_visible_to_player == is_noticed:
		switch_step = 0
		return is_noticed

	switch_step = (switch_step + 1) % switch_after_steps
	var is_ready_to_switch = switch_step == 0
	if is_ready_to_switch:
		is_noticed = is_visible_to_player

	return is_noticed


func _on_camera_entered(camera: Camera):
	if camera.get_parent().name == "player":
		is_visible_to_player = true


func _on_camera_exited(camera: Camera):
	if camera.get_parent().name == "player":
		is_visible_to_player = false


func reset():
	switch_step = 0
