extends FloatingGUI

class_name RestartFloatingGUI

var is_restart_pressed := false
var is_close_pressed := false
var is_open_pressed := false
var is_restarting := false

onready var restart_label = get_node("restart")


func read_input_from_event(event: InputEvent) -> void:
	if event is InputEventJoypadButton:
		if (
			event.device == VRHumanInputCollector.CONTROLLER_LEFT
			and event.button_index == VRHumanInputCollector.BUTTON_A
		):
			is_open_pressed = true

		if (
			event.device == VRHumanInputCollector.CONTROLLER_LEFT
			and event.button_index == VRHumanInputCollector.BUTTON_B
		):
			is_restart_pressed = true

		if (
			event.device == VRHumanInputCollector.CONTROLLER_RIGHT
			and event.button_index == VRHumanInputCollector.BUTTON_B
		):
			is_close_pressed = true

	# just reset without the menu when using mouse and keyboard
	if Globals.is_player_using_mouse_keyboard_controls() and event.is_action_pressed("reset"):
		is_opened = true
		is_restart_pressed = true


func set_message(text: String) -> void:
	restart_label.text = text


func _input(event: InputEvent):
	read_input_from_event(event)


func _physics_process(_delta):
	if not is_opened and is_open_pressed:
		open()

	if is_opened and is_restart_pressed:
		is_restarting = true
		close()

	if is_opened and is_close_pressed:
		close()

	is_open_pressed = false
	is_restart_pressed = false
	is_close_pressed = false
