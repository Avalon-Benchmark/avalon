extends HumanPlayerSpec

class_name MouseKeyboardHumanPlayerSpec


func get_scene_instance() -> Object:
	return load("res://game/player/scenes/mouse_keyboard_human_player.tscn").instance()


func get_input_collector():
	return MouseKeyboardHumanInputCollector.new()
