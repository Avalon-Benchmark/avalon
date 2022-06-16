extends HumanPlayerSpec

class_name MouseKeyboardHumanPlayerSpec


func create_player() -> Object:
	return load("res://game/player/scenes/mouse_keyboard_human_player.tscn").instance()
