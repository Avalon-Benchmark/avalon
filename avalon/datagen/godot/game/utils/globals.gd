extends Node

var _player = null


func get_player() -> Object:
	if _player == null:
		_player = get_tree().root.find_node("player", true, false)
	return _player


func is_player_using_mouse_keyboard_controls() -> bool:
	return get_player() is MouseKeyboardPlayer
