extends Node

var _player = null


func get_player() -> Object:
	var is_player_ref_valid = (
		_player != null
		and is_instance_valid(_player)
		and not _player.is_queued_for_deletion()
	)
	if not is_player_ref_valid:
		_player = get_tree().root.find_node("player", true, false)
	return _player


func is_player_using_mouse_keyboard_controls() -> bool:
	return get_player() is MouseKeyboardPlayer
