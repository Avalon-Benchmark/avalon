extends HumanPlayerSpec

class_name VRHumanPlayerSpec


func create_player() -> Object:
	return load("res://game/player/scenes/vr_human_player.tscn").instance()
