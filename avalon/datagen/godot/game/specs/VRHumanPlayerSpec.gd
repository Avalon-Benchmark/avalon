extends HumanPlayerSpec

class_name VRHumanPlayerSpec


func get_scene_instance() -> Object:
	return load("res://game/player/scenes/vr_human_player.tscn").instance()


func get_input_collector():
	return VRHumanInputCollector.new()
