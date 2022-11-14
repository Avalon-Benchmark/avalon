extends AgentPlayerSpec

class_name VRAgentPlayerSpec


func get_scene_instance() -> Object:
	return load("res://game/player/scenes/vr_agent_player.tscn").instance()


func get_input_collector():
	return VRAgentInputCollector.new()
