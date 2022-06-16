extends AgentPlayerSpec

class_name VRAgentPlayerSpec


func create_player() -> Object:
	return load("res://game/player/scenes/vr_agent_player.tscn").instance()
