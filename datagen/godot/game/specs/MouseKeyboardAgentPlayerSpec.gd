extends AgentPlayerSpec

class_name MouseKeyboardAgentPlayerSpec


func create_player() -> Object:
	return load("res://game/player/scenes/mouse_keyboard_agent_player.tscn").instance()
