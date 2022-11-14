extends AgentPlayerSpec

class_name MouseKeyboardAgentPlayerSpec


func get_scene_instance() -> Object:
	return load("res://game/player/scenes/mouse_keyboard_agent_player.tscn").instance()


func get_input_collector():
	return MouseKeyboardAgentInputCollector.new()
