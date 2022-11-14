extends InputCollector

class_name PlayerInputCollector


func to_normalized_relative_action(_player):
	return HARD.assert(false, "Not implemented")


func scaled_relative_action_from_normalized_relative_action(_normalized_action, _player):
	return HARD.assert(false, "Not implemented")


func get_action(player) -> PlayerAction:
	var action = PlayerAction.new()
	action.normalized = to_normalized_relative_action(player)
	action.scaled = scaled_relative_action_from_normalized_relative_action(
		action.normalized, player
	)
	return action
