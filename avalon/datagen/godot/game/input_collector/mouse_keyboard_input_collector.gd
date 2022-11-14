extends PlayerInputCollector

class_name MouseKeyboardInputCollector

const MAX_PITCH_ROTATION = deg2rad(80)


func scaled_relative_action_from_normalized_relative_action(normalized_action, player):
	var action = MouseKeyboardAction.new()

	action.is_left_hand_throwing = normalized_action.is_left_hand_throwing
	action.is_right_hand_throwing = normalized_action.is_right_hand_throwing
	action.is_left_hand_grasping = normalized_action.is_left_hand_grasping
	action.is_right_hand_grasping = normalized_action.is_right_hand_grasping
	action.is_jumping = normalized_action.is_jumping
	action.is_eating = normalized_action.is_eating
	action.is_crouching = normalized_action.is_crouching

	action.head_delta_position = (
		normalized_action.head_delta_position
		* player.max_head_linear_speed
	)
	action.head_delta_rotation = (
		normalized_action.head_delta_rotation
		* Tools.vec_deg2rad(player.max_head_angular_speed)
	)
	action.left_hand_delta_position = (
		normalized_action.left_hand_delta_position
		* player.max_hand_linear_speed
	)
	action.left_hand_delta_rotation = (
		normalized_action.left_hand_delta_rotation
		* deg2rad(player.max_hand_angular_speed)
	)
	action.right_hand_delta_position = (
		normalized_action.right_hand_delta_position
		* player.max_hand_linear_speed
	)
	action.right_hand_delta_rotation = (
		normalized_action.right_hand_delta_rotation
		* deg2rad(player.max_hand_angular_speed)
	)

	return action
