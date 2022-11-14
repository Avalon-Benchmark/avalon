extends PlayerInputCollector

class_name VRInputCollector


func scaled_relative_action_from_normalized_relative_action(normalized_action: VRAction, player):
	var scaled_action = VRAction.new()

	scaled_action.is_left_hand_grasping = normalized_action.is_left_hand_grasping
	scaled_action.is_right_hand_grasping = normalized_action.is_right_hand_grasping
	scaled_action.is_jumping = normalized_action.is_jumping

	scaled_action.head_delta_position = (
		normalized_action.head_delta_position
		* player.max_head_linear_speed
	)
	scaled_action.head_delta_rotation = (
		normalized_action.head_delta_rotation
		* Tools.vec_deg2rad(player.max_head_angular_speed)
	)
	scaled_action.left_hand_delta_position = (
		normalized_action.left_hand_delta_position
		* player.max_hand_linear_speed
	)
	scaled_action.left_hand_delta_rotation = (
		normalized_action.left_hand_delta_rotation
		* deg2rad(player.max_hand_angular_speed)
	)
	scaled_action.right_hand_delta_position = (
		normalized_action.right_hand_delta_position
		* player.max_hand_linear_speed
	)
	scaled_action.right_hand_delta_rotation = (
		normalized_action.right_hand_delta_rotation
		* deg2rad(player.max_hand_angular_speed)
	)
	scaled_action.relative_origin_delta_rotation = (
		normalized_action.relative_origin_delta_rotation
		* Tools.vec_deg2rad(player.max_head_angular_speed)
	)

	return scaled_action
