extends MouseKeyboardInputCollector

class_name MouseKeyboardAgentInputCollector

var head_delta_position_x := 0.0
var head_delta_position_z := 0.0
var head_delta_rotation_x := 0.0
var head_delta_rotation_y := 0.0
var is_left_hand_grasping := false
var is_right_hand_grasping := false
var is_left_hand_throwing := false
var is_right_hand_throwing := false
var is_jumping := false
var is_eating := false
var is_crouching := false


func reset():
	head_delta_position_x = 0.0
	head_delta_position_z = 0.0
	head_delta_rotation_x = 0.0
	head_delta_rotation_y = 0.0
	is_left_hand_grasping = false
	is_right_hand_grasping = false
	is_left_hand_throwing = false
	is_right_hand_throwing = false
	is_jumping = false
	is_eating = false
	is_crouching = false


func to_normalized_relative_action(player):
	var action = MouseKeyboardAction.new()
	action.head_delta_position = Tools.vec3_sphere_clamp(
		Vector3(head_delta_position_x, 0.0, head_delta_position_z), 1.0
	)
	action.head_delta_rotation.x = (
		Tools.clamp_delta_rotation(
			player.target_head.rotation.x,
			deg2rad(player.max_head_angular_speed.x) * head_delta_rotation_x,
			MAX_PITCH_ROTATION
		)
		/ deg2rad(player.max_head_angular_speed.x)
	)
	action.head_delta_rotation.y = head_delta_rotation_y
	printt(action.head_delta_rotation, deg2rad(player.max_head_angular_speed.x))
	action.is_left_hand_grasping = is_left_hand_grasping
	action.is_right_hand_grasping = is_right_hand_grasping
	action.is_left_hand_throwing = is_left_hand_throwing
	action.is_right_hand_throwing = is_right_hand_throwing
	action.is_jumping = is_jumping
	action.is_eating = is_eating
	action.is_crouching = is_crouching

	return action


func read_input_from_pipe(action_pipe: StreamPeerBuffer) -> void:
	head_delta_position_x = action_pipe.get_float()
	head_delta_position_z = action_pipe.get_float()
	head_delta_rotation_x = action_pipe.get_float()
	head_delta_rotation_y = action_pipe.get_float()
	is_left_hand_grasping = action_pipe.get_float() == 1.0
	is_right_hand_grasping = action_pipe.get_float() == 1.0
	is_left_hand_throwing = action_pipe.get_float() == 1.0
	is_right_hand_throwing = action_pipe.get_float() == 1.0
	is_jumping = action_pipe.get_float() == 1.0
	is_eating = action_pipe.get_float() == 1.0
	is_crouching = action_pipe.get_float() == 1.0
