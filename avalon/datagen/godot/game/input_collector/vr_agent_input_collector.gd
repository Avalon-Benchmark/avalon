extends VRInputCollector

class_name VRAgentInputCollector

var head_delta_position_x: float
var head_delta_position_y: float
var head_delta_position_z: float
var head_delta_rotation_x: float
var head_delta_rotation_y: float
var head_delta_rotation_z: float
var left_hand_delta_position_x: float
var left_hand_delta_position_y: float
var left_hand_delta_position_z: float
var left_hand_delta_rotation_x: float
var left_hand_delta_rotation_y: float
var left_hand_delta_rotation_z: float
var right_hand_delta_position_x: float
var right_hand_delta_position_y: float
var right_hand_delta_position_z: float
var right_hand_delta_rotation_x: float
var right_hand_delta_rotation_y: float
var right_hand_delta_rotation_z: float
var is_left_hand_grasping: bool
var is_right_hand_grasping: bool
var is_jumping: bool


func reset():
	head_delta_position_x = 0.0
	head_delta_position_y = 0.0
	head_delta_position_z = 0.0
	head_delta_rotation_x = 0.0
	head_delta_rotation_y = 0.0
	head_delta_rotation_z = 0.0
	left_hand_delta_position_x = 0.0
	left_hand_delta_position_y = 0.0
	left_hand_delta_position_z = 0.0
	left_hand_delta_rotation_x = 0.0
	left_hand_delta_rotation_y = 0.0
	left_hand_delta_rotation_z = 0.0
	right_hand_delta_position_x = 0.0
	right_hand_delta_position_y = 0.0
	right_hand_delta_position_z = 0.0
	right_hand_delta_rotation_x = 0.0
	right_hand_delta_rotation_y = 0.0
	right_hand_delta_rotation_z = 0.0
	is_left_hand_grasping = false
	is_right_hand_grasping = false
	is_jumping = false


func to_normalized_relative_action(_player):
	var action = VRAction.new()
	action.head_delta_position = Vector3(
		head_delta_position_x, head_delta_position_y, head_delta_position_z
	)
	action.head_delta_rotation.x = head_delta_rotation_x
	action.head_delta_rotation.y = head_delta_rotation_y
	action.head_delta_rotation.z = head_delta_rotation_z
	action.left_hand_delta_position = Vector3(
		left_hand_delta_position_x, left_hand_delta_position_y, left_hand_delta_position_z
	)
	action.left_hand_delta_rotation.x = left_hand_delta_rotation_x
	action.left_hand_delta_rotation.y = left_hand_delta_rotation_y
	action.left_hand_delta_rotation.z = left_hand_delta_rotation_z
	action.is_left_hand_grasping = is_left_hand_grasping
	action.right_hand_delta_position = Vector3(
		right_hand_delta_position_x, right_hand_delta_position_y, right_hand_delta_position_z
	)
	action.right_hand_delta_rotation.x = right_hand_delta_rotation_x
	action.right_hand_delta_rotation.y = right_hand_delta_rotation_y
	action.right_hand_delta_rotation.z = right_hand_delta_rotation_z
	action.is_right_hand_grasping = is_right_hand_grasping
	action.is_jumping = is_jumping
	return action


func read_input_from_pipe(action_pipe: StreamPeerBuffer) -> void:
	head_delta_position_x = action_pipe.get_float()
	head_delta_position_y = action_pipe.get_float()
	head_delta_position_z = action_pipe.get_float()
	head_delta_rotation_x = action_pipe.get_float()
	head_delta_rotation_y = action_pipe.get_float()
	head_delta_rotation_z = action_pipe.get_float()
	left_hand_delta_position_x = action_pipe.get_float()
	left_hand_delta_position_y = action_pipe.get_float()
	left_hand_delta_position_z = action_pipe.get_float()
	left_hand_delta_rotation_x = action_pipe.get_float()
	left_hand_delta_rotation_y = action_pipe.get_float()
	left_hand_delta_rotation_z = action_pipe.get_float()
	right_hand_delta_position_x = action_pipe.get_float()
	right_hand_delta_position_y = action_pipe.get_float()
	right_hand_delta_position_z = action_pipe.get_float()
	right_hand_delta_rotation_x = action_pipe.get_float()
	right_hand_delta_rotation_y = action_pipe.get_float()
	right_hand_delta_rotation_z = action_pipe.get_float()
	is_left_hand_grasping = action_pipe.get_float() == 1.0
	is_right_hand_grasping = action_pipe.get_float() == 1.0
	is_jumping = action_pipe.get_float() == 1.0


func read_input_from_data(action: PoolRealArray) -> void:
	head_delta_position_x = action[0]
	head_delta_position_y = action[1]
	head_delta_position_z = action[2]
	head_delta_rotation_x = action[3]
	head_delta_rotation_y = action[4]
	head_delta_rotation_z = action[5]
