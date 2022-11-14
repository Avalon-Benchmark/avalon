extends VRInputCollector

class_name VRHumanInputCollector

const AXIS_STICK_X = 0
const AXIS_STICK_Y = 1
const AXIS_TRIGGER = 2
const AXIS_SQUEEZE = 4

const BUTTON_STICK = 14
const BUTTON_TRIGGER = 15
const BUTTON_TRIGGER_ALT = 2
const BUTTON_Y = 1
const BUTTON_X = 7
const BUTTON_A = 7
const BUTTON_B = 1

const CONTROLLER_LEFT = 0
const CONTROLLER_RIGHT = 1

var arvr_origin_transform := Transform.IDENTITY
var arvr_camera_transform := Transform.IDENTITY
var arvr_left_hand_transform := Transform.IDENTITY
var arvr_right_hand_transform := Transform.IDENTITY

var human_height := 0.0

var look_stick := Vector2.ZERO
var strafe_stick := Vector2.ZERO
var is_jump_pressed := false
var is_left_hand_grab_pressed := false
var is_right_hand_grab_pressed := false

# note: adjusts how much the stick will move relative to turning our head
var look_sensitivity := 0.85


func reset():
	arvr_origin_transform = Transform.IDENTITY
	arvr_camera_transform = Transform.IDENTITY
	arvr_left_hand_transform = Transform.IDENTITY
	arvr_right_hand_transform = Transform.IDENTITY

	look_stick = Vector2.ZERO
	strafe_stick = Vector2.ZERO
	is_jump_pressed = false


func to_normalized_relative_action(player):
	var action := VRAction.new()

	if player.is_human_playback_enabled:
		player.configure_nodes_for_playback(
			arvr_camera_transform,
			arvr_left_hand_transform,
			arvr_right_hand_transform,
			arvr_origin_transform,
			human_height if human_height != 0.0 else 2.0,
			false
		)

	action.is_jumping = is_jump_pressed
	action.is_left_hand_grasping = is_left_hand_grab_pressed
	action.is_right_hand_grasping = is_right_hand_grab_pressed

	# stick position has to move the ARVR origin
	var relative_origin_delta_position = Vector3.ZERO
	var relative_origin_delta_rotation = Vector3.ZERO
	if not player.is_climbing():
		relative_origin_delta_position = Tools.vec3_sphere_clamp(
			Vector3(strafe_stick.x, 0.0, -strafe_stick.y), 1.0
		)
		relative_origin_delta_rotation = Vector3(0.0, -look_stick.x, 0.0) * look_sensitivity

	if player.is_unable_to_track(player.arvr_camera):
		print("cannot track arvr camera")
		# do nothing if we can't track the arvr camera
		return action
	var head_delta_quat = Tools.get_delta_quaternion(
		player.prev_arvr_camera_global_transform.basis.get_rotation_quat(),
		player.arvr_camera.global_transform.basis.get_rotation_quat()
	)

	action.head_delta_rotation = (
		Tools.vec3_range_lerp(
			head_delta_quat.get_euler(),
			-Tools.vec_deg2rad(player.max_head_angular_speed),
			Tools.vec_deg2rad(player.max_head_angular_speed),
			-Vector3.ONE,
			Vector3.ONE
		)
		+ relative_origin_delta_rotation
	)
	action.relative_origin_delta_rotation = relative_origin_delta_rotation

	var global_head_delta_position = (
		player.arvr_camera.global_transform.origin
		- player.prev_arvr_camera_global_transform.origin
	)

	var relative_head_delta_position = Tools.get_move_delta_position(
		player.arvr_camera.global_transform.basis.inverse(), global_head_delta_position
	)

	# must add back in head y position because the above function removes it
	relative_head_delta_position.y = (
		global_head_delta_position.y
		* player.height
		/ player.human_height
	)

	action.head_delta_position = (
		Tools.normalize(
			relative_head_delta_position,
			-player.max_head_linear_speed,
			player.max_head_linear_speed
		)
		+ relative_origin_delta_position
	)

	if not player.is_unable_to_track(player.arvr_left_hand):
		var relative_left_hand_delta_position: Vector3
		if player.target_left_hand.is_grasping_heavy_thing():
			relative_left_hand_delta_position = _get_relative_hand_delta_position_while_climbing(
				player.prev_arvr_left_hand_global_transform.origin,
				player.arvr_left_hand.global_transform.origin,
				player.prev_arvr_camera_global_transform,
				player.arvr_camera.global_transform
			)
		else:
			relative_left_hand_delta_position = _get_relative_hand_delta_position(
				player.prev_arvr_left_hand_global_transform.origin,
				player.arvr_left_hand.global_transform.origin,
				player.prev_arvr_camera_global_transform,
				player.arvr_camera.global_transform
			)

		action.left_hand_delta_position = Tools.normalize(
			relative_left_hand_delta_position,
			-player.max_hand_linear_speed,
			player.max_hand_linear_speed
		)

		var left_hand_delta_rotation := _get_hand_rotation(
			player.prev_arvr_left_hand_global_transform,
			player.arvr_left_hand.global_transform,
			head_delta_quat
		)

		action.left_hand_delta_rotation = Tools.normalize(
			left_hand_delta_rotation,
			-deg2rad(player.max_hand_angular_speed),
			deg2rad(player.max_hand_angular_speed)
		)

	if not player.is_unable_to_track(player.arvr_right_hand):
		var relative_right_hand_delta_position: Vector3
		if player.target_right_hand.is_grasping_heavy_thing():
			relative_right_hand_delta_position = _get_relative_hand_delta_position_while_climbing(
				player.prev_arvr_right_hand_global_transform.origin,
				player.arvr_right_hand.global_transform.origin,
				player.prev_arvr_camera_global_transform,
				player.arvr_camera.global_transform
			)
		else:
			relative_right_hand_delta_position = _get_relative_hand_delta_position(
				player.prev_arvr_right_hand_global_transform.origin,
				player.arvr_right_hand.global_transform.origin,
				player.prev_arvr_camera_global_transform,
				player.arvr_camera.global_transform
			)

		action.right_hand_delta_position = Tools.normalize(
			relative_right_hand_delta_position,
			-player.max_hand_linear_speed,
			player.max_hand_linear_speed
		)

		var right_hand_delta_rotation := _get_hand_rotation(
			player.prev_arvr_right_hand_global_transform,
			player.arvr_right_hand.global_transform,
			head_delta_quat
		)

		action.right_hand_delta_rotation = Tools.normalize(
			right_hand_delta_rotation,
			-deg2rad(player.max_hand_angular_speed),
			deg2rad(player.max_hand_angular_speed)
		)

	return action


func read_input_from_event(event: InputEvent) -> void:
	if event is InputEventJoypadButton:
		if event.device == CONTROLLER_RIGHT and event.button_index == BUTTON_A:
			is_jump_pressed = true

		var is_trigger_pressed = (
			event.button_index == BUTTON_TRIGGER
			or event.button_index == BUTTON_TRIGGER_ALT
		)
		if event.device == CONTROLLER_RIGHT and is_trigger_pressed:
			is_right_hand_grab_pressed = event.pressed

		if event.device == CONTROLLER_LEFT and is_trigger_pressed:
			is_left_hand_grab_pressed = event.pressed


func read_input_before_physics() -> void:
	look_stick.x = Input.get_joy_axis(CONTROLLER_RIGHT, AXIS_STICK_X)
	look_stick.y = Input.get_joy_axis(CONTROLLER_RIGHT, AXIS_STICK_Y)

	strafe_stick.x = Input.get_joy_axis(CONTROLLER_LEFT, AXIS_STICK_X)
	strafe_stick.y = Input.get_joy_axis(CONTROLLER_LEFT, AXIS_STICK_Y)


func _get_transform_from_pipe(action_pipe: StreamPeerBuffer) -> Transform:
	var origin = Vector3(
		action_pipe.get_double(), action_pipe.get_double(), action_pipe.get_double()
	)
	return Transform(
		Vector3(action_pipe.get_double(), action_pipe.get_double(), action_pipe.get_double()),
		Vector3(action_pipe.get_double(), action_pipe.get_double(), action_pipe.get_double()),
		Vector3(action_pipe.get_double(), action_pipe.get_double(), action_pipe.get_double()),
		origin
	)


func read_input_from_pipe(action_pipe: StreamPeerBuffer) -> void:
	human_height = action_pipe.get_double()
	arvr_origin_transform = _get_transform_from_pipe(action_pipe)
	arvr_camera_transform = _get_transform_from_pipe(action_pipe)
	arvr_left_hand_transform = _get_transform_from_pipe(action_pipe)
	arvr_right_hand_transform = _get_transform_from_pipe(action_pipe)

	look_stick = Vector2(action_pipe.get_double(), action_pipe.get_double())
	strafe_stick = Vector2(action_pipe.get_double(), action_pipe.get_double())

	is_left_hand_grab_pressed = action_pipe.get_double() == 1.0
	is_right_hand_grab_pressed = action_pipe.get_double() == 1.0
	is_jump_pressed = action_pipe.get_double() == 1.0


func _get_relative_hand_delta_position(
	prev_arvr_hand_position: Vector3,
	curr_arvr_hand_position: Vector3,
	prev_arvr_camera_transform: Transform,
	curr_arvr_camera_transform: Transform
) -> Vector3:
	var hand_movement_from_head_movement = (
		# VR humans don't have a pre-defined arm length
		Tools.get_new_hand_position(
			prev_arvr_hand_position,
			Vector3.ZERO,
			prev_arvr_camera_transform,
			curr_arvr_camera_transform,
			INF
		)
		- prev_arvr_hand_position
	)
	var global_hand_delta_position = curr_arvr_hand_position - prev_arvr_hand_position
	var global_hand_delta_position_without_head_movement = (
		global_hand_delta_position
		- hand_movement_from_head_movement
	)
	return (
		curr_arvr_camera_transform.basis.inverse()
		* global_hand_delta_position_without_head_movement
	)


func _get_relative_hand_delta_position_while_climbing(
	prev_arvr_hand_position: Vector3,
	curr_arvr_hand_position: Vector3,
	prev_arvr_camera_transform: Transform,
	curr_arvr_camera_transform: Transform
) -> Vector3:
	var hand_movement_from_head_movement = (
		# VR humans don't have a pre-defined arm length
		Tools.get_new_hand_position_while_climbing(
			prev_arvr_hand_position,
			Vector3.ZERO,
			prev_arvr_camera_transform,
			curr_arvr_camera_transform,
			INF
		)
		- prev_arvr_hand_position
	)
	var global_hand_delta_position = curr_arvr_hand_position - prev_arvr_hand_position
	var global_hand_delta_position_without_head_movement = (
		global_hand_delta_position
		- hand_movement_from_head_movement
	)
	return (
		curr_arvr_camera_transform.basis.inverse()
		* global_hand_delta_position_without_head_movement
	)


func _get_hand_rotation(
	prev_arvr_hand_transform: Transform, curr_arvr_hand_transform: Transform, head_delta_quat: Quat
) -> Vector3:
	var hand_delta_quat = Tools.get_delta_quaternion(
		prev_arvr_hand_transform, curr_arvr_hand_transform
	)
	return (hand_delta_quat * head_delta_quat.inverse()).get_euler()


func _put_transform_in_stream(stream: StreamPeerBuffer, transform: Transform):
	stream.put_double(transform.origin.x)
	stream.put_double(transform.origin.y)
	stream.put_double(transform.origin.z)
	stream.put_double(transform.basis.x.x)
	stream.put_double(transform.basis.x.y)
	stream.put_double(transform.basis.x.z)
	stream.put_double(transform.basis.y.x)
	stream.put_double(transform.basis.y.y)
	stream.put_double(transform.basis.y.z)
	stream.put_double(transform.basis.z.x)
	stream.put_double(transform.basis.z.y)
	stream.put_double(transform.basis.z.z)


func write_into_stream(stream: StreamPeerBuffer, player) -> void:
	stream.put_double(player.human_height)
	# save off the current ARVR tracked nodes
	_put_transform_in_stream(stream, player.arvr_origin.transform)
	_put_transform_in_stream(stream, player.arvr_camera.transform)
	_put_transform_in_stream(stream, player.arvr_left_hand.get_parent().transform)
	_put_transform_in_stream(stream, player.arvr_right_hand.get_parent().transform)

	stream.put_double(look_stick.x)
	stream.put_double(look_stick.y)
	stream.put_double(strafe_stick.x)
	stream.put_double(strafe_stick.y)
	stream.put_double(is_left_hand_grab_pressed)
	stream.put_double(is_right_hand_grab_pressed)
	stream.put_double(is_jump_pressed)
