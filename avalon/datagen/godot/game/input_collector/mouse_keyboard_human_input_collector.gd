extends MouseKeyboardInputCollector

class_name MouseKeyboardHumanInputCollector

var mouse_x: float = 0.0
var mouse_y: float = 0.0

var is_move_forward_pressed: bool = false
var is_move_backward_pressed: bool = false
var is_move_left_pressed: bool = false
var is_move_right_pressed: bool = false
var is_jump_pressed: bool = false
var is_eat_pressed: bool = false
var is_grab_pressed: bool = false
var is_throw_pressed: bool = false
var is_crouch_pressed: bool = false

var is_wheel_up_just_released: bool = false
var is_wheel_down_just_released: bool = false

var is_mouse_mode_toggled: bool = false
var is_active_hand_toggled: bool = false

const MOUSE_SENSITIVITY = 0.004
const MOUSE_PAN_DEADZONE = 0.01
const SCROLL_SENSITIVITY = 0.05

var _current_hand = CONST.RIGHT_HAND

# bullet physics clamps angular velocities that go to fast
# see this code from Godot for more details:
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-372-void btRigidBody::integrateVelocities(btScalar step)
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-373-{
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-374-  if (isStaticOrKinematicObject())
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-375-          return;
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-376-
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-377-  m_linearVelocity += m_totalForce * (m_inverseMass * step);
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-378-  m_angularVelocity += m_invInertiaTensorWorld * m_totalTorque * step;
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-379-
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp:380:#define MAX_ANGVEL SIMD_HALF_PI
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-381-  /// clamp angular velocity. collision calculations will fail on higher angular velocities
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-382-  btScalar angvel = m_angularVelocity.length();
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp:383:  if (angvel * step > MAX_ANGVEL)
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-384-  {
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp:385:          m_angularVelocity *= (MAX_ANGVEL / step) / angvel;
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-386-  }
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-387-  #if defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-388-  clampVelocity(m_angularVelocity);
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-389-  #endif
# thirdparty/bullet/BulletDynamics/Dynamics/btRigidBody.cpp-390-}
const MAX_ROTATION = deg2rad(45)


func is_right_hand_active():
	return _current_hand == CONST.RIGHT_HAND


func is_left_hand_active():
	return _current_hand == CONST.LEFT_HAND


func _init():
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func reset():
	mouse_x = 0.0
	mouse_y = 0.0
	is_move_forward_pressed = false
	is_move_backward_pressed = false
	is_move_left_pressed = false
	is_move_right_pressed = false
	is_jump_pressed = false
	is_eat_pressed = false
	is_grab_pressed = false
	is_throw_pressed = false
	is_crouch_pressed = false
	is_wheel_up_just_released = false
	is_wheel_down_just_released = false
	is_mouse_mode_toggled = false
	is_active_hand_toggled = false


func to_normalized_relative_action(player):
	var action = MouseKeyboardAction.new()

	action.is_left_hand_grasping = is_left_hand_active() and is_grab_pressed
	action.is_right_hand_grasping = is_right_hand_active() and is_grab_pressed
	action.is_left_hand_throwing = is_left_hand_active() and is_throw_pressed
	action.is_right_hand_throwing = is_right_hand_active() and is_throw_pressed
	action.is_jumping = is_jump_pressed
	action.is_eating = is_eat_pressed
	action.is_crouching = is_crouch_pressed
	action.head_delta_position.x += 1.0 if is_move_right_pressed else 0.0
	action.head_delta_position.x -= 1.0 if is_move_left_pressed else 0.0
	action.head_delta_position.z -= 1.0 if is_move_forward_pressed else 0.0
	action.head_delta_position.z += 1.0 if is_move_backward_pressed else 0.0

	action.head_delta_position = Tools.vec3_sphere_clamp(action.head_delta_position, 1.0)
	action.head_delta_rotation = Vector3(
		(
			Tools.clamp_delta_rotation(
				player.target_head.rotation.x,
				(
					deg2rad(player.max_head_angular_speed.x)
					* clamp(mouse_y, -MAX_ROTATION, MAX_ROTATION)
				),
				MAX_PITCH_ROTATION
			)
			/ deg2rad(player.max_head_angular_speed.x)
		),
		clamp(mouse_x, -MAX_ROTATION, MAX_ROTATION),
		0.0
	)

	action.left_hand_delta_position.z += (
		SCROLL_SENSITIVITY
		if is_wheel_down_just_released and is_left_hand_active()
		else 0.0
	)
	action.left_hand_delta_position.z -= (
		SCROLL_SENSITIVITY
		if is_wheel_up_just_released and is_left_hand_active()
		else 0.0
	)
	action.right_hand_delta_position.z += (
		SCROLL_SENSITIVITY
		if is_wheel_down_just_released and is_right_hand_active()
		else 0.0
	)
	action.right_hand_delta_position.z -= (
		SCROLL_SENSITIVITY
		if is_wheel_up_just_released and is_right_hand_active()
		else 0.0
	)

	action.head_delta_rotation = Tools.normalize(
		action.head_delta_rotation, -MAX_ROTATION, MAX_ROTATION
	)

	return action


func read_input_from_event(event: InputEvent) -> void:
	if event is InputEventPanGesture:
		if abs(event.delta.y) > MOUSE_PAN_DEADZONE:
			if event.delta.y > 0:
				Input.action_press("wheel_up")
				Input.action_release("wheel_up")
			else:
				Input.action_press("wheel_down")
				Input.action_release("wheel_down")

	if event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		mouse_x += -event.relative.x * MOUSE_SENSITIVITY
		mouse_y += -event.relative.y * MOUSE_SENSITIVITY

	if event.is_action_pressed("toggle_mouse_capture"):
		is_mouse_mode_toggled = true

		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

	if event.is_action_pressed("toggle_active_hand"):
		is_active_hand_toggled = true
		match _current_hand:
			CONST.RIGHT_HAND:
				_current_hand = CONST.LEFT_HAND
			CONST.LEFT_HAND:
				_current_hand = CONST.RIGHT_HAND
			_:
				HARD.assert(false, "Not a valid hand %s" % _current_hand)


func read_input_before_physics() -> void:
	is_move_forward_pressed = Input.is_action_pressed("move_forward", true)
	is_move_backward_pressed = Input.is_action_pressed("move_backward", true)
	is_move_left_pressed = Input.is_action_pressed("move_left", true)
	is_move_right_pressed = Input.is_action_pressed("move_right", true)
	is_jump_pressed = Input.is_action_pressed("jump", true)
	is_eat_pressed = Input.is_action_pressed("eat", true)
	is_grab_pressed = Input.is_action_pressed("grab", true)
	is_throw_pressed = Input.is_action_pressed("throw", true)
	is_crouch_pressed = Input.is_action_pressed("crouch", true)
	is_wheel_up_just_released = Input.is_action_just_released("wheel_up")
	is_wheel_down_just_released = Input.is_action_just_released("wheel_down")


func write_into_stream(stream: StreamPeerBuffer, _player) -> void:
	stream.put_float(is_move_forward_pressed)
	stream.put_float(is_move_backward_pressed)
	stream.put_float(is_move_left_pressed)
	stream.put_float(is_move_right_pressed)
	stream.put_float(is_jump_pressed)
	stream.put_float(is_eat_pressed)
	stream.put_float(is_grab_pressed)
	stream.put_float(is_throw_pressed)
	stream.put_float(is_crouch_pressed)
	stream.put_float(is_wheel_up_just_released)
	stream.put_float(is_wheel_down_just_released)
	stream.put_float(is_mouse_mode_toggled)
	stream.put_float(is_active_hand_toggled)
