extends Player

class_name VRPlayer


func rotate_head(action: AvalonAction, _delta: float):
	var quat = (
		Quat(action.head_delta_rotation)
		* target_head.global_transform.basis.get_rotation_quat()
	)
	var new_basis = Basis(quat)
	# Uncomment this line to lock head rotation (roll) during agent training
	# new_basis = new_basis.rotated(new_basis.z, -1 * target_head.global_transform.basis.get_euler().z)
	target_head.global_transform.basis = new_basis


func move_while_climbing(
	_head_delta_position: Vector3,
	left_hand_delta_position: Vector3,
	right_hand_delta_position: Vector3,
	_delta: float
) -> void:
	gravity_velocity = Vector3.ZERO

	# climbing movement is only dictated by how much your hands move and not
	#	how much your body moved
	var total_climb_movement = Vector3.ZERO

	# TODO this let's you move twice as fast if you move in the same direction with both hands
	if target_left_hand.is_grasping_heavy_thing():
		total_climb_movement -= target_head.global_transform.basis * left_hand_delta_position
	if target_right_hand.is_grasping_heavy_thing():
		total_climb_movement -= target_head.global_transform.basis * right_hand_delta_position

	var _collision_body = physical_body.move_and_collide(total_climb_movement, INFINITE_INERTIA)


func do_left_hand_action(action: AvalonAction) -> void:
	target_left_hand.do_action(action.is_left_hand_grasping, physical_body)
	set_hand_mesh_visibility(target_left_hand)


func do_right_hand_action(action: AvalonAction) -> void:
	target_right_hand.do_action(action.is_right_hand_grasping, physical_body)
	set_hand_mesh_visibility(target_right_hand)


func move_left_hand(action: AvalonAction, _delta: float) -> void:
	if target_left_hand.is_grasping_heavy_thing():
		target_left_hand.global_transform.origin = Tools.get_new_hand_position_while_climbing(
			target_left_hand.global_transform.origin,
			action.left_hand_delta_position,
			prev_target_head_global_transform,
			target_head.global_transform,
			arm_length
		)
	else:
		target_left_hand.global_transform.origin = Tools.get_new_hand_position(
			target_left_hand.global_transform.origin,
			action.left_hand_delta_position,
			prev_target_head_global_transform,
			target_head.global_transform,
			arm_length
		)


func move_right_hand(action: AvalonAction, _delta: float) -> void:
	if target_right_hand.is_grasping_heavy_thing():
		target_right_hand.global_transform.origin = Tools.get_new_hand_position_while_climbing(
			target_right_hand.global_transform.origin,
			action.right_hand_delta_position,
			prev_target_head_global_transform,
			target_head.global_transform,
			arm_length
		)
	else:
		target_right_hand.global_transform.origin = Tools.get_new_hand_position(
			target_right_hand.global_transform.origin,
			action.right_hand_delta_position,
			prev_target_head_global_transform,
			target_head.global_transform,
			arm_length
		)
