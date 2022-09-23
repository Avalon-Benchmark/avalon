extends VRPlayer

class_name VRHumanPlayer

var arvr_origin: Spatial
var arvr_camera: Spatial
var arvr_left_hand: Spatial
var arvr_right_hand: Spatial

var prev_arvr_origin_transform: Transform
var prev_arvr_camera_global_transform: Transform
var prev_arvr_left_hand_global_transform: Transform
var prev_arvr_right_hand_global_transform: Transform

# there isn't a great way to tell when we've stopped tracking other than the numbers get very, very large
const _VERY_LARGE_NUMBER_FOR_WHEN_TRACKING_BREAKS = 100000
const NUM_FRAMES_BEFORE_VR_ENABLED = 20

var was_not_tracking := false
var human_height := 2.0
var is_vr_warmed_up := false


func initialize():
	.initialize()
	# don't limit arm length for VR human players
	arm_length = INF
	extra_height_margin = 0.2


func set_target_body_transforms():
	target_head.global_transform = arvr_camera.global_transform
	target_left_hand.global_transform = arvr_left_hand.global_transform
	target_right_hand.global_transform = arvr_right_hand.global_transform
	physical_body.global_transform.origin = Vector3(
		arvr_camera.global_transform.origin.x,
		arvr_origin.transform.origin.y + height / 2,
		arvr_camera.global_transform.origin.z
	)
	collision_head.global_transform = arvr_camera.global_transform


func is_warming_up(delta: float, frame: int) -> bool:
	if not is_vr_warmed_up and frame == NUM_FRAMES_BEFORE_VR_ENABLED:
		is_vr_warmed_up = true
		set_human_height()
		return true

	# VR needs a "warm-up" period before the controllers start to be tracked
	if not is_vr_warmed_up and frame < NUM_FRAMES_BEFORE_VR_ENABLED:
		# move all the nodes to the correct starting point or else it will think there's velocities to apply
		set_target_body_transforms()
		# then move the physical body to where the targets are
		apply_action_to_physical_body(null, delta)
		# and finally reset the old transforms
		update_previous_transforms_and_velocities(true)
		return true

	return false


func set_spawn(spawn_transform: Transform) -> void:
	reset_on_new_world()

	var relative_arvr_camera_distance_from_origin = (
		arvr_camera.global_transform.origin
		- arvr_origin.transform.origin
	)

	arvr_origin.transform.origin = Vector3(
		spawn_transform.origin.x - relative_arvr_camera_distance_from_origin.x,
		spawn_transform.origin.y - height / 2 + (height - human_height),
		spawn_transform.origin.z - relative_arvr_camera_distance_from_origin.z
	)

	arvr_origin.force_update_transform()

	.set_spawn(spawn_transform)

	prev_arvr_origin_transform = arvr_origin.transform

	# manually override prev arvr global transforms so we can get the right first action when we spawn
	prev_arvr_camera_global_transform = target_head.global_transform
	prev_arvr_left_hand_global_transform = target_left_hand.global_transform
	prev_arvr_right_hand_global_transform = target_right_hand.global_transform


func set_nodes_in_ready() -> void:
	.set_nodes_in_ready()

	arvr_origin = get_node("arvr_origin")
	arvr_camera = arvr_origin.get_node("arvr_camera")
	eyes = _get_eyes()

	# TODO sigh .... they really shouldn't be using the child node here
	arvr_left_hand = arvr_origin.get_node("arvr_left_hand").get_node("hand")
	arvr_right_hand = arvr_origin.get_node("arvr_right_hand").get_node("hand")

	target_left_hand.get_node("hand/active").visible = false
	target_left_hand.get_node("hand/default").visible = false
	target_left_hand.get_node("hand/disabled").visible = false
	target_right_hand.get_node("hand/active").visible = false
	target_right_hand.get_node("hand/default").visible = false
	target_right_hand.get_node("hand/disabled").visible = false

	set_target_body_transforms()


func _get_eyes() -> Node:
	return arvr_camera


func configure_nodes_for_playback(
	arvr_camera_transform: Transform,
	arvr_left_hand_transform: Transform,
	arvr_right_hand_transform: Transform,
	arvr_origin_transform: Transform,
	new_human_height: float,
	is_updating_arvr_origin: bool = false
):
	if is_updating_arvr_origin:
		arvr_origin.transform = arvr_origin_transform
		arvr_origin.force_update_transform()

	arvr_camera.transform = arvr_camera_transform
	arvr_camera.force_update_transform()

	arvr_left_hand.get_parent().transform = arvr_left_hand_transform
	arvr_left_hand.get_parent().force_update_transform()

	arvr_right_hand.get_parent().transform = arvr_right_hand_transform
	arvr_right_hand.get_parent().force_update_transform()

	human_height = new_human_height


func set_human_height() -> void:
	human_height = (arvr_camera.global_transform.origin - arvr_origin.transform.origin).y


func rotate_head(action: AvalonAction, delta: float):
	.rotate_head(action, delta)

	var head_delta_quat = Tools.get_delta_quaternion(
		prev_arvr_camera_global_transform.basis.get_rotation_quat(),
		arvr_camera.global_transform.basis.get_rotation_quat()
	)
	var head_and_origin_delta_quat = Quat(action.head_delta_rotation)
	var origin_delta_quat = head_and_origin_delta_quat * head_delta_quat.inverse()

	# to perform stick rotation we need to rotate then move the origin so that the camera doesn't "move" positions
	var new_arvr_origin_quat = (
		Quat(action.relative_origin_delta_rotation)
		* arvr_origin.transform.basis.get_rotation_quat()
	)

	var is_relative_origin_delta_rotation_zero = (
		action.relative_origin_delta_rotation.x == 0.0
		and action.relative_origin_delta_rotation.y == 0.0
		and action.relative_origin_delta_rotation.z == 0.0
	)

	# NOTE: sometimes godot when you apply a "null" rotation to something it doesn't exactly equal what it did before
	if not is_relative_origin_delta_rotation_zero:
		arvr_origin.transform.basis = Basis(new_arvr_origin_quat)

	# move camera back to where it was before the rotation by adusting the origin
	var origin_position_relative_to_camera = (
		arvr_origin.transform.origin
		- arvr_camera.global_transform.origin
	)
	arvr_origin.transform.origin += (
		origin_delta_quat * origin_position_relative_to_camera
		- origin_position_relative_to_camera
	)


func move_head(action: AvalonAction) -> void:
	.move_head(action)

	var actual_body_delta_position = get_actual_delta_position_after_move()
	var head_vertical_delta_position = (
		target_head.global_transform.origin.y
		- prev_target_head_global_transform.origin.y
		- actual_body_delta_position.y
	)
	var global_feet_position = physical_body.global_transform.origin.y - height / 2
	var current_head_height = target_head.global_transform.origin.y - global_feet_position
	var extra_distance_above_height = clamp(
		current_head_height - (height + extra_height_margin), 0, INF
	)
	arvr_origin.transform.origin.y += (
		(head_vertical_delta_position) * (height - human_height) / height
		- extra_distance_above_height
	)
	target_head.global_transform.origin.y -= extra_distance_above_height


func move(action: AvalonAction, delta: float) -> void:
	# do base move and adjust VR nodes accordingly
	.move(action, delta)

	# get the total distance your body moved
	# 	your body may not move as far as expected, for example due to collisions
	var actual_body_delta_position = get_actual_delta_position_after_move()
	# get how much you moved due to you head position changing
	#	the `arvr_camera` moves relative to the origin
	var actual_global_head_delta_position = (
		arvr_camera.global_transform.origin
		- prev_arvr_camera_global_transform.origin
	)

	# while on the ground, we don't want to adjust the arvr origin due to crouching
	if not is_climbing():
		actual_global_head_delta_position.y = 0
	# we need to adjust your origin position based on how you actually moved WITHOUT considering how much your head moved
	#	Note: when using the stick, your origin gets moved
	# 	(talk to bryden for more details)
	arvr_origin.transform.origin += (actual_body_delta_position - actual_global_head_delta_position)


func update_previous_transforms_and_velocities(is_reset_for_spawn = false) -> void:
	.update_previous_transforms_and_velocities(is_reset_for_spawn)

	if not is_unable_to_track(arvr_origin):
		prev_arvr_origin_transform = arvr_origin.transform
	if not is_unable_to_track(arvr_camera):
		prev_arvr_camera_global_transform = arvr_camera.global_transform
	if not is_unable_to_track(arvr_left_hand):
		prev_arvr_left_hand_global_transform = arvr_left_hand.global_transform
	if not is_unable_to_track(arvr_right_hand):
		prev_arvr_right_hand_global_transform = arvr_right_hand.global_transform


func _is_vector_finite(vec: Vector3) -> bool:
	return (
		vec.length() < _VERY_LARGE_NUMBER_FOR_WHEN_TRACKING_BREAKS
		and not is_inf(vec.length())
		and not is_nan(vec.length())
	)


func _is_basis_finite(basis: Basis) -> bool:
	return _is_vector_finite(basis.x) and _is_vector_finite(basis.y) and _is_vector_finite(basis.z)


func _is_transform_finite(transform: Transform) -> bool:
	return _is_basis_finite(transform.basis) and _is_vector_finite(transform.origin)


func _is_node_finite(node: Spatial) -> bool:
	return is_instance_valid(node) and _is_transform_finite(node.global_transform)


func is_unable_to_track(node: Spatial) -> bool:
	if not _is_node_finite(node):
		printt("unable to track:", node)
	return not _is_node_finite(node)


func is_any_node_not_able_to_track() -> bool:
	return (
		is_unable_to_track(arvr_camera)
		or is_unable_to_track(arvr_left_hand)
		or is_unable_to_track(arvr_right_hand)
	)


func fix_tracking_once_working() -> bool:
	var is_tracking_not_working = is_any_node_not_able_to_track()

	if was_not_tracking and not is_tracking_not_working:
		# move target bodies where they should be once we start tracking again
		target_head.global_transform = arvr_camera.global_transform
		target_left_hand.global_transform = arvr_left_hand.global_transform
		target_right_hand.global_transform = arvr_right_hand.global_transform

		physical_body.global_transform.origin = Vector3(
			arvr_camera.global_transform.origin.x,
			(arvr_origin.transform.origin.y - (height - human_height)) + height / 2,
			arvr_camera.global_transform.origin.z
		)
		physical_head.global_transform = target_head.global_transform
		physical_left_hand.global_transform = target_left_hand.global_transform
		physical_right_hand.global_transform = target_right_hand.global_transform

		# store the previous state of the physical body
		update_previous_transforms_and_velocities()

	was_not_tracking = is_tracking_not_working

	return is_tracking_not_working


func get_hand_meshes(hand: PlayerHand) -> Array:
	var actual_hand: Spatial
	if "left" in hand.name:
		actual_hand = arvr_left_hand
	if "right" in hand.name:
		actual_hand = arvr_right_hand
	return [
		actual_hand.get_node("active"),
		actual_hand.get_node("disabled"),
		actual_hand.get_node("default")
	]
