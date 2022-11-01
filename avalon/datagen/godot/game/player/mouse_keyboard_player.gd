extends Player

class_name MouseKeyboardPlayer

export var throw_hand_move_distance := 1.0  # m
export var crouch_speed_coefficient := 0.5

# NOTE: these must be present in `reset_on_new_world`!
var is_crouched := false
var is_left_hand_throwing := false
var is_right_hand_throwing := false


func initialize() -> void:
	.initialize()
	is_crouched = false


func reset_on_new_world():
	.reset_on_new_world()
	is_crouched = false
	is_left_hand_throwing = false
	is_right_hand_throwing = false


func rotate_head(action: AvalonAction, _delta: float):
	var current_rotation = target_head.global_transform.basis.get_euler()
	var new_basis = Basis()
	new_basis = new_basis.rotated(Vector3.UP, action.head_delta_rotation.y + current_rotation.y)
	new_basis = new_basis.rotated(new_basis.x, action.head_delta_rotation.x + current_rotation.x)
	target_head.global_transform.basis = new_basis


func rotate_left_hand(action: AvalonAction, _delta: float):
	# mouse and keyboard players can't rotate their hands separately from the their heads
	HARD.assert(action.left_hand_delta_rotation.is_equal_approx(Vector3.ZERO))
	.rotate_left_hand(action, _delta)


func rotate_right_hand(action: AvalonAction, _delta: float):
	# mouse and keyboard players can't rotate their hands separately from the their heads
	HARD.assert(action.left_hand_delta_rotation.is_equal_approx(Vector3.ZERO))
	.rotate_right_hand(action, _delta)


func get_crouch_height() -> float:
	# this allows you to pick things off the ground while not being too jarring visually
	return height - arm_length


func get_head_vertical_delta_position(action: AvalonAction) -> float:
	# if not currently crouched and pressing crouch button
	var is_crouching = action.is_crouching and not is_crouched
	# if currently crouched and pressing crouch button
	var is_standing = not action.is_crouching and is_crouched

	# prevent crouching while jumping in the air
	if not is_on_floor() and is_crouching:
		return 0.0
	elif is_crouching:
		is_crouched = true
		return -get_crouch_height()
	elif is_standing:
		is_crouched = false
		return get_crouch_height()
	return 0.0


func get_move_velocity(action: AvalonAction, delta: float) -> Vector3:
	var velocity = .get_move_velocity(action, delta)
	if is_crouched:
		velocity *= crouch_speed_coefficient
	return velocity


func move_while_climbing(
	head_delta_position: Vector3,
	_left_hand_delta_position: Vector3,
	_right_hand_delta_position: Vector3,
	_delta: float
) -> void:
	if (
		not climbing_ray.is_colliding()
		or not InteractionHandlers.is_position_climbable(
			get_tree(), climbing_ray.get_collision_point()
		)
	):
		return

	var climbing_velocity = resolve_climbing_velocity(
		climbing_ray,
		# note: the ray points in the same direction as your head
		target_head.global_transform.basis,
		# makes `move_forward` and `move_backward` move you up and down when climbing
		Vector2(head_delta_position.x, -head_delta_position.z)
	)

	var _collision = physical_body.move_and_collide(climbing_velocity, INFINITE_INERTIA)


func get_throw_force_magnitude() -> float:
	# TODO this is the actual force magnitude to relies on mass getting set properly
	# return mass * arm_mass_ratio * (throw_hand_move_distance * PHYSICS_FPS)
	return 10.0


func do_hand_action(hand: PlayerHand, is_grasping: bool, is_throwing: bool, is_eating: bool) -> void:
	var throw_impulse = Vector3.ZERO
	var held_thing = hand.get_held_thing()

	# try eating what's in your hand first before trying to throw it
	if is_eating and InteractionHandlers.is_edible(held_thing):
		var energy = InteractionHandlers.attempt_eat(held_thing)
		physical_body.remove_collision_exception_with(held_thing)
		hand.set_held_thing(null)
		if HARD.mode():
			print("ate %s for %f energy" % [held_thing.name, energy])
		add_energy(energy)
	elif is_throwing:
		throw_impulse = hand.global_transform.basis * Vector3.FORWARD * get_throw_force_magnitude()

	hand.do_action(is_grasping and not is_throwing, physical_body, throw_impulse)

	set_hand_mesh_visibility(hand)


func do_left_hand_action(action: AvalonAction) -> void:
	is_left_hand_throwing = action.is_left_hand_throwing
	do_hand_action(
		target_left_hand,
		action.is_left_hand_grasping,
		action.is_left_hand_throwing,
		action.is_eating
	)


func do_right_hand_action(action: AvalonAction) -> void:
	is_right_hand_throwing = action.is_right_hand_throwing
	do_hand_action(
		target_right_hand,
		action.is_right_hand_grasping,
		action.is_right_hand_throwing,
		action.is_eating
	)


func _get_all_energy_expenditures() -> Dictionary:
	var expenditure = ._get_all_energy_expenditures()

	if is_left_hand_throwing:
		expenditure["physical_left_hand_kinetic_energy_expenditure"] += kinetic_energy_expenditure(
			mass * arm_mass_ratio,
			physical_left_hand.linear_velocity + Vector3.FORWARD * throw_hand_move_distance,
			prev_physical_left_hand_linear_velocity
		)

	if is_right_hand_throwing:
		expenditure["physical_right_hand_kinetic_energy_expenditure"] += kinetic_energy_expenditure(
			mass * arm_mass_ratio,
			physical_right_hand.linear_velocity + Vector3.FORWARD * throw_hand_move_distance,
			prev_physical_right_hand_linear_velocity
		)
	return expenditure


static func resolve_climbing_velocity(
	climbing_ray: RayCast, orientation_basis: Basis, linear_velocity: Vector2
) -> Vector3:
	var normal = climbing_ray.get_collision_normal()
	var forward = orientation_basis.z
	var right = orientation_basis.x
	var climb_basis = Basis(right, normal.cross(right), normal).orthonormalized()

	# `angle_plane` is the plane on which the angle between your ray and the surface normal is defined
	var angle_plane_normal = forward.cross(normal)

	# make sure normal vector of the plane always points to the right so you actually get negative climb angles
	if right.dot(angle_plane_normal) < 0:
		angle_plane_normal *= -1

	# get a signed angle between two vectors that represents your `climb_angle`
	# when `climb_angle` < 0 the climbing surface points towards you
	# when `climb_angle` > 0 the climbing surface points away from you
	# when `climb_angle` == 0 the climbing surface is perpendicular to your
	var climb_angle = rad2deg(
		atan2(normal.cross(Vector3.UP).dot(angle_plane_normal), normal.dot(Vector3.UP))
	)

	# adjust your climbing velocity relative to the surface your climbing
	var climbing_velocity = climb_basis * Vector3(linear_velocity.x, linear_velocity.y, 0)

	# TODO ???
	var eps = 0.1
	if climb_angle > 0 - eps:
		# deprecating this for now, this seems to work fairly well when you get pushed out of the ground instead
		pass
		# climbing_velocity.y += 1.0

	return climbing_velocity
