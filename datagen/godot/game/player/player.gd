extends Node

class_name Player

var spec: PlayerSpec

var target_player: Spatial
var target_head: Spatial
var target_left_hand
var target_right_hand
var physical_player: Spatial
var physical_body: PhysicsBody
var physical_head: Spatial
var physical_left_hand: Spatial
var physical_right_hand: Spatial
var collision_body: CollisionShape
var collision_head: CollisionShape
var climbing_ray: RayCast
var ground_ray: RayCast
var eat_area: Area

# NOTE: these must be present in `reset_on_new_world`!
var hit_points: float
var gravity_velocity := Vector3.ZERO
var surface_normal := Vector3.UP
var is_jumping := false
var current_observation := {}
var _hit_points_gained_from_eating: float = 0.0
var _hit_points_lost_from_enemies: float = 0.0
var is_dead := false
var y_vel_history = []

# TODO make this exportable once they actually dynamically update things
export var height := 2.0  # TODO set height during VR calibration
export var head_radius := 0.1  # from average human head circumference (57cm)

export var max_head_linear_speed := 0.3
export var max_head_angular_speed := Vector3(5.0, 10.0, 1.0)
export var max_hand_linear_speed := 1.0
export var max_hand_angular_speed := 15.0
export var jump_height := 1.5
export var arm_length := 1.5
export var starting_hit_points := 1.0
export var mass := 60.0  # TODO change player mass to 60kg and update all items to be relative to this
export var arm_mass_ratio := 0.05  # from https://exrx.net/Kinesiology/Segments
export var head_mass_ratio := 0.08  # from https://exrx.net/Kinesiology/Segments
export var standup_speed_after_climbing := 0.1  # in m/s
export var min_head_position_off_of_floor := 0.2
export var push_force_magnitude := 5.0
export var throw_force_magnitude := 3.0
export var starting_left_hand_position_relative_to_head := Vector3(-0.5, -0.5, -0.5)
export var starting_right_hand_position_relative_to_head := Vector3(0.5, -0.5, -0.5)
export var minimum_fall_speed := 10.0
export var fall_damage_coefficient := 0.000026
export var total_energy_coefficient := 0.0
export var body_kinetic_energy_coefficient := 0.0
export var body_potential_energy_coefficient := 0.0
export var head_potential_energy_coefficient := 0.0
export var left_hand_kinetic_energy_coefficient := 0.0
export var left_hand_potential_energy_coefficient := 0.0
export var right_hand_kinetic_energy_coefficient := 0.0
export var right_hand_potential_energy_coefficient := 0.0
# TODO this isn't quite the right place for it but better than the alternatives
export var num_frames_alive_after_food_is_gone := 50
export var eat_area_radius := 0.25
export var is_displaying_debug_meshes := false

# `move_and_slide` constants, we should keep this fixed because they can change the actual movement distance
const UP_DIRECTION := Vector3.UP
const STOP_ON_SLOPE := true
const MAX_SLIDES := 4  # DO NOT CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING
const FLOOR_MAX_ANGLE := deg2rad(45)
const INFINITE_INERTIA := false
var PHYSICS_FPS: float = ProjectSettings.get_setting("physics/common/physics_fps")
var GRAVITY_MAGNITUDE: float = ProjectSettings.get_setting("physics/3d/default_gravity")

# storing previous state of the body
var prev_hit_points: float
var prev_target_head_global_transform: Transform
var prev_target_left_hand_global_transform: Transform
var prev_target_right_hand_global_transform: Transform
var prev_physical_body_global_transform: Transform
var prev_physical_head_global_transform: Transform
var prev_physical_left_hand_global_transform: Transform
var prev_physical_right_hand_global_transform: Transform
var prev_physical_body_linear_velocity: Vector3
var prev_physical_head_linear_velocity: Vector3
var prev_physical_left_hand_linear_velocity: Vector3
var prev_physical_right_hand_linear_velocity: Vector3
var prev_physical_head_angular_velocity: Vector3
var prev_physical_left_hand_angular_velocity: Vector3
var prev_physical_right_hand_angular_velocity: Vector3


func _ready() -> void:
	ready()
	get_nodes()
	update_previous_transforms_and_velocities(true)
	# reset current observation, happens after `update_previous_transforms_and_velocities` so everthing starts at 0
	set_current_observation()


func ready():
	# TODO temporary hack until we can remove sim specs
	max_head_linear_speed = spec.max_head_linear_speed
	max_head_angular_speed = spec.max_head_angular_speed
	max_hand_linear_speed = spec.max_hand_linear_speed
	max_hand_angular_speed = spec.max_hand_angular_speed
	jump_height = spec.jump_height
	arm_length = spec.arm_length
	starting_hit_points = spec.starting_hit_points
	mass = spec.mass
	arm_mass_ratio = spec.arm_mass_ratio
	head_mass_ratio = spec.head_mass_ratio
	standup_speed_after_climbing = spec.standup_speed_after_climbing
	min_head_position_off_of_floor = spec.min_head_position_off_of_floor
	push_force_magnitude = spec.push_force_magnitude
	throw_force_magnitude = spec.throw_force_magnitude
	minimum_fall_speed = spec.minimum_fall_speed
	fall_damage_coefficient = spec.fall_damage_coefficient
	total_energy_coefficient = spec.total_energy_coefficient
	body_kinetic_energy_coefficient = spec.body_kinetic_energy_coefficient
	body_potential_energy_coefficient = spec.body_potential_energy_coefficient
	head_potential_energy_coefficient = spec.head_potential_energy_coefficient
	left_hand_kinetic_energy_coefficient = spec.left_hand_kinetic_energy_coefficient
	left_hand_potential_energy_coefficient = spec.left_hand_potential_energy_coefficient
	right_hand_kinetic_energy_coefficient = spec.right_hand_kinetic_energy_coefficient
	right_hand_potential_energy_coefficient = spec.right_hand_potential_energy_coefficient
	# TODO this isn't quite the right place for it but better than the alternatives
	num_frames_alive_after_food_is_gone = spec.num_frames_alive_after_food_is_gone
	eat_area_radius = spec.eat_area_radius
	is_displaying_debug_meshes = spec.is_displaying_debug_meshes
	hit_points = starting_hit_points

	# TODO probably more check we can add here
	HARD.assert(max_head_linear_speed > 0, "`max_head_linear_speed` must be greater than 0")
	HARD.assert(max_head_angular_speed.x > 0, "`max_head_angular_speed.x` must be greater than 0")
	HARD.assert(max_head_angular_speed.y > 0, "`max_head_angular_speed.y` must be greater than 0")
	HARD.assert(max_head_angular_speed.z > 0, "`max_head_angular_speed.z` must be greater than 0")
	HARD.assert(max_hand_linear_speed > 0, "`max_hand_linear_speed` must be greater than 0")
	HARD.assert(max_hand_angular_speed > 0, "`max_hand_angular_speed` must be greater than 0")
	HARD.assert(jump_height > 0, "`jump_height` must be greater than 0")


func get_nodes():
	target_player = get_node("target_player")
	target_head = target_player.get_node("target_head")
	target_left_hand = target_player.get_node("target_left_hand")
	target_right_hand = target_player.get_node("target_right_hand")

	physical_player = get_node("physical_player")
	physical_body = physical_player.get_node("physical_body")
	physical_head = physical_player.get_node("physical_head")
	physical_left_hand = physical_player.get_node("physical_left_hand")
	physical_right_hand = physical_player.get_node("physical_right_hand")
	collision_body = physical_body.get_node("collision_body")
	collision_head = physical_body.get_node("collision_head")
	# TODO these depend on height
	climbing_ray = physical_head.get_node("climbing_ray")
	ground_ray = physical_body.get_node("ground_ray")

	# TODO parameterize the eat area shape?
	eat_area = physical_head.get_node("eat_area")

	climbing_ray.add_exception(physical_body)
	climbing_ray.add_exception(physical_left_hand)
	climbing_ray.add_exception(physical_right_hand)
	ground_ray.add_exception(physical_head)
	ground_ray.add_exception(physical_body)
	ground_ray.add_exception(physical_left_hand)
	ground_ray.add_exception(physical_right_hand)

	target_left_hand.global_transform.origin = (
		starting_left_hand_position_relative_to_head
		+ target_head.global_transform.origin
	)
	target_right_hand.global_transform.origin = (
		starting_right_hand_position_relative_to_head
		+ target_head.global_transform.origin
	)

	# make the eat area comically large so we can test bringing food into our face
	var shape: SphereShape = eat_area.get_child(0).shape
	shape.radius = eat_area_radius

	HARD.assert(
		arm_length > starting_left_hand_position_relative_to_head.length(),
		(
			"Arm length %s must be longer than starting left hand position %s"
			% [arm_length, starting_left_hand_position_relative_to_head.length()]
		)
	)
	HARD.assert(
		arm_length > starting_right_hand_position_relative_to_head.length(),
		(
			"Arm length %s must be longer than starting right hand position %s"
			% [arm_length, starting_right_hand_position_relative_to_head.length()]
		)
	)

	if is_displaying_debug_meshes:
		physical_head.get_node("beak").visible = true
		physical_head.get_node("face").visible = true
		physical_left_hand.get_node("marker").visible = true
		physical_right_hand.get_node("marker").visible = true


# TODO need to think of a safer way to reset agent state between videos
func reset_on_new_world() -> void:
	# reset any internal state when moving to a new world
	hit_points = starting_hit_points
	gravity_velocity = Vector3.ZERO
	surface_normal = Vector3.UP
	is_jumping = false
	current_observation = {}
	_hit_points_gained_from_eating = 0.0
	_hit_points_lost_from_enemies = 0.0
	is_dead = false
	y_vel_history = []

	physical_head.linear_velocity = Vector3.ZERO
	physical_head.angular_velocity = Vector3.ZERO
	physical_left_hand.linear_velocity = Vector3.ZERO
	physical_left_hand.angular_velocity = Vector3.ZERO
	physical_right_hand.linear_velocity = Vector3.ZERO
	physical_right_hand.angular_velocity = Vector3.ZERO

	# reset hand state
	for hand in get_hands():
		var thing = hand.get_held_thing()
		if is_instance_valid(thing):
			physical_body.remove_collision_exception_with(thing)
			hand.set_held_thing(null)
		hand.set_thing_colliding_with_hand(null)


func set_current_observation() -> void:
	current_observation = _get_proprioceptive_oberservation()


func set_spawn(spawn_transform: Transform):
	reset_on_new_world()

	var new_basis = Basis()
	new_basis = new_basis.rotated(Vector3.UP, spawn_transform.basis.get_euler().y)
	new_basis = new_basis.rotated(new_basis.x, spawn_transform.basis.get_euler().x)

	target_head.global_transform.origin = spawn_transform.origin + Vector3(0, 1, 0)
	target_head.global_transform.basis = new_basis
	target_left_hand.global_transform.origin = (
		new_basis * starting_left_hand_position_relative_to_head
		+ target_head.global_transform.origin
	)

	target_left_hand.global_transform.basis = new_basis
	target_right_hand.global_transform.origin = (
		new_basis * starting_right_hand_position_relative_to_head
		+ target_head.global_transform.origin
	)
	target_right_hand.global_transform.basis = new_basis

	physical_body.global_transform.origin = spawn_transform.origin
	physical_head.global_transform.origin = spawn_transform.origin + Vector3(0, 1, 0)
	physical_head.global_transform.basis = new_basis

	physical_left_hand.global_transform.origin = target_left_hand.global_transform.origin
	physical_left_hand.global_transform.basis = target_left_hand.global_transform.basis
	physical_right_hand.global_transform.origin = target_right_hand.global_transform.origin
	physical_right_hand.global_transform.basis = target_right_hand.global_transform.basis

	# store the previous state of the physical body
	update_previous_transforms_and_velocities(true)

	# reset current observation, happens after `update_previous_transforms_and_velocities` so everthing starts at 0
	set_current_observation()


func apply_action(action: AvalonAction, delta: float):
	# head collision shape is actually part of your body -- make sure it get's moved so climbing will work
	update_head_collision_shape()

	# BEWARE: you must perform the following actions in this exact order or else things will not work
	rotate_head(action, delta)
	rotate_left_hand(action, delta)
	rotate_right_hand(action, delta)

	# we MUST move our body before moving our hands, it determines how far our hands can actually move
	move(action, delta)

	move_left_hand(action, delta)
	move_right_hand(action, delta)

	do_left_hand_action(action)
	do_right_hand_action(action)

	# check to see if any food is being held near the agents mouth so it can nom nom nom
	eat()

	# set the velocity of the physical body so the physical body tries to move to where the fake body is
	apply_action_to_physical_body(action, delta)

	# this needs happens after we've updated the physical bodies linear velocity
	set_current_observation()

	# once we've fully consumed the action, update previous transforms and velocities
	update_previous_transforms_and_velocities()


func apply_action_to_physical_body(_action: AvalonAction, delta: float):
	physical_head.linear_velocity = (
		(target_head.global_transform.origin - physical_head.global_transform.origin)
		/ delta
	)
	physical_head.angular_velocity = (
		Tools.calc_angular_velocity_from_basis(
			physical_head.global_transform.basis, target_head.global_transform.basis
		)
		/ delta
	)

	physical_left_hand.linear_velocity = (
		(target_left_hand.global_transform.origin - physical_left_hand.global_transform.origin)
		/ delta
	)
	physical_left_hand.angular_velocity = (
		Tools.calc_angular_velocity_from_basis(
			physical_left_hand.global_transform.basis, target_left_hand.global_transform.basis
		)
		/ delta
	)

	physical_right_hand.linear_velocity = (
		(target_right_hand.global_transform.origin - physical_right_hand.global_transform.origin)
		/ delta
	)
	physical_right_hand.angular_velocity = (
		Tools.calc_angular_velocity_from_basis(
			physical_right_hand.global_transform.basis, target_right_hand.global_transform.basis
		)
		/ delta
	)


func update_head_collision_shape() -> void:
	collision_head.global_transform = target_head.global_transform


func rotate_head(_action: AvalonAction, _delta: float):
	HARD.assert(false, "`rotate_head` not implemented")


func get_new_hand_basis(current_hand_basis: Basis, relative_delta_hand_rotation: Vector3) -> Basis:
	return Basis(
		(
			Quat(relative_delta_hand_rotation)
			* get_head_delta_quaternion()
			* current_hand_basis.get_rotation_quat()
		)
	)


func rotate_left_hand(action: AvalonAction, _delta: float):
	target_left_hand.global_transform.basis = get_new_hand_basis(
		target_left_hand.global_transform.basis, action.left_hand_delta_rotation
	)


func rotate_right_hand(action: AvalonAction, _delta: float):
	target_right_hand.global_transform.basis = get_new_hand_basis(
		target_right_hand.global_transform.basis, action.right_hand_delta_rotation
	)


func move_while_climbing(
	_head_delta_position: Vector3,
	_left_hand_delta_position: Vector3,
	_right_hand_delta_position: Vector3,
	_delta: float
) -> void:
	HARD.assert(false, "`move_while_climbing` not implemented")


func jump(action: AvalonAction, _delta: float) -> void:
	is_jumping = (is_on_floor() or is_able_to_wall_jump()) and action.is_jumping
	if is_jumping:
		gravity_velocity += get_jump_velocity() + get_velocity_change_due_to_gravity()

		surface_normal = UP_DIRECTION

		var _actual_velocity = physical_body.move_and_slide(
			gravity_velocity,
			UP_DIRECTION,
			STOP_ON_SLOPE,
			MAX_SLIDES,
			FLOOR_MAX_ANGLE,
			INFINITE_INERTIA
		)


func get_move_velocity(action: AvalonAction, delta: float) -> Vector3:
	var head_delta_position = action.head_delta_position
	return (
		Tools.get_move_delta_position(target_head.global_transform.basis, head_delta_position)
		/ delta
	)


func move_while_not_climbing(action: AvalonAction, delta: float) -> void:
	var body_velocity = get_move_velocity(action, delta)
	gravity_velocity += get_velocity_change_due_to_gravity()

	# adjust gravity to point into the surface you're traversing
	var projected_gravity = gravity_velocity.project(-surface_normal)

	var _actual_velocity = physical_body.move_and_slide(
		body_velocity + projected_gravity,
		UP_DIRECTION,
		STOP_ON_SLOPE,
		MAX_SLIDES,
		FLOOR_MAX_ANGLE,
		INFINITE_INERTIA
	)

	# update surface normal, helpful for climbing up ramps or slanted terrain
	var floor_collision = get_floor_collision()
	if floor_collision:
		surface_normal = floor_collision.normal
	else:
		surface_normal = UP_DIRECTION

	# don't accumulate gravity while on the ground
	if is_on_floor():
		gravity_velocity = Vector3.ZERO

	# only apply impulses if not on the ground, this should make it easier to jump on things
	if floor_collision and is_on_floor():
		apply_collision_impulses(floor_collision.collider, body_velocity)


func apply_collision_impulses(floor_node: Node, body_velocity: Vector3):
	var slide_count = physical_body.get_slide_count()
	if slide_count > 0:
		for i in range(slide_count):
			var slide_collision: KinematicCollision = physical_body.get_slide_collision(i)
			var collider = slide_collision.collider

			if InteractionHandlers.is_pushable(collider) and collider != floor_node:
				collider.sleeping = false

				# clamp the mass ratio to 1 so small objects CANNOT move faster than the player
				#	add a pushing force so we can move objects heavier then ourselves but don't
				#	end up pushing small objects faster
				var mass_ratio = clamp(mass / collider.mass * push_force_magnitude, 0, 1)

				# the collision position is not consistent, just make the impulse occur at the x, z position
				var collision_offset = slide_collision.position - collider.global_transform.origin
				collision_offset.y = 0

				# scale collision magnitude by objects current speed
				#	this has the tradeoff of not applying the "right" impulse to objects but it prevents
				#	multiple impulses from getting applied
				var collision_magnitude = clamp(
					body_velocity.length() - collider.linear_velocity.length(), 0, INF
				)
				var collision_impulse = (
					collision_magnitude
					* body_velocity.normalized()
					* mass_ratio
					* collider.mass
				)

				InteractionHandlers.attempt_push(collider, collision_impulse, collision_offset)


func get_head_vertical_delta_position(action: AvalonAction) -> float:
	return action.head_delta_position.y


func get_head_distance_from_feet(
	head_vertical_delta_position: float, curr_head_distance_from_feet: float
) -> float:
	return curr_head_distance_from_feet + head_vertical_delta_position


func move_head(action: AvalonAction) -> void:
	var head_vertical_delta_position = get_head_vertical_delta_position(action)
	var global_feet_position = physical_body.global_transform.origin.y - height / 2
	var distance_body_moved_vertically = get_actual_delta_position_after_move().y
	var new_global_head_position = (
		distance_body_moved_vertically
		+ target_head.global_transform.origin.y
	)
	var curr_head_distance_from_feet = new_global_head_position - global_feet_position
	var new_head_distance_from_feet = get_head_distance_from_feet(
		head_vertical_delta_position, curr_head_distance_from_feet
	)

	# when climbing, still move your target head up by how much your body moved
	#	but don't move your head up and down your body (like when you crouch)
	if is_climbing():
		target_head.global_transform.origin = Vector3(
			physical_body.global_transform.origin.x,
			new_global_head_position,
			physical_body.global_transform.origin.z
		)
	# this enables a "crouching" behaviour by moving your head up and down inside your body
	else:
		target_head.global_transform.origin = Vector3(
			physical_body.global_transform.origin.x,
			global_feet_position + new_head_distance_from_feet,
			physical_body.global_transform.origin.z
		)


func move_body_out_of_terrain(_delta: float) -> void:
	var normal = ground_ray.get_collision_normal()
	var _collision = physical_body.move_and_collide(
		normal.normalized() * standup_speed_after_climbing
	)


func is_climbing():
	return target_left_hand.is_grasping_heavy_thing() or target_right_hand.is_grasping_heavy_thing()


func move(action: AvalonAction, delta: float) -> void:
	if (
		ground_ray.is_colliding()
		and not is_climbing()
		and Tools.is_static(ground_ray.get_collider())
	):
		move_body_out_of_terrain(delta)
		gravity_velocity = Vector3.ZERO
	elif is_climbing():
		# use `collision_head` while climbing so you can pull yourself up on ledges
		collision_body.disabled = true
		collision_head.disabled = false

		# TODO refactor to just take in an action
		move_while_climbing(
			action.head_delta_position,
			action.left_hand_delta_position,
			action.right_hand_delta_position,
			delta
		)
		gravity_velocity = Vector3.ZERO
	else:
		# use `collision_body` while walking around so you know when you're touching the ground
		collision_body.disabled = false
		collision_head.disabled = true

		move_while_not_climbing(action, delta)

		# `jump` must happen after move for `is_on_floor` to work
		jump(action, delta)

	move_head(action)


func move_left_hand(action: AvalonAction, _delta: float) -> void:
	target_left_hand.global_transform.origin = Tools.get_new_hand_position(
		target_left_hand.global_transform.origin,
		action.left_hand_delta_position,
		prev_target_head_global_transform,
		target_head.global_transform,
		arm_length
	)


func move_right_hand(action: AvalonAction, _delta: float) -> void:
	target_right_hand.global_transform.origin = Tools.get_new_hand_position(
		target_right_hand.global_transform.origin,
		action.right_hand_delta_position,
		prev_target_head_global_transform,
		target_head.global_transform,
		arm_length
	)


func do_left_hand_action(_action: AvalonAction) -> void:
	HARD.assert(false, "`do_left_hand_action` not implemented")


func do_right_hand_action(_action: AvalonAction) -> void:
	HARD.assert(false, "`do_right_hand_action` Not implemented")


func get_hands() -> Array:
	return [target_left_hand, target_right_hand]


func eat() -> void:
	for hand in get_hands():
		var thing = hand.get_held_thing()
		if (
			is_instance_valid(thing)
			and eat_area.overlaps_body(thing)
			and InteractionHandlers.is_edible(thing)
		):
			# need to be careful to remove collision shape before eating
			var energy = InteractionHandlers.attempt_eat(thing)
			physical_body.remove_collision_exception_with(thing)
			hand.set_held_thing(null)
			add_energy(energy)
			print("ate %s for %f energy" % [thing.name, energy])


func take_damage(damage: float) -> void:
	_hit_points_lost_from_enemies += damage

	if HARD.mode():
		print("dealt %s damage!" % [damage])


func add_energy(energy: float) -> void:
	_hit_points_gained_from_eating += energy


func kinetic_energy_expenditure(
	thing_mass: float, current_velocity: Vector3, prev_velocity: Vector3
) -> float:
	var average_velocity = (current_velocity + prev_velocity) / 2
	var delta_velocity = current_velocity - prev_velocity
	return clamp(0, thing_mass * average_velocity.dot(delta_velocity), INF)


func potential_energy_expenditure(thing_mass: float, delta_distance: float) -> float:
	return clamp(0, thing_mass * GRAVITY_MAGNITUDE * delta_distance, INF)


func get_body_mass_with_held_things() -> float:
	var total := mass
	for hand in get_hands():
		var thing = hand.get_held_thing()
		if is_instance_valid(thing) and not Tools.is_static(thing) and "mass" in thing:
			total += thing.mass
	return total


func _get_hand_mass_with_held_thing(hand: Node) -> float:
	var total := mass
	var thing = hand.get_held_thing()
	if is_instance_valid(thing) and not Tools.is_static(thing) and "mass" in thing:
		total += thing.mass
	return total * arm_mass_ratio


func get_left_hand_mass_with_held_thing() -> float:
	return _get_hand_mass_with_held_thing(target_left_hand)


func get_right_hand_mass_with_held_thing() -> float:
	return _get_hand_mass_with_held_thing(target_right_hand)


func _get_body_kinetic_energy_expenditure(
	thing_mass: float, current_velocity: Vector3, prev_velocity: Vector3
) -> float:
	# set the y velocity to zero since jump is separately and there should be no energy cost to falling
	current_velocity.y = 0
	prev_velocity.y = 0
	return kinetic_energy_expenditure(thing_mass, current_velocity, prev_velocity)


func _get_fall_damage() -> float:
	var fall_damage := 0.0
	var physical_body_linear_velocity = prev_physical_body_linear_velocity

	if is_on_floor():
		fall_damage += (
			fall_damage_coefficient
			* clamp(
				mass * (pow(physical_body_linear_velocity.y, 2) - pow(minimum_fall_speed, 2)),
				0,
				INF
			)
		)
	return fall_damage


func _get_all_energy_expenditures() -> Dictionary:
	# note: this must be called before `update_previous_transforms_and_velocities`!
	var expenditure = {}

	var total_mass = get_body_mass_with_held_things()
	var total_left_hand_mass = get_left_hand_mass_with_held_thing()
	var total_right_hand_mass = get_right_hand_mass_with_held_thing()
	var physical_body_linear_velocity = get_physical_body_linear_velocity()

	# TODO pushing is not accounted for yet -- a lot more energy is spent holding things!
	expenditure["physical_body_kinetic_energy_expenditure"] = (
		body_kinetic_energy_coefficient
		* _get_body_kinetic_energy_expenditure(
			total_mass, physical_body_linear_velocity, prev_physical_body_linear_velocity
		)
	)

	if is_jumping:
		expenditure["physical_body_kinetic_energy_expenditure"] += (
			body_kinetic_energy_coefficient
			* kinetic_energy_expenditure(
				total_mass,
				Vector3(0.0, get_physical_body_linear_velocity().y, 0.0),
				Vector3(0.0, prev_physical_body_linear_velocity.y, 0.0)
			)
		)

	var body_delta_position = (
		physical_body.global_transform.origin
		- prev_physical_body_global_transform.origin
	)
	if is_climbing():
		expenditure["physical_body_potential_energy_expenditure"] = (
			body_potential_energy_coefficient
			* potential_energy_expenditure(total_mass, body_delta_position.y)
		)
	else:
		expenditure["physical_body_potential_energy_expenditure"] = 0.0

	var head_delta_rotation = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_head_global_transform, physical_head.global_transform).get_euler()
	)
	var head_potential_delta_distance = head_radius * sin(head_delta_rotation.z)
	expenditure["physical_head_potential_energy_expenditure"] = (
		head_potential_energy_coefficient
		* potential_energy_expenditure(mass * head_mass_ratio, head_potential_delta_distance)
	)

	expenditure["physical_left_hand_kinetic_energy_expenditure"] = (
		left_hand_kinetic_energy_coefficient
		* kinetic_energy_expenditure(
			total_left_hand_mass,
			physical_left_hand.linear_velocity - physical_body_linear_velocity,
			prev_physical_left_hand_linear_velocity - prev_physical_body_linear_velocity
		)
	)
	var left_hand_delta_position = (
		physical_left_hand.global_transform.origin
		- prev_physical_left_hand_global_transform.origin
		- body_delta_position
	)
	expenditure["physical_left_hand_potential_energy_expenditure"] = (
		left_hand_potential_energy_coefficient
		* potential_energy_expenditure(total_left_hand_mass, left_hand_delta_position.y)
	)

	expenditure["physical_right_hand_kinetic_energy_expenditure"] = (
		right_hand_kinetic_energy_coefficient
		* kinetic_energy_expenditure(
			total_right_hand_mass,
			physical_right_hand.linear_velocity - physical_body_linear_velocity,
			prev_physical_right_hand_linear_velocity - prev_physical_body_linear_velocity
		)
	)
	var right_hand_delta_position = (
		physical_right_hand.global_transform.origin
		- prev_physical_right_hand_global_transform.origin
		- body_delta_position
	)
	expenditure["physical_right_hand_potential_energy_expenditure"] = (
		right_hand_potential_energy_coefficient
		* potential_energy_expenditure(total_right_hand_mass, right_hand_delta_position.y)
	)

	return expenditure


func _get_proprioceptive_oberservation() -> Dictionary:
	# note: this must be called before `update_previous_transforms_and_velocities`!
	var data = {}

	data["physical_body_position"] = physical_body.global_transform.origin
	data["physical_head_position"] = physical_head.global_transform.origin
	data["physical_left_hand_position"] = physical_left_hand.global_transform.origin
	data["physical_right_hand_position"] = physical_right_hand.global_transform.origin

	data["physical_body_rotation"] = Tools.vec_rad2deg(
		physical_body.global_transform.basis.get_euler()
	)
	data["physical_head_rotation"] = Tools.vec_rad2deg(
		physical_head.global_transform.basis.get_euler()
	)
	data["physical_left_hand_rotation"] = Tools.vec_rad2deg(
		physical_left_hand.global_transform.basis.get_euler()
	)
	data["physical_right_hand_rotation"] = Tools.vec_rad2deg(
		physical_right_hand.global_transform.basis.get_euler()
	)

	data["physical_body_linear_velocity"] = get_physical_body_linear_velocity()
	data["physical_head_linear_velocity"] = physical_head.linear_velocity
	data["physical_left_hand_linear_velocity"] = physical_left_hand.linear_velocity
	data["physical_right_hand_linear_velocity"] = physical_right_hand.linear_velocity

	data["physical_head_angular_velocity"] = physical_head.angular_velocity
	data["physical_left_hand_angular_velocity"] = physical_left_hand.angular_velocity
	data["physical_right_hand_angular_velocity"] = physical_right_hand.angular_velocity

	data["physical_body_delta_position"] = (
		physical_body.global_transform.origin
		- prev_physical_body_global_transform.origin
	)
	data["physical_head_delta_position"] = (
		physical_head.global_transform.origin
		- prev_physical_head_global_transform.origin
	)
	data["physical_left_hand_delta_position"] = (
		physical_left_hand.global_transform.origin
		- prev_physical_left_hand_global_transform.origin
	)
	data["physical_right_hand_delta_position"] = (
		physical_right_hand.global_transform.origin
		- prev_physical_right_hand_global_transform.origin
	)

	data["physical_head_relative_position"] = (
		physical_head.global_transform.origin
		- physical_body.global_transform.origin
	)
	data["physical_left_hand_relative_position"] = (
		physical_left_hand.global_transform.origin
		- physical_body.global_transform.origin
	)
	data["physical_right_hand_relative_position"] = (
		physical_right_hand.global_transform.origin
		- physical_body.global_transform.origin
	)
	data["physical_head_relative_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_body_global_transform, physical_head.global_transform).get_euler()
	)
	data["physical_left_hand_relative_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_body_global_transform, physical_left_hand.global_transform).get_euler()
	)
	data["physical_right_hand_relative_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_body_global_transform, physical_right_hand.global_transform).get_euler()
	)

	data["physical_body_delta_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_body_global_transform, physical_body.global_transform).get_euler()
	)
	data["physical_head_delta_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_head_global_transform, physical_head.global_transform).get_euler()
	)
	data["physical_left_hand_delta_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_left_hand_global_transform, physical_left_hand.global_transform).get_euler()
	)
	data["physical_right_hand_delta_rotation"] = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_right_hand_global_transform, physical_right_hand.global_transform).get_euler()
	)

	data["physical_body_delta_linear_velocity"] = (
		get_physical_body_linear_velocity()
		- prev_physical_body_linear_velocity
	)
	data["physical_head_delta_linear_velocity"] = (
		physical_head.linear_velocity
		- prev_physical_head_linear_velocity
	)
	data["physical_left_hand_delta_linear_velocity"] = (
		physical_left_hand.linear_velocity
		- prev_physical_left_hand_linear_velocity
	)
	data["physical_right_hand_delta_linear_velocity"] = (
		physical_right_hand.linear_velocity
		- prev_physical_right_hand_linear_velocity
	)

	data["physical_head_delta_angular_velocity"] = (
		physical_head.angular_velocity
		- prev_physical_head_angular_velocity
	)
	data["physical_left_hand_delta_angular_velocity"] = (
		physical_left_hand.angular_velocity
		- prev_physical_left_hand_angular_velocity
	)
	data["physical_right_hand_delta_angular_velocity"] = (
		physical_right_hand.angular_velocity
		- prev_physical_right_hand_angular_velocity
	)

	data["left_hand_thing_colliding_with_hand"] = target_left_hand.get_id_of_thing_colliding_with_hand()
	data["left_hand_held_thing"] = target_left_hand.get_id_of_held_thing()
	data["right_hand_thing_colliding_with_hand"] = target_right_hand.get_id_of_thing_colliding_with_hand()
	data["right_hand_held_thing"] = target_right_hand.get_id_of_held_thing()

	var total_energy_expenditure := 0.0
	var expenditure = _get_all_energy_expenditures()
	for key in expenditure:
		data[key] = expenditure[key]
		total_energy_expenditure += expenditure[key]
	total_energy_expenditure *= total_energy_coefficient
	data["total_energy_expenditure"] = total_energy_expenditure
	data["fall_damage"] = _get_fall_damage()

	return data


func update_previous_transforms_and_velocities(is_reset_for_spawn = false) -> void:
	prev_target_head_global_transform = target_head.global_transform
	prev_target_left_hand_global_transform = target_left_hand.global_transform
	prev_target_right_hand_global_transform = target_right_hand.global_transform

	if is_reset_for_spawn:
		prev_physical_body_linear_velocity = Vector3.ZERO
	else:
		prev_physical_body_linear_velocity = get_physical_body_linear_velocity()

	prev_physical_body_global_transform = physical_body.global_transform
	prev_physical_head_global_transform = physical_head.global_transform
	prev_physical_left_hand_global_transform = physical_left_hand.global_transform
	prev_physical_right_hand_global_transform = physical_right_hand.global_transform

	prev_physical_head_linear_velocity = physical_head.linear_velocity
	prev_physical_left_hand_linear_velocity = physical_left_hand.linear_velocity
	prev_physical_right_hand_linear_velocity = physical_right_hand.linear_velocity

	prev_physical_head_angular_velocity = physical_head.angular_velocity
	prev_physical_left_hand_angular_velocity = physical_left_hand.angular_velocity
	prev_physical_right_hand_angular_velocity = physical_right_hand.angular_velocity


# TODO make this safer
# NOTE: this must be called after a physics tick (so at the start of another tick)
func get_oberservation_and_reward() -> Dictionary:
	# we won't know what damage we've taken until after a physics tick
	current_observation["hit_points_lost_from_enemies"] = _hit_points_lost_from_enemies
	current_observation["hit_points_gained_from_eating"] = _hit_points_gained_from_eating
	var reward = (
		_hit_points_gained_from_eating
		- (
			_hit_points_lost_from_enemies
			+ current_observation["total_energy_expenditure"]
			+ current_observation["fall_damage"]
		)
	)
	_hit_points_lost_from_enemies = 0.0
	_hit_points_gained_from_eating = 0.0

	# prevent huge negative rewards
	reward = max(reward, -hit_points)

	# TODO don't love that this happens here
	# update hit points now
	hit_points += reward
	current_observation["reward"] = reward
	current_observation["hit_points"] = hit_points

	if hit_points <= 0:
		# TODO handle death, need to end the current rollout
		is_dead = true

	current_observation["is_dead"] = int(is_dead)

	return current_observation


func get_head_delta_quaternion() -> Quat:
	return Tools.get_delta_quaternion(
		prev_target_head_global_transform, target_head.global_transform
	)


func get_velocity_change_due_to_gravity() -> Vector3:
	return Vector3.DOWN * GRAVITY_MAGNITUDE * 1 / PHYSICS_FPS


func get_actual_delta_position_after_move() -> Vector3:
	# NOTE: use this instead of velocity returned by `move_and_slide`
	return physical_body.global_transform.origin - prev_physical_body_global_transform.origin


func get_current_position() -> Vector3:
	return physical_body.global_transform.origin


func get_physical_body_linear_velocity() -> Vector3:
	return get_actual_delta_position_after_move() * PHYSICS_FPS


func get_jump_velocity() -> Vector3:
	var t = sqrt(2 * jump_height / GRAVITY_MAGNITUDE)
	# correcting for acceleration being applied discretely
	return GRAVITY_MAGNITUDE * t * Vector3.UP - get_velocity_change_due_to_gravity() / 2


func get_throw_factor(object_mass: float) -> float:
	if object_mass < mass * arm_mass_ratio * throw_force_magnitude:
		return 1.0
	else:
		return mass * arm_mass_ratio * throw_force_magnitude / object_mass


func get_floor_collision() -> KinematicCollision:
	for i in range(physical_body.get_slide_count()):
		var collision = physical_body.get_slide_collision(i)
		if collision.normal.angle_to(UP_DIRECTION) <= FLOOR_MAX_ANGLE:
			return collision
	return null


func get_wall_collisions() -> Array:
	var collisions = []
	for i in range(physical_body.get_slide_count()):
		var collision = physical_body.get_slide_collision(i)
		if collision.normal.angle_to(UP_DIRECTION) > FLOOR_MAX_ANGLE:
			collisions.append(collision)
	return collisions


func is_on_floor() -> bool:
	return physical_body.is_on_floor()


func is_able_to_wall_jump() -> bool:
	var y_vel = abs(get_physical_body_linear_velocity().y)

	var no_recent_jumps := true
	for value in y_vel_history:
		if value > 1.0:
			no_recent_jumps = false
			break
	y_vel_history.append(y_vel)
	if len(y_vel_history) > 15:
		y_vel_history.pop_front()
	else:
		# still waiting to build up enough history
		no_recent_jumps = false

	var result = physical_body.is_on_wall() and y_vel < 1.0 and no_recent_jumps
	return result


func get_hand_meshes(hand: PlayerHand) -> Array:
	return [
		hand.get_node("hand/active"), hand.get_node("hand/disabled"), hand.get_node("hand/default")
	]


func set_hand_mesh_visibility(hand: PlayerHand):
	var thing_colliding_with_hand = hand.get_thing_colliding_with_hand(false)
	var currently_held_thing = hand.get_held_thing()

	var hand_meshes = get_hand_meshes(hand)
	var active = hand_meshes[0]
	var disabled = hand_meshes[1]
	var default = hand_meshes[2]

	if (
		Tools.is_terrain(thing_colliding_with_hand)
		and not InteractionHandlers.is_position_climbable(get_tree(), hand.global_transform.origin)
	):
		active.visible = false
		disabled.visible = true
		default.visible = false
	else:
		# TODO another hack we should remove, if we for sure know something is unclimbable test
		# using the extra climbing ray if we can grab something
		thing_colliding_with_hand = hand.get_thing_colliding_with_hand(true)
		if currently_held_thing or thing_colliding_with_hand:
			active.visible = true
			disabled.visible = false
			default.visible = false
		else:
			active.visible = false
			disabled.visible = false
			default.visible = true
