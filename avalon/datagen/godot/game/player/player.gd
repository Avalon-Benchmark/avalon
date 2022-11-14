extends ControlledNode

class_name Player

# player nodes
var target_player: Spatial
var target_head: Spatial
var target_left_hand: Spatial
var target_right_hand: Spatial
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
var eyes: Camera

# NOTE: these must be present in `reset_on_new_world`!
# player state
var current_observation := {}
export var hit_points: float
export var gravity_velocity := Vector3.ZERO
export var surface_normal := Vector3.UP
export var is_jumping := false
export var _hit_points_gained_from_eating: float = 0.0
export var _hit_points_lost_from_enemies: float = 0.0
export var is_dead := false
export var floor_y_vel_history: Array
export var falling_y_vel_history: Array
export var wall_grab_history: Array
# see `_get_fall_damage` for more explanation on what these constants are used for
export var frames_after_taking_fall_damage := 0
export var frames_after_reaching_fall_damage_speeds := 0

# player configuration
var height := 2.0
var head_radius := 1.0  # from average human head circumference (57cm)
var starting_left_hand_position_relative_to_head := Vector3(-0.5, -0.5, -0.5)
var starting_right_hand_position_relative_to_head := Vector3(0.5, -0.5, -0.5)

var spec: PlayerSpec
# TODO: sigh ... this is very error prone how these get set
export var max_head_linear_speed: float
export var max_head_angular_speed: Vector3
export var max_hand_linear_speed: float
export var max_hand_angular_speed: float
export var jump_height: float
export var arm_length: float
export var starting_hit_points: float
export var mass: float
export var arm_mass_ratio: float  # from https://exrx.net/Kinesiology/Segments
export var head_mass_ratio: float  # from https://exrx.net/Kinesiology/Segments
export var standup_speed_after_climbing: float
export var min_head_position_off_of_floor: float
export var push_force_magnitude: float
export var throw_force_magnitude: float
export var minimum_fall_speed: float
export var fall_damage_coefficient: float
export var num_frames_alive_after_food_is_gone: float
export var eat_area_radius: float
export var is_displaying_debug_meshes: bool
export var is_human_playback_enabled: bool
export var is_slowed_from_crouching: bool

# `move_and_slide` constants, we should keep this fixed because they can change the actual movement distance
const UP_DIRECTION := Vector3.UP
const STOP_ON_SLOPE := true
const MAX_SLIDES := 4  # DO NOT CHANGE THIS UNLESS YOU KNOW WHAT YOU'RE DOING
const FLOOR_MAX_ANGLE := deg2rad(45)
const INFINITE_INERTIA := false
var PHYSICS_FPS: float = ProjectSettings.get_setting("physics/common/physics_fps")
var GRAVITY_MAGNITUDE: float = ProjectSettings.get_setting("physics/3d/default_gravity")
# TODO move into config
var extra_height_margin := 0.0

# see `_get_fall_damage` for more explanation on what these constants are used for
const _FALL_DAMAGE_WINDOW = 4
const _FALL_DAMAGE_WAIT_PERIOD = 2
const _FALL_DAMAGE_DEBOUNCE_PERIOD = 4

# storing previous state of the body
export var prev_hit_points: float
export var prev_target_head_global_transform: Transform
export var prev_target_left_hand_global_transform: Transform
export var prev_target_right_hand_global_transform: Transform
export var prev_physical_body_global_transform: Transform
export var prev_physical_head_global_transform: Transform
export var prev_physical_left_hand_global_transform: Transform
export var prev_physical_right_hand_global_transform: Transform
export var prev_physical_body_linear_velocity: Vector3
export var prev_physical_head_linear_velocity: Vector3
export var prev_physical_left_hand_linear_velocity: Vector3
export var prev_physical_right_hand_linear_velocity: Vector3
export var prev_physical_head_angular_velocity: Vector3
export var prev_physical_left_hand_angular_velocity: Vector3
export var prev_physical_right_hand_angular_velocity: Vector3

export var _is_state_initialized = false

var PERF_SIMPLE_AGENT: bool = ProjectSettings.get_setting("avalon/simple_agent")
var PERF_ACTION_APPLY: bool = ProjectSettings.get_setting("avalon/action_apply")


func _ready() -> void:
	if not _is_state_initialized:
		_reset_histories()
		hit_points = starting_hit_points

	validate_configuration()
	set_nodes_in_ready()

	if not _is_state_initialized:
		update_previous_transforms_and_velocities(true)

	# reset current observation, happens after `update_previous_transforms_and_velocities` so everthing starts at 0
	set_current_observation()
	_is_state_initialized = true


func validate_configuration():
	HARD.assert(max_head_linear_speed != null, "max_head_linear_speed is not defined")
	HARD.assert(max_head_angular_speed != null, "max_head_angular_speed is not defined")
	HARD.assert(max_hand_linear_speed != null, "max_hand_linear_speed is not defined")
	HARD.assert(max_hand_angular_speed != null, "max_hand_angular_speed is not defined")
	HARD.assert(jump_height != null, "jump_height is not defined")
	HARD.assert(arm_length != null, "arm_length is not defined")
	HARD.assert(starting_hit_points != null, "starting_hit_points is not defined")
	HARD.assert(mass != null, "mass is not defined")
	HARD.assert(arm_mass_ratio != null, "arm_mass_ratio is not defined")
	HARD.assert(head_mass_ratio != null, "head_mass_ratio is not defined")
	HARD.assert(standup_speed_after_climbing != null, "standup_speed_after_climbing is not defined")
	HARD.assert(
		min_head_position_off_of_floor != null, "min_head_position_off_of_floor is not defined"
	)
	HARD.assert(push_force_magnitude != null, "push_force_magnitude is not defined")
	HARD.assert(throw_force_magnitude != null, "throw_force_magnitude is not defined")
	HARD.assert(minimum_fall_speed != null, "minimum_fall_speed is not defined")
	HARD.assert(fall_damage_coefficient != null, "fall_damage_coefficient is not defined")
	HARD.assert(
		num_frames_alive_after_food_is_gone != null,
		"num_frames_alive_after_food_is_gone is not defined"
	)
	HARD.assert(eat_area_radius != null, "eat_area_radius is not defined")
	HARD.assert(is_displaying_debug_meshes != null, "is_displaying_debug_meshes is not defined")
	HARD.assert(is_human_playback_enabled != null, "is_human_playback_enabled is not defined")
	HARD.assert(is_slowed_from_crouching != null, "is_slowed_from_crouching is not defined")


func set_nodes_in_ready():
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

	eat_area = physical_head.get_node("eat_area")

	climbing_ray.add_exception(physical_body)
	climbing_ray.add_exception(physical_left_hand)
	climbing_ray.add_exception(physical_right_hand)
	ground_ray.add_exception(physical_head)
	ground_ray.add_exception(physical_body)
	ground_ray.add_exception(physical_left_hand)
	ground_ray.add_exception(physical_right_hand)

	if PERF_SIMPLE_AGENT:
		climbing_ray.enabled = false
		ground_ray.enabled = false
		eat_area.monitoring = false
		target_left_hand.visible = false
		target_right_hand.visible = false

	if not _is_state_initialized:
		target_left_hand.global_transform.origin = (
			starting_left_hand_position_relative_to_head
			+ target_head.global_transform.origin
		)
		target_right_hand.global_transform.origin = (
			starting_right_hand_position_relative_to_head
			+ target_head.global_transform.origin
		)

	eyes = _get_eyes()

	# make the eat area comically large so we can test bringing food into our face
	var shape: SphereShape = eat_area.get_child(0).shape
	shape.radius = eat_area_radius

	_validate_arm_lengths()
	_set_debug_mesh_visibility()


func _validate_arm_lengths():
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


func _set_debug_mesh_visibility() -> void:
	physical_head.get_node("beak").visible = is_displaying_debug_meshes
	physical_head.get_node("face").visible = is_displaying_debug_meshes
	physical_left_hand.get_node("marker").visible = is_displaying_debug_meshes
	physical_right_hand.get_node("marker").visible = is_displaying_debug_meshes


func _get_eyes() -> Node:
	return get_node("camera")


func _reset_histories() -> void:
	floor_y_vel_history = []
	falling_y_vel_history = []
	wall_grab_history = []


func reset_on_new_world() -> void:
	# reset any internal state when moving to a new world
	_reset_histories()
	hit_points = starting_hit_points
	gravity_velocity = Vector3.ZERO
	surface_normal = Vector3.UP
	is_jumping = false
	current_observation = {}
	_hit_points_gained_from_eating = 0.0
	_hit_points_lost_from_enemies = 0.0
	is_dead = false
	frames_after_taking_fall_damage = 0
	frames_after_reaching_fall_damage_speeds = 0

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
	current_observation = _get_proprioceptive_observation()


func is_warming_up(_delta: float, _frame: int) -> bool:
	return false


func fix_tracking_once_working() -> bool:
	return false


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


func apply_action(player_action: PlayerAction, delta: float):
	if not player_action is PlayerAction:
		HARD.stop("Player.apply_action expected PlayerAction but received %s" % player_action)

	var action = player_action.scaled

	if not PERF_ACTION_APPLY:
		return
	# head collision shape is actually part of your body -- make sure it get's moved so climbing will work
	update_head_collision_shape()

	# BEWARE: you must perform the following actions in this exact order or else things will not work
	rotate_head(action, delta)
	if not PERF_SIMPLE_AGENT:
		rotate_left_hand(action, delta)
		rotate_right_hand(action, delta)

	# we MUST move our body before moving our hands, it determines how far our hands can actually move
	move(action, delta)

	if not PERF_SIMPLE_AGENT:
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
		(target_head.transform.origin - physical_head.transform.origin)
		/ delta
	)
	physical_head.angular_velocity = (
		Tools.calc_angular_velocity_from_basis(
			physical_head.transform.basis, target_head.transform.basis
		)
		/ delta
	)

	if PERF_SIMPLE_AGENT:
		return

	physical_left_hand.linear_velocity = (
		(target_left_hand.transform.origin - physical_left_hand.transform.origin)
		/ delta
	)
	physical_left_hand.angular_velocity = (
		Tools.calc_angular_velocity_from_basis(
			physical_left_hand.transform.basis, target_left_hand.transform.basis
		)
		/ delta
	)

	physical_right_hand.linear_velocity = (
		(target_right_hand.transform.origin - physical_right_hand.transform.origin)
		/ delta
	)
	physical_right_hand.angular_velocity = (
		Tools.calc_angular_velocity_from_basis(
			physical_right_hand.transform.basis, target_right_hand.transform.basis
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


func _accumulate_falling_y_velocity_history():
	# NOTE: this must happen before we try and jump or record fall damage
	var y_vel = abs(get_physical_body_linear_velocity().y)
	falling_y_vel_history.push_front(y_vel)
	if len(falling_y_vel_history) > 15:
		falling_y_vel_history.pop_back()


func was_falling() -> bool:
	var delta_y_speed = _get_delta_y_speed_over_window()
	return frames_after_taking_fall_damage > 0 or delta_y_speed > minimum_fall_speed * 0.75


func jump(action: AvalonAction, _delta: float) -> void:
	# prevent jumping when recently falling
	if was_falling():
		return

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

	var velocity = (
		Tools.get_move_delta_position(target_head.global_transform.basis, head_delta_position)
		/ delta
	)

	if not is_slowed_from_crouching:
		return velocity

	var crouch_speed_ratio = clamp(
		(get_curr_head_distance_from_feet() + min_head_position_off_of_floor) / 2.0,
		min_head_position_off_of_floor,
		1.0
	)
	return velocity * crouch_speed_ratio


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


func get_curr_head_distance_from_feet() -> float:
	var global_feet_position = physical_body.global_transform.origin.y - height / 2
	return target_head.global_transform.origin.y - global_feet_position


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
	var _collision = physical_body.move_and_collide(Vector3.UP * standup_speed_after_climbing)


func is_climbing():
	if PERF_SIMPLE_AGENT:
		return false
	else:
		return (
			target_left_hand.is_grasping_heavy_thing()
			or target_right_hand.is_grasping_heavy_thing()
		)


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

		_accumulate_falling_y_velocity_history()

		gravity_velocity = Vector3.ZERO
	else:
		# use `collision_body` while walking around so you know when you're touching the ground
		collision_body.disabled = false
		collision_head.disabled = true

		move_while_not_climbing(action, delta)

		_accumulate_falling_y_velocity_history()

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


func _get_body_kinetic_energy_expenditure(
	thing_mass: float, current_velocity: Vector3, prev_velocity: Vector3
) -> float:
	# set the y velocity to zero since jump is separately and there should be no energy cost to falling
	current_velocity.y = 0
	prev_velocity.y = 0
	return kinetic_energy_expenditure(thing_mass, current_velocity, prev_velocity)


func _get_delta_y_speed_over_window() -> float:
	var max_y_speed = falling_y_vel_history.slice(0, _FALL_DAMAGE_WINDOW).max()
	var min_y_speed = falling_y_vel_history.slice(0, _FALL_DAMAGE_WINDOW).min()
	return max_y_speed - min_y_speed


func _get_fall_damage() -> float:
	# NOTE: Falling is physics FPS dependent. What happens currently is that it takes several frames after hitting the
	# 	ground to actually come to rest. Instead of looking at the instanteous difference in speeds, we look at a the
	#	past _FALL_DAMAGE_WINDOW speeds and compare the smallest and largest speeds in that window to determine
	#	the correct change of velocity.

	var fall_damage := 0.0

	if len(falling_y_vel_history) <= _FALL_DAMAGE_WINDOW:
		return fall_damage

	var delta_y_speed = _get_delta_y_speed_over_window()
	var has_taken_fall_damage_previously = frames_after_taking_fall_damage > 0
	var is_moving_fast_enough_to_take_damage = delta_y_speed > minimum_fall_speed * 0.75

	# wait _FALL_DAMAGE_WAIT_PERIOD frames after a sufficiently sized change in velocity to see
	# 	if we come to rest
	if (
		not has_taken_fall_damage_previously
		and is_moving_fast_enough_to_take_damage
		and frames_after_reaching_fall_damage_speeds < _FALL_DAMAGE_WAIT_PERIOD
	):
		frames_after_reaching_fall_damage_speeds += 1
		return fall_damage

	# after waiting, calculate fall damage of the using the greatest difference of speeds in the past
	# 	_FALL_DAMAGE_WINDOW frames
	if not has_taken_fall_damage_previously and is_moving_fast_enough_to_take_damage:
		fall_damage = (
			fall_damage_coefficient
			* clamp(mass * (pow(delta_y_speed, 2) - pow(minimum_fall_speed, 2)), 0, INF)
		)
		frames_after_reaching_fall_damage_speeds = 0

		# if we actually take fall damage, wait _FALL_DAMAGE_DEBOUNCE_PERIOD frames before we
		#	can take damage again
		if fall_damage > 0.0:
			frames_after_taking_fall_damage = _FALL_DAMAGE_DEBOUNCE_PERIOD
			return fall_damage

	if has_taken_fall_damage_previously:
		frames_after_taking_fall_damage -= 1

	return fall_damage


func _get_all_energy_expenditures() -> Dictionary:
	if PERF_SIMPLE_AGENT:
		return {}

	# note: this must be called before `update_previous_transforms_and_velocities`!
	var expenditure = {}

	var total_mass = mass
	var total_left_hand_mass = mass * arm_mass_ratio
	var total_right_hand_mass = mass * arm_mass_ratio
	var physical_body_linear_velocity = get_physical_body_linear_velocity()

	# TODO pushing is not accounted for yet
	expenditure["physical_body_kinetic_energy_expenditure"] = (_get_body_kinetic_energy_expenditure(
		total_mass, physical_body_linear_velocity, prev_physical_body_linear_velocity
	))

	if is_jumping:
		expenditure["physical_body_kinetic_energy_expenditure"] += (kinetic_energy_expenditure(
			total_mass,
			Vector3(0.0, get_physical_body_linear_velocity().y, 0.0),
			Vector3(0.0, prev_physical_body_linear_velocity.y, 0.0)
		))

	var body_delta_position = (
		physical_body.global_transform.origin
		- prev_physical_body_global_transform.origin
	)
	if is_climbing():
		expenditure["physical_body_potential_energy_expenditure"] = (potential_energy_expenditure(
			total_mass, body_delta_position.y
		))
	else:
		expenditure["physical_body_potential_energy_expenditure"] = 0.0

	var head_delta_rotation = Tools.vec_rad2deg(
		Tools.get_delta_quaternion(prev_physical_head_global_transform, physical_head.global_transform).get_euler()
	)
	var head_potential_delta_distance = head_radius * sin(head_delta_rotation.z)
	expenditure["physical_head_potential_energy_expenditure"] = (potential_energy_expenditure(
		mass * head_mass_ratio, head_potential_delta_distance
	))

	expenditure["physical_left_hand_kinetic_energy_expenditure"] = (kinetic_energy_expenditure(
		total_left_hand_mass,
		physical_left_hand.linear_velocity - physical_body_linear_velocity,
		prev_physical_left_hand_linear_velocity - prev_physical_body_linear_velocity
	))
	var left_hand_delta_position = (
		physical_left_hand.global_transform.origin
		- prev_physical_left_hand_global_transform.origin
		- body_delta_position
	)
	expenditure["physical_left_hand_potential_energy_expenditure"] = (potential_energy_expenditure(
		total_left_hand_mass, left_hand_delta_position.y
	))

	expenditure["physical_right_hand_kinetic_energy_expenditure"] = (kinetic_energy_expenditure(
		total_right_hand_mass,
		physical_right_hand.linear_velocity - physical_body_linear_velocity,
		prev_physical_right_hand_linear_velocity - prev_physical_body_linear_velocity
	))
	var right_hand_delta_position = (
		physical_right_hand.global_transform.origin
		- prev_physical_right_hand_global_transform.origin
		- body_delta_position
	)
	expenditure["physical_right_hand_potential_energy_expenditure"] = (potential_energy_expenditure(
		total_right_hand_mass, right_hand_delta_position.y
	))

	return expenditure


func _get_proprioceptive_observation() -> Dictionary:
	if PERF_SIMPLE_AGENT:
		return {
			"physical_body": physical_body.transform,
			"physical_head": physical_head.transform,
		}

	# note: this must be called before `update_previous_transforms_and_velocities`!
	var data = {}

	data["target_head_position"] = target_head.global_transform.origin
	data["target_left_hand_position"] = target_left_hand.global_transform.origin
	data["target_right_hand_position"] = target_right_hand.global_transform.origin

	data["target_head_rotation"] = Tools.vec_rad2deg(target_head.global_transform.basis.get_euler())
	data["target_left_hand_rotation"] = Tools.vec_rad2deg(
		target_left_hand.global_transform.basis.get_euler()
	)
	data["target_right_hand_rotation"] = Tools.vec_rad2deg(
		target_right_hand.global_transform.basis.get_euler()
	)

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

	var expenditure = _get_all_energy_expenditures()
	for key in expenditure:
		data[key] = expenditure[key]
	data["fall_damage"] = _get_fall_damage()

	return data


func update_previous_transforms_and_velocities(is_reset_for_spawn = false) -> void:
	prev_target_head_global_transform = target_head.global_transform

	if is_reset_for_spawn:
		prev_physical_body_linear_velocity = Vector3.ZERO
	else:
		prev_physical_body_linear_velocity = get_physical_body_linear_velocity()

	prev_physical_body_global_transform = physical_body.global_transform
	prev_physical_head_global_transform = physical_head.global_transform
	prev_physical_head_linear_velocity = physical_head.linear_velocity
	prev_physical_head_angular_velocity = physical_head.angular_velocity

	if PERF_SIMPLE_AGENT:
		return

	prev_target_left_hand_global_transform = target_left_hand.global_transform
	prev_target_right_hand_global_transform = target_right_hand.global_transform

	prev_physical_left_hand_global_transform = physical_left_hand.global_transform
	prev_physical_right_hand_global_transform = physical_right_hand.global_transform

	prev_physical_left_hand_linear_velocity = physical_left_hand.linear_velocity
	prev_physical_right_hand_linear_velocity = physical_right_hand.linear_velocity

	prev_physical_left_hand_angular_velocity = physical_left_hand.angular_velocity
	prev_physical_right_hand_angular_velocity = physical_right_hand.angular_velocity


# NOTE: this must be called after a physics tick (so at the start of another tick)
func get_observation_and_reward() -> Dictionary:
	if PERF_SIMPLE_AGENT:
		return current_observation
	# we won't know what damage we've taken until after a physics tick
	current_observation["hit_points_lost_from_enemies"] = _hit_points_lost_from_enemies
	current_observation["hit_points_gained_from_eating"] = _hit_points_gained_from_eating
	var reward = (
		_hit_points_gained_from_eating
		- (_hit_points_lost_from_enemies + current_observation["fall_damage"])
	)
	_hit_points_lost_from_enemies = 0.0
	_hit_points_gained_from_eating = 0.0

	# prevent huge negative rewards
	reward = max(reward, -hit_points)

	# update hit points now
	hit_points += reward
	current_observation["reward"] = reward
	current_observation["hit_points"] = hit_points

	if hit_points <= 0:
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
	for value in floor_y_vel_history:
		if value > 1.0:
			no_recent_jumps = false
			break
	floor_y_vel_history.append(y_vel)
	if len(floor_y_vel_history) > 15:
		floor_y_vel_history.pop_front()
	else:
		# still waiting to build up enough history
		no_recent_jumps = false

	var recently_grabbed_wall = false
	for value in wall_grab_history:
		if value:
			recently_grabbed_wall = true
			break
	var is_grabbing_wall = is_climbing()
	wall_grab_history.append(is_grabbing_wall)
	if len(wall_grab_history) > 15:
		wall_grab_history.pop_front()

	var result = (
		physical_body.is_on_wall()
		and y_vel < 1.0
		and no_recent_jumps
		and not recently_grabbed_wall
	)
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
