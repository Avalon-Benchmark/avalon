extends Item

class_name Animal

enum { AWAY_FROM_PLAYER = -1, TOWARDS_PLAYER = 1, TOWARDS_TERRITORY = 2 }

const NEVER = "NEVER"
const DEAD = "DEAD"
const ALWAYS = "ALWAYS"

const DAMAGE_INVINCIBILITY_FRAMES = 10
const INVINCIBILITY_INDICATOR_FRAMES = 5

const INVINCIBLE_HP = 100

# re-export
var GROUND = MovementController.GROUND
var CLIMB = MovementController.CLIMB
var AIR = MovementController.AIR

export var energy := 1.0
export var edible_when := DEAD
export var grabbable_when := ALWAYS
export var hit_points := 1
export var primary_domain: String
export var is_able_to_climb: bool

export var weapon_contact_count := 0
export var observed_velocity := Vector3.ZERO
export var observed_movement_delta := Vector3.ZERO

export var fall_damage_velocity_threshold := 40.0

export var freeze_at_distance_from_player := 50
export var player_detection_radius := 12.0

var is_player_in_detection_radius := false

var invincibility_frame := 0

var controller: MovementController
var _collision_shape: CollisionShape

var _player: Node

var inactive_behavior
var active_behavior
var previous_behavior

var avoid_ocean_behavior

var is_frozen: bool = false

var possible_impediments := []


func _ready():
	controller = $movement_controller
	_collision_shape = $collision_shape
	add_collision_exception_with(controller)
	if controller.current_domain != CLIMB:
		controller.current_domain = primary_domain

	if is_alive():
		mobilize()
	if edible_when != NEVER:
		add_to_group("food")


func select_next_behavior():
	if is_player_in_detection_radius:
		return active_behavior
	return inactive_behavior


func _safely_select_next_behavior():
	var behavior = select_next_behavior()
	if is_avoiding_ocean():
		behavior = avoid_ocean_behavior
	return behavior


func _switch_behavior(behavior):
	if HARD.mode():
		var previous = previous_behavior.get_name() if previous_behavior else ""
		var new = behavior.get_name()
		print("behavior change in %s: %s -> %s" % [name, previous, new])
	if previous_behavior != null:
		previous_behavior.reset()
	previous_behavior = behavior


func maybe_freeze(player_distance: float) -> bool:
	var should_freeze = player_distance >= freeze_at_distance_from_player
	if should_freeze == is_frozen:
		return should_freeze

	if should_freeze:
		controller.get_floor_ray().enabled = false
	elif is_alive():
		controller.get_floor_ray().enabled = true

	return should_freeze


func _physics_process(delta):
	var player_distance = distance_to_player(is_flying())
	is_player_in_detection_radius = player_distance <= player_detection_radius
	if is_behaving_like_item or maybe_freeze(player_distance):
		return

	handle_damage_debounce()

	var behavior = _safely_select_next_behavior()

	if behavior != previous_behavior:
		_switch_behavior(behavior)

	var pos = global_transform.origin
	observed_velocity = behavior.do(self, delta)
	global_transform = controller.align_and_position_parent(global_transform)

	observed_movement_delta = global_transform.origin - pos

	controller.cast_rays(is_flying())
	controller.correct_gravity(observed_movement_delta, ground_contact_count > 0)

	# in order to actually detect collsions from our own movement while in kinematic mode,
	# we need to resolve contact changes ourselves.
	# We also need to avoid double-calling _on_body_* due to other rigid body colliding with us
	var contact_changes = controller.get_contact_changes()
	for exited in contact_changes["exited"]:
		on_body_exited(exited)
	for entered in contact_changes["entered"]:
		on_body_entered(entered)


func _integrate_forces(state):
	var is_likely_to_fall_through_floor_due_to_drop = (
		not is_behaving_like_item
		and state.linear_velocity.length() > 0
	)
	if is_likely_to_fall_through_floor_due_to_drop:
		state.linear_velocity = Vector3.ZERO


func get_player_position() -> Vector3:
	if _player == null:
		_player = get_node("/root/Globals").player.physical_body
	return _player.global_transform.origin


func distance_to_player(is_in_2d: bool = false) -> float:
	var player_pos = get_player_position()
	var animal_pos = global_transform.origin
	if is_in_2d:
		player_pos.y = animal_pos.y
	return animal_pos.distance_to(player_pos)


func is_alive() -> bool:
	return hit_points > 0


func is_edible() -> bool:
	return _is_in_state(edible_when)


func is_grabbable() -> bool:
	return _is_in_state(grabbable_when)


func is_throwable() -> bool:
	return true


func _is_in_state(when_state: String) -> bool:
	match when_state:
		NEVER:
			return false
		DEAD:
			return not is_alive()
		ALWAYS:
			return true
		_:
			HARD.assert(
				"when_state %s should be an animal state (%s)" % [when_state, [NEVER, DEAD, ALWAYS]]
			)
			return false


func is_held() -> bool:
	return InteractionHandlers.has_joint(self)


func grab(_hand_tracker: RigidBody):
	if is_alive():
		immobilize()
	return self


func release():
	if is_alive() and (is_flying() or ground_contact_count > 0):
		mobilize()


func eat() -> float:
	if is_alive():
		hit_points -= 1
		if not is_alive():
			_die()
			return 0.0
	hide()
	return energy


# NOTE we can't disable the main collision shape b/c we need it for signals
func immobilize():
	controller.disable()
	mode = MODE_RIGID
	behave_like_item()


func mobilize():
	mode = MODE_KINEMATIC
	controller.enable()
	is_behaving_like_item = false


func get_local_forward() -> Vector3:
	# -Z is the direction looking_at uses (IIRC)
	return -global_transform.basis.z


func get_first_forwad_impediment():
	var is_extent_included = true
	for pi in possible_impediments:
		var is_impediment = is_point_in_front_of(pi.global_transform.origin, is_extent_included)
		if is_impediment:
			return pi


func possibly_change_to_sidestep(default_dir: Vector3) -> Vector3:
	var impediment = get_first_forwad_impediment()
	if impediment == null:
		return default_dir

	var impediment_pos = impediment.global_transform.origin
	var sidestep_dir = global_transform.basis.x
	var is_left_faster = is_point_right_of(impediment_pos)
	if is_left_faster:
		sidestep_dir = -sidestep_dir

	if HARD.mode():
		var l_or_r = "left" if is_left_faster else "right"
		print("%s sidestepping %s to the %s" % [name, impediment, l_or_r])

	return sidestep_dir


func forward_hop(hop_speed: Vector2, is_sidestep_ok: bool = true) -> Vector3:
	var dir = get_local_forward()
	if is_sidestep_ok:
		dir = possibly_change_to_sidestep(dir)
	var step = dir * hop_speed.x
	step.y = hop_speed.y
	return step


func get_movement_direction_towards(target: Vector3, is_sidestep_ok: bool = false) -> Vector3:
	var pos = global_transform.origin
	var dir = pos.direction_to(target)
	if is_sidestep_ok:
		dir = possibly_change_to_sidestep(dir)

	if is_grounded():
		dir.y = 0
	return dir.normalized()


func _die(is_collision_on_right: bool = false):
	if HARD.mode():
		print("killed %s!" % name)

	var flip = deg2rad(90)
	if is_collision_on_right:
		flip *= 1

	if has_node("wing_collision_shape"):
		$wing_collision_shape.disabled = false
		# flip entirely over
		rotate_x(flip * 2)
	else:
		rotate_x(flip)

	immobilize()
	show_pain_or_dead()
	hit_points = 0


func show_pain_or_dead():
	$dead_mesh.show()
	$alive_mesh.hide()


func hide_pain_or_dead():
	$alive_mesh.show()
	$dead_mesh.hide()


func is_temporariy_invisible():
	return invincibility_frame > 0


func handle_damage_debounce():
	if invincibility_frame == 0:
		return

	invincibility_frame = (invincibility_frame + 1) % DAMAGE_INVINCIBILITY_FRAMES

	if $dead_mesh.visible and invincibility_frame >= INVINCIBILITY_INDICATOR_FRAMES:
		hide_pain_or_dead()


func toggle_pain_indicator():
	if $dead_mesh.visible:
		hide_pain_or_dead()
	else:
		show_pain_or_dead()


func _on_body_entered(body: Node):
	if mode == MODE_KINEMATIC:
		return
	return on_body_entered(body)


func on_body_entered(body: Node):
	._on_body_entered(body)

	if not is_alive():
		return

	if "tree" in body.name:
		if HARD.mode():
			print("%s has possible impediment %s" % [name, body])
		possible_impediments.append(body)

	if hit_points >= INVINCIBLE_HP:
		return

	_calculate_damage_and_death(body)

	if is_alive() and not is_held() and is_behaving_like_item and ground_contact_count > 0:
		mobilize()


func _calculate_damage_and_death(body: Node):
	if is_temporariy_invisible():
		return

	var impact_velocity = linear_velocity if is_behaving_like_item else observed_velocity
	var damage = 0

	if body is Weapon:
		weapon_contact_count += 1
		damage = floor(body.get_inflicted_damage(impact_velocity))
		if HARD.mode():
			var speed = body.linear_velocity.length()
			print("%s taking %s damage from %s at velocity %s" % [self, damage, body, speed])

	elif Tools.is_terrain(body):
		var is_first_collision = ground_contact_count == 1
		if is_held() or not is_first_collision:
			return
		# fall dammage is just y for now
		var impact = -impact_velocity.y
		var is_impact_sufficient = impact > fall_damage_velocity_threshold
		if not is_impact_sufficient:
			return
		var fall_damage_impact_multiplier = 1.0 / fall_damage_velocity_threshold
		damage = floor(impact * fall_damage_impact_multiplier)
		print("%s taking %s damage from fall on %s at speed %s" % [self, damage, body, impact])
	if damage == 0:
		return

	hit_points -= damage
	if not is_alive():
		_die(is_point_right_of(body.transform.origin))
	else:
		invincibility_frame = 1
		show_pain_or_dead()


func _on_body_exited(body: Node):
	if mode == MODE_KINEMATIC:
		return
	return on_body_exited(body)


func on_body_exited(body: Node):
	._on_body_exited(body)
	if body is Weapon:
		weapon_contact_count -= 1
	elif body is StaticBody:
		var idx = possible_impediments.find(body)
		if idx == -1:
			return
		possible_impediments.remove(idx)
		if HARD.mode():
			print("%s impediment %s removed" % [name, body])


func get_detection_radius() -> float:
	var zone_collision_shape: CollisionShape = $player_detection_zone/collision_shape
	var shape = zone_collision_shape.shape
	HARD.assert(shape is SphereShape or shape is CylinderShape)
	return shape.radius


func _apply_ground_tilt(target_transform: Transform) -> Transform:
	target_transform.basis.x.y = transform.basis.x.y
	target_transform.basis.y.y = transform.basis.y.y
	target_transform.basis.z.y = transform.basis.z.y
	return target_transform


func _interpolate_rotation_transform(
	target_transform: Transform, weight: float, force_apply_tilt: bool = false
) -> float:
	if force_apply_tilt or is_grounded():
		target_transform = _apply_ground_tilt(target_transform)
	transform = transform.interpolate_with(target_transform, weight)
	return transform.basis.z.angle_to(target_transform.basis.z)


func face_towards(position: Vector3, weight: float) -> float:
	return _interpolate_rotation_transform(transform.looking_at(position, Vector3.UP), weight)


func face_away(position: Vector3, weight: float, force_apply_tilt: bool = false) -> float:
	var target_transform = transform.looking_at(position, Vector3.UP)
	var local_up = target_transform.basis.y.normalized()
	target_transform.basis = target_transform.basis.rotated(local_up, deg2rad(180))
	return _interpolate_rotation_transform(target_transform, weight, force_apply_tilt)


func face(direction: int, position: Vector3, turn_weight: float) -> float:
	match direction:
		TOWARDS_PLAYER, TOWARDS_TERRITORY:
			return face_towards(position, turn_weight)
		AWAY_FROM_PLAYER:
			return face_away(position, turn_weight)
		_:
			HARD.assert(false, "direction must be TOWARDS (1) or AWAY (-1)")
			return INF


func is_point_right_of(position: Vector3) -> bool:
	var relative_position = to_local(position)
	var local_right_relative_translation = relative_position.x
	return local_right_relative_translation > 0


func is_point_in_front_of(position: Vector3, is_extent_included: bool = false) -> bool:
	var relative_position = to_local(position)
	var local_forward_relative_translation = -relative_position.z
	if not is_extent_included:
		return local_forward_relative_translation > 0

	var shape = _collision_shape.shape as BoxShape
	return local_forward_relative_translation > shape.extents.z


func get_ongoing_movement() -> Vector3:
	if is_flying() or not controller.is_on_floor():
		return self.observed_velocity
	return Vector3.ZERO


func is_grounded() -> bool:
	return controller.current_domain == GROUND


func is_flying() -> bool:
	return controller.current_domain == AIR


func is_climbing() -> bool:
	return controller.current_domain == CLIMB


func stop_climbing() -> void:
	controller.current_domain = GROUND


func is_avoiding_ocean() -> bool:
	return controller.is_heading_towards_ocean() or avoid_ocean_behavior.is_already_avoiding()


func is_close_to_bottom_of_climb() -> bool:
	return is_climbing() and controller.get_floor_ray().is_colliding()


func is_mid_hop():
	return is_grounded() and (not controller.is_on_floor()) and is_moving()


func is_moving():
	return observed_movement_delta.length() > 0.1


func _rng_key(behavior_name: String) -> String:
	return "%s/%s" % [get_path(), behavior_name]
