extends KinematicBody

class_name MovementController

var PHYSICS_FPS: float = ProjectSettings.get_setting("physics/common/physics_fps")
var GRAVITY_MAGNITUDE: float = ProjectSettings.get_setting("physics/3d/default_gravity")
var VELOCITY_CHANGE_FROM_GRAVITY = Vector3.DOWN * GRAVITY_MAGNITUDE * 1 / PHYSICS_FPS

const OCEAN_ELEVATION = 0
const MAX_TERRAIN_ELEVATION = 200
const FAR_BELOW_OCEAN_ELEVATION = -20

const MAX_SLIDES := 4
const FLOOR_MAX_ANGLE := deg2rad(45)
const IS_INFINITE_INERTIA_ENABLED := false
const IS_STOP_ON_SLOPE_ENABLED := true

const GROUND = "GROUND"
const CLIMB = "CLIMB"
const AIR = "AIR"

export var current_domain: String
export var gravity_velocity := Vector3.ZERO

var _previous_contact_id_buffer: Dictionary  # [object_id -> bool]


func is_disabled() -> bool:
	return $collision_shape.disabled


func enable():
	$collision_shape.disabled = false
	$floor_ray.enabled = true


func disable():
	$collision_shape.disabled = true
	$floor_ray.enabled = false


func _move_and_slide(
	linear_velocity: Vector3, snap: Vector3 = Vector3.ZERO, up: Vector3 = Vector3.UP
) -> Vector3:
	var observed = move_and_slide_with_snap(
		linear_velocity,
		snap,
		up,
		IS_STOP_ON_SLOPE_ENABLED,
		MAX_SLIDES,
		FLOOR_MAX_ANGLE,
		IS_INFINITE_INERTIA_ENABLED
	)
	if current_domain == AIR:
		return observed
	return clamp_velocity(linear_velocity, observed)


func clamp_velocity(applied: Vector3, observed: Vector3) -> Vector3:
	var xz_clamp = Vector2(applied.x, applied.z).length()
	var observed_xz = Vector2(observed.x, observed.z)
	if observed.length() > xz_clamp:
		var xz = observed_xz.normalized() * xz_clamp
		observed.x = xz.x
		observed.z = xz.y

	var clamp_threshold = 0.0 if applied.y <= 0.0 else applied.y
	if floor(observed.y) > clamp_threshold:
		if HARD.mode():
			print(
				"%s clamping observed_vel.y=%s to %s" % [get_parent(), observed.y, clamp_threshold]
			)
		var _correction = move_and_collide(Vector3(0, clamp_threshold - observed.y, 0))
		observed.y = clamp_threshold
	return observed


func move(linear_velocity: Vector3, _delta: float, snap: Vector3 = Vector3.ZERO) -> Vector3:
	if current_domain != AIR:
		gravity_velocity += VELOCITY_CHANGE_FROM_GRAVITY
	else:
		gravity_velocity = Vector3.ZERO

	var actual_velocity := _move_and_slide(linear_velocity + gravity_velocity, snap)

	if is_on_floor():
		gravity_velocity = Vector3.ZERO

	return actual_velocity


func correct_gravity(observed_movement_delta: Vector3, is_in_contact_with_ground: bool):
	if observed_movement_delta.y >= 0 and is_in_contact_with_ground and gravity_velocity.y < -10:
		gravity_velocity.y *= 0.5


func hop(linear_velocity: Vector3, _delta: float) -> Vector3:
	if is_on_floor():
		var actual_velocity := _move_and_slide(linear_velocity)
		gravity_velocity = Vector3.UP * linear_velocity.y
		return actual_velocity

	var getting_unstuck_from_slope_velocity = _move_and_slide(linear_velocity)
	return getting_unstuck_from_slope_velocity


func is_able_to_climb() -> bool:
	if HARD.mode():
		HARD.assert(get_parent().is_able_to_climb, "%s is not able to climb" % get_parent())
	var climbing_ray = get_climbing_ray()
	return (
		climbing_ray.is_colliding()
		and InteractionHandlers.is_climbable(climbing_ray.get_collider())
		and InteractionHandlers.is_position_climbable(
			get_tree(), climbing_ray.get_collision_point()
		)
	)


func climb(linear_velocity: Vector2, delta: float) -> Vector3:
	var climbing_ray = get_climbing_ray()
	current_domain = CLIMB
	gravity_velocity = Vector3.ZERO
	var climbing_velocity = MouseKeyboardPlayer.resolve_climbing_velocity(
		climbing_ray, global_transform.basis, linear_velocity
	)
	return _move_and_slide(climbing_velocity * delta)


func get_floor_ray() -> RayCast:
	return $floor_ray as RayCast


func get_climbing_ray() -> RayCast:
	HARD.assert($climbing_ray is RayCast, "%s cannot climb without a $climbing_ray" % self)
	return $climbing_ray as RayCast


# https://kidscancode.org/godot_recipes/3d/3d_align_surface/
func align_with_ray(
	xform: Transform,
	ray: RayCast,
	direction: Vector3 = Vector3.DOWN,
	max_slope: float = deg2rad(45)
) -> Transform:
	var new_y = ray.get_collision_normal()
	var surface_slope = (-direction).angle_to(new_y)
	if (not ray.is_colliding()) or surface_slope > max_slope:
		new_y = -direction
	var new_x = -xform.basis.z.cross(new_y)
	if new_x.length() == 0:
		return xform
	xform.basis.y = new_y
	xform.basis.x = new_x
	xform.basis = xform.basis.orthonormalized()
	return xform


func align_and_position_parent(xform: Transform, floor_alignment_interpolation_weight: float = 0.2) -> Transform:
	xform.origin = global_transform.origin
	transform.origin = Vector3.ZERO
	if current_domain == AIR:
		return xform
	return xform.interpolate_with(
		align_with_ray(xform, get_floor_ray()), floor_alignment_interpolation_weight
	)


func distance_from_desired_altitude() -> float:
	# TODO return this assert?
	#	HARD.assert(
	#		current_domain == AIR,
	#		"doesn't make sense to call is_at_desired_altitude unless animal.is_flying()"
	#	)
	if current_domain != AIR:
		print("WARNING: invalid domain for distance_from_desired_altitude: %s" % [current_domain])
		return 0.0

	# TODO layer/mask such that only floor collides
	var ray = get_floor_ray()
	var distance_from_floor = global_transform.origin.distance_to(ray.get_collision_point())
	return distance_from_floor + ray.cast_to.y


func is_at_desired_altitude() -> bool:
	return abs(distance_from_desired_altitude()) < 2


func is_above_desired_altitude() -> bool:
	return not get_floor_ray().is_colliding()


func get_heading_dir() -> Vector3:
	var heading = get_parent().observed_velocity
	if heading.length() == 0:
		heading = -global_transform.basis.z
	return Vector3(heading.x, 0, heading.z).normalized()


func is_heading_towards_ocean() -> bool:
	var heading_in_dir = get_heading_dir()

	var cast_forward_by = 5 if current_domain == AIR else 3
	var offset = heading_in_dir * (cast_forward_by + $collision_shape.shape.extents.z)
	var cast_from = global_transform.origin + offset
	cast_from.y = MAX_TERRAIN_ELEVATION
	var cast_to = Vector3(cast_from.x, FAR_BELOW_OCEAN_ELEVATION, cast_from.z)

	var space_state = get_world().direct_space_state
	var collison = space_state.intersect_ray(cast_from, cast_to, [])

	var is_looking_at_void = not "collider" in collison
	return is_looking_at_void or collison["position"].y <= OCEAN_ELEVATION


func cast_rays(is_flying: bool):
	if is_flying:
		$floor_ray.global_transform.basis = Basis()


func get_contact_changes() -> Dictionary:
	if _previous_contact_id_buffer == null:
		_previous_contact_id_buffer = {}

	var new_buffer = {}
	var impact_changes = {
		"entered": [],
		"exited": [],
	}

	for index in get_slide_count():
		var collider = get_slide_collision(index).collider
		if not (collider is PhysicsBody):
			continue
		var object_id = collider.get_instance_id()
		if object_id in new_buffer:
			continue
		new_buffer[object_id] = true
		var is_persisting = _previous_contact_id_buffer.erase(object_id)
		if is_persisting:
			continue
		impact_changes["entered"].append(collider)

	for object_id in _previous_contact_id_buffer:
		var exited = instance_from_id(object_id)
		impact_changes["exited"].append(exited)

	_previous_contact_id_buffer = new_buffer
	return impact_changes
