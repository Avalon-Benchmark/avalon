extends DynamicEntity

class_name Item

const _DEFAULT_SHAPE_RADIUS := 1.0

export var _is_state_initialized = false

export var ground_contact_history = []
export var ground_contact_count := 0
export var is_behaving_like_item = true

export var default_linear_damp := 0.0
export var default_angular_damp := 0.0

export var max_ground_contact_history_len := 10
export var damping_velocity_cutoff := 10.0
export var max_damping := 2.0
export var angular_damp_factor := 2.0
export var linear_damp_factor := 0.5
# the higher this number, the more damping that is applied to objects that are in contact with the ground
export var ground_contact_damp_factor := 4.0
# the higher this number the less gravity force that gets applied to objects stuck in the ground
export var gravity_scale_factor := 0.8  # must be <1.0

# used for moving the object out of the ground
export var bounding_radius_factor = 1.05

export var safe_scale := Vector3.ONE
export var base_color := ""

export var _prev_origin: Vector3

export var is_held := false
export var is_hidden := false

export var previous_velocity := Vector3.ZERO

# ignores body_entered signals when reloading, which only happens after the first frame
var _reload_ground_contact_count_correction_buffer := 0


func is_impossible_to_eat() -> bool:
	return is_hidden


func _ready():
	HARD.assert(OK == connect("body_entered", self, "_on_body_entered"), "Failed to connect signal")
	HARD.assert(OK == connect("body_exited", self, "_on_body_exited"), "Failed to connect signal")
	if not _is_state_initialized:
		_initialize_state()
	else:
		_reload_ground_contact_count_correction_buffer = ground_contact_count


func _initialize_state():
	contact_monitor = true
	contacts_reported = 10

	ground_contact_history = []

	if (self.safe_scale - Vector3.ONE).length() > 0.01:
		set_scale_safely(self.safe_scale)

	if self.base_color:
		set_base_color(self.base_color)

	_set_previous_position()
	_is_state_initialized = true


# required because godot is weird/buggy and things go crazy if you apply scale to a collision shape
func set_scale_safely(scale: Vector3):
	for child in self.get_children():
		if child is CollisionShape:
			var shape = child.shape.duplicate()
			if shape is SphereShape:
				shape.radius *= scale.x
			elif shape is BoxShape:
				shape.extents.x *= scale.x
				shape.extents.y *= scale.y
				shape.extents.z *= scale.z
			elif shape is CylinderShape:
				shape.radius *= scale.x
				shape.height *= scale.y
			elif shape is ConvexPolygonShape:
				var new_points = []
				for point in shape.points:
					var new_point = Vector3(point.x * scale.x, point.y * scale.y, point.z * scale.z)
					new_points.append(new_point)
				shape.points = new_points
			else:
				HARD.stop("Unhandled collision shape type: " + str(shape))
			child.shape = shape
		if child is MeshInstance:
			child.scale *= scale
	self.safe_scale = scale


func set_base_color(color):
	for child in self.get_children():
		if child is MeshInstance:
			var material = child.mesh.surface_get_material(0).duplicate()
			material.albedo_color = Color(color)
			material.vertex_color_use_as_albedo = false
			child.material_override = material


func _on_body_entered(body: Node):
	if Tools.is_terrain(body):
		if _reload_ground_contact_count_correction_buffer > 0:
			_reload_ground_contact_count_correction_buffer -= 1
			return
		ground_contact_count += 1


func _on_body_exited(body: Node):
	if Tools.is_terrain(body):
		ground_contact_count -= 1
		if ground_contact_count < 0:
			ground_contact_count = 0


func _reset():
	# NOTE: `ground_contact_count` will resolve itself DO NOT add it here
	gravity_scale = 1.0
	linear_damp = 0.0
	angular_damp = 0.0


func hide() -> void:
	# instead if freeing objects we turn off their collision shape and visbility
	visible = false
	for child in get_children():
		if child is CollisionShape:
			child.disabled = true
	global_transform.origin = Vector3.ZERO
	sleeping = true
	is_hidden = true


func is_grabbable() -> bool:
	return false


func is_climbable() -> bool:
	return false


func is_throwable() -> bool:
	return false


func is_pushable() -> bool:
	return false


func is_edible() -> bool:
	return false


func grab(_physical_hand: RigidBody) -> Node:
	_reset()
	return self


func climb(_physical_hand: RigidBody) -> Node:
	_reset()
	return self


func throw(throw_impulse: Vector3) -> void:
	_reset()
	apply_central_impulse(throw_impulse)


func push(push_impulse: Vector3, push_offset: Vector3) -> void:
	_reset()
	apply_impulse(push_offset, push_impulse)


func eat() -> float:
	return HARD.assert(false, "Not implemented")


func hold(_physical_hand: RigidBody) -> Node:
	is_held = true
	_reset()
	return self


func release() -> void:
	is_held = false
	# TODO get player in a better way
	var player = get_tree().root.find_node("player", true, false)
	var throw_factor = player.get_throw_factor(mass)
	linear_velocity *= throw_factor
	angular_velocity *= throw_factor


func get_bounding_sphere_radius() -> float:
	for child in get_children():
		if child is MeshInstance:
			var aabb = child.get_aabb()
			return aabb.size.length() / 2 * bounding_radius_factor
	return _DEFAULT_SHAPE_RADIUS * bounding_radius_factor


func _set_previous_position() -> void:
	_prev_origin = global_transform.origin


func behave_like_item() -> void:
	if not is_behaving_like_item:
		_set_previous_position()
		is_behaving_like_item = true


func handle_object_falling_through_floor() -> bool:
	var space_state = get_world().direct_space_state

	# see if the object has passed through the ground by casting a ray from where we were to where
	# we are now
	var terrain_ray = space_state.intersect_ray(_prev_origin, global_transform.origin, [self])

	var did_move_out_of_floor = false
	if terrain_ray:
		var collider = terrain_ray["collider"]

		if Tools.is_terrain(collider):
			if HARD.mode():
				printt("moving out of terrain %s" % self)
			var normal = terrain_ray["normal"]
			var distance_travelled = global_transform.origin - _prev_origin
			global_transform.origin = (
				terrain_ray["position"]
				- (distance_travelled.normalized() * get_bounding_sphere_radius())
			)

			var bounce = 0.0
			if physics_material_override and not physics_material_override.absorbent:
				bounce = physics_material_override.bounce
			linear_velocity = linear_velocity.bounce(normal) * bounce
			did_move_out_of_floor = true

	_set_previous_position()

	return did_move_out_of_floor


func _physics_process(_delta):
	# TODO bit awkward, but need a hook to bypass call_multilevel behavior for animals
	if not is_behaving_like_item or is_hidden:
		return

	HARD.assert(
		ground_contact_count >= 0,
		"Ground contact is less than zero, this will break all the physics"
	)

	previous_velocity = linear_velocity

	# sometimes godot physics sucks and the object goes right through the floor
	# if that happens we correct course manually
	if not is_held:
		var _was_moved_out_of_terrain = handle_object_falling_through_floor()
	else:
		_set_previous_position()
		return

	# if we're at rest, reset everything
	if sleeping:
		_reset()
		return

	# update out ground contact history
	if ground_contact_count > 0:
		ground_contact_history.push_back(1)
	else:
		ground_contact_history.push_back(0)
	while ground_contact_history.size() > max_ground_contact_history_len:
		ground_contact_history.pop_front()
	var recent_ground_contacts = 0
	for contact in ground_contact_history:
		recent_ground_contacts += contact
	var ground_contact_frequency := (
		float(recent_ground_contacts)
		/ float(max_ground_contact_history_len)
	)

	# we scale the force of gravity depending on how many frames
	# this object has been in contact with the ground over the
	# pas second or so. This helps prevent the object from falling
	# through the ground
	gravity_scale = (
		(1.0 - ground_contact_frequency) * gravity_scale_factor
		+ (1.0 - gravity_scale_factor)
	)

	# if we did not hit the ground this frame, nothing special to do:
	if ground_contact_count == 0:
		linear_damp = default_linear_damp
		angular_damp = default_angular_damp
		return

	# if we hit the ground relatively fast
	# then we dont really want to mess with anything
	# (ie, normal bounce should apply)
	if linear_velocity.length_squared() > damping_velocity_cutoff * damping_velocity_cutoff:
		linear_damp = default_linear_damp
		angular_damp = default_angular_damp
		return

	# if we've hit the ground kind of slowly
	# then we want to start damping this motion
	var damping_factor := (
		(damping_velocity_cutoff - linear_velocity.length())
		/ damping_velocity_cutoff
	)
	var damping = damping_factor * max_damping
	damping += ground_contact_frequency * ground_contact_damp_factor
	linear_damp = default_linear_damp + damping * linear_damp_factor
	angular_damp = default_angular_damp + damping * angular_damp_factor


func get_recent_velocity_buffer() -> Array:
	return [previous_velocity, linear_velocity]


func _calculate_impact_magnitude(other) -> float:
	if other is Vector3:
		return _max_magnitude_difference(other)
	elif other.has_method("get_recent_velocity_buffer"):
		var impact_magnitude = 0.0
		for v in other.get_recent_velocity_buffer():
			var step_magnitude = _max_magnitude_difference(v)
			if step_magnitude > impact_magnitude:
				impact_magnitude = step_magnitude
		return impact_magnitude
	elif other is RigidBody:
		return _max_magnitude_difference(other.linear_velocity)

	HARD.assert(
		other is Node, "_calculate_impact_magnitude should only be called with velocities or nodes"
	)
	return _max_magnitude_difference(Vector3.ZERO)


func _max_magnitude_difference(other_velocity: Vector3) -> float:
	var max_diff = 0.0
	for v in get_recent_velocity_buffer():
		var diff = (other_velocity - v).length()
		if diff > max_diff:
			max_diff = diff
	return max_diff
