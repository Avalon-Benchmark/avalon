extends Stackable

class_name Log

onready var player = get_tree().root.find_node("player", true, false)
var _prev_endpoints: Array


func _ready():
	continuous_cd = true
	contact_monitor = true
	contacts_reported = 5
	linear_damp_factor = 1.0
	angular_damp_factor = 0.3
	ground_contact_damp_factor = 4.0
	_prev_endpoints = _get_endpoints_for_raycasting_with_terrain()


func get_aabb() -> AABB:
	for child in get_children():
		if child is MeshInstance:
			return child.get_aabb()
	return HARD.assert(false, "Couldn't find a mesh instance for %s" % name)


func get_max_distance_from_collision_plane(collision_plane: Plane) -> float:
	var aabb = get_aabb()
	var max_distance = 0.0
	for i in range(8):
		var endpoint = aabb.get_endpoint(i)
		var rotated_endpoint = global_transform.basis * endpoint + global_transform.origin
		if not collision_plane.is_point_over(rotated_endpoint):
			var dist = abs(collision_plane.distance_to(rotated_endpoint))
			if dist > max_distance:
				max_distance = dist
	return max_distance


func get_endpoints() -> Array:
	var endpoints = []
	var aabb = get_aabb()
	for i in range(8):
		var endpoint = aabb.get_endpoint(i)
		endpoints.append(global_transform.basis * endpoint + global_transform.origin)
	endpoints.append(global_transform.origin)
	return endpoints


func _get_endpoints_for_raycasting_with_terrain():
	var aabb = get_aabb()
	var top = Vector3.ZERO
	var bottom = Vector3.ZERO
	for i in range(8):
		var endpoint = aabb.get_endpoint(i)
		if endpoint.y > 0:
			top += endpoint
		else:
			bottom += endpoint
	if top or bottom:
		pass
	top /= 4
	bottom /= 4

	var axis_index = aabb.get_longest_axis_index()
	var max_size = aabb.get_longest_axis_size()

	# move endpoints slightly off the top / bottom to make sure when used for raycasting they don't
	# fire unintentionally
	top[axis_index] -= max_size * 0.1
	bottom[axis_index] += max_size * 0.1

	top = global_transform.basis * top + global_transform.origin
	bottom = global_transform.basis * bottom + global_transform.origin

	return [global_transform.origin, top, bottom]


func _set_previous_position() -> void:
	_prev_endpoints = _get_endpoints_for_raycasting_with_terrain()


func handle_object_falling_through_floor() -> bool:
	var space_state = get_world().direct_space_state
	# check multiple points on the log for terrain collisions
	var endpoints = _get_endpoints_for_raycasting_with_terrain()
	for i in range(len(endpoints)):
		var prev_endpoint = _prev_endpoints[i]
		var curr_endpoint = endpoints[i]
		var distance_travelled = curr_endpoint - prev_endpoint
		var terrain_ray = space_state.intersect_ray(prev_endpoint, curr_endpoint, [self])
		if terrain_ray:
			var collider = terrain_ray["collider"]
			if Tools.is_terrain(collider):
				if HARD.mode():
					printt("moving out of terrain %s" % self)
				var normal = terrain_ray["normal"]
				# sigh ... sometimes the normal points down into the terrain which is very sad
				if normal.angle_to(Vector3.UP) > PI / 2:
					continue
				var position = terrain_ray["position"]
				var collision_plane = Plane(normal, normal.dot(position))
				# check all corners of the AABB and find the one that is furtherest in
				var distance_to_move_object_out_of_terrain = get_max_distance_from_collision_plane(
					collision_plane
				)
				global_transform.origin += (
					distance_to_move_object_out_of_terrain
					* bounding_radius_factor
					* -distance_travelled.normalized()
				)
				var bounce = physics_material_override.bounce if physics_material_override else 0.05
				linear_velocity = linear_velocity.bounce(normal) * bounce
				# get new endpoints since we've moved it out of the ground and don't want to trigger spurious
				# terrain detections
				_set_previous_position()
				return true

	_set_previous_position()
	return false


func _physics_process(_delta):
	var player_collisions = player.get_wall_collisions()
	var player_floor_collision = player.get_floor_collision()
	if player_floor_collision:
		player_collisions.append(player_floor_collision)
	for collision in player_collisions:
		var is_item_on_floor = ground_contact_count > 0
		if collision.collider == self and is_item_on_floor:
			if HARD.mode():
				print("player collided with %s, making static" % self)
			mode = RigidBody.MODE_STATIC
