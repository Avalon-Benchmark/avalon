extends Spatial
class_name PlayerHand

var physical_hand: RigidBody
var hand_marker: MeshInstance

var _held_thing: PhysicsBody
var _thing_colliding_with_hand: PhysicsBody

var MARKER_ACTIVE := preload("res://materials/active.tres")
var MARKER_OBJECT := preload("res://materials/object.tres")
var MARKER_SELECTED := preload("res://materials/selected.tres")

# how much farther you can reach when grabbing the ground.
# Prevents you from having to bend over to grab mildly sloped terrain
var EXTRA_REACH_FOR_TERRAIN := 1.5

var MAX_HOLD_RADIUS := 0.5
var INITIAL_GRAB_RADIUS := 0.15
var MAX_CLIMB_ANGLE = deg2rad(40)


func _ready():
	var root = get_tree().root
	physical_hand = root.find_node(
		"physical_%s" % name.to_lower().replace("target_", ""), true, false
	)
	hand_marker = physical_hand.get_node("marker")


# TODO seems silly we pass body in here, it's definetly leaky between player hand and player
func do_action(is_grasping: bool, body: PhysicsBody, throw_impulse: Vector3 = Vector3.ZERO) -> void:
	var is_throwing = not throw_impulse.is_equal_approx(Vector3.ZERO)
	var thing_colliding_with_hand = get_thing_colliding_with_hand()
	var currently_held_thing = get_held_thing()

	if is_throwing:
		if InteractionHandlers.is_throwable(currently_held_thing):
			_throw(body, currently_held_thing, throw_impulse)
	elif is_grasping:
		# continue holding on to the currently held thing
		if currently_held_thing:
			_hold(currently_held_thing)

			# TODO can use this for more things just outside of climbing (ie. doors)
			var interactable_things_near_hand = _get_things_colliding_with_hand_in_radius(
				MAX_HOLD_RADIUS
			)
			var is_held_thing_in_reach = (
				currently_held_thing in interactable_things_near_hand
				or currently_held_thing == thing_colliding_with_hand
			)

			# if we're no longer holding something we should make sure we release it
			if (
				not get_held_thing()
				or (
					InteractionHandlers.is_climbable(currently_held_thing)
					and not is_held_thing_in_reach
				)
			):
				_release(body, currently_held_thing)

		# try to grab something that is colliding with the hand
		else:
			if InteractionHandlers.is_grabbable(thing_colliding_with_hand):
				_grab(body, thing_colliding_with_hand)
			if InteractionHandlers.is_climbable(thing_colliding_with_hand):
				_climb(thing_colliding_with_hand)
	else:
		# if dropping a thing while we're hold it
		if currently_held_thing:
			_release(body, currently_held_thing)


func _hold(object: Spatial) -> void:
	set_held_thing(InteractionHandlers.attempt_hold(physical_hand, object))


func _release(body: PhysicsBody, object: Node) -> void:
	remove_collision_exception(body)
	InteractionHandlers.attempt_release(object)
	set_held_thing(null)


func _grab(body: PhysicsBody, object: Node) -> void:
	set_held_thing(InteractionHandlers.attempt_grab(physical_hand, object))

	# don't add a collision exception with entities
	if not InteractionHandlers.is_able_to_collide_while_grabbing(object):
		add_collision_exception(body)


func _climb(object: Node) -> void:
	set_held_thing(InteractionHandlers.attempt_climb(physical_hand, object))


func _throw(body: PhysicsBody, object: Node, throw_impulse: Vector3) -> void:
	remove_collision_exception(body)
	InteractionHandlers.attempt_throw(object, throw_impulse)
	set_held_thing(null)


func add_collision_exception(body: PhysicsBody) -> void:
	var currently_held_thing = get_held_thing()
	if (
		is_instance_valid(currently_held_thing)
		and not currently_held_thing in body.get_collision_exceptions()
	):
		body.add_collision_exception_with(currently_held_thing)


func remove_collision_exception(body: PhysicsBody):
	var currently_held_thing = get_held_thing()
	if (
		is_instance_valid(currently_held_thing)
		and currently_held_thing in body.get_collision_exceptions()
	):
		body.remove_collision_exception_with(currently_held_thing)


func get_held_thing():
	if not is_instance_valid(_held_thing):
		_held_thing = null
	return _held_thing


func set_held_thing(object: Node):
	_held_thing = object


func _get_things_colliding_with_hand_in_radius(radius: float, is_ignoring_terrain: bool = true):
	var world := get_world()
	var space := world.direct_space_state
	var query := PhysicsShapeQueryParameters.new()

	var shape := SphereShape.new()
	shape.radius = radius
	query.set_shape(shape)
	query.transform = global_transform

	var things = []
	for result in space.intersect_shape(query):
		var thing = result["collider"]
		if thing.name == "physical_body":
			continue
		if Tools.is_terrain(thing) and is_ignoring_terrain:
			continue

		var collision_shape_idx = result["shape"]
		var collision_shapes = []
		for child in thing.get_children():
			if child is CollisionShape:
				collision_shapes.append(child)
		if collision_shape_idx >= 0 and collision_shape_idx <= len(collision_shapes) - 1:
			# The physics engine sometimes seems to spew back garbage for shape ids
			var colliding_shape = collision_shapes[collision_shape_idx]
			# TODO I'm inclined to let `grab` handle these cases (it can return null if something is not grabbable)
			#	but will leave this as is for now
			if colliding_shape is NonGrabbable:
				continue
		if InteractionHandlers.is_interactable_with_hand(thing):
			things.append(thing)
	return things


func can_grab_flat_terrain_surface() -> bool:
	# TODO sigh ... this is hacky we shouldn't get the player like this
	var _player = get_parent().get_parent()
	var is_body_in_terrain = _player.ground_ray.is_colliding()
	var is_climbing = _player.is_climbing()
	return is_climbing or is_body_in_terrain


func get_thing_colliding_with_hand(is_ignoring_terrain: bool = true) -> PhysicsBody:
	var things = _get_things_colliding_with_hand_in_radius(INITIAL_GRAB_RADIUS, is_ignoring_terrain)
	var thing = things.pop_front()
	set_thing_colliding_with_hand(thing)

	if thing:
		return thing

	# see if hand has passed through the terrain
	var space_state = get_world().direct_space_state
	var head = physical_hand.get_parent().get_node("physical_head")
	var body = physical_hand.get_parent().get_node("physical_body")
	var hand_dir = (global_transform.origin - head.global_transform.origin).normalized()
	var far_hand = global_transform.origin + hand_dir * EXTRA_REACH_FOR_TERRAIN
	var terrain_ray = space_state.intersect_ray(
		far_hand,
		head.global_transform.origin,
		[self, physical_hand, get_parent(), physical_hand.get_parent(), head, body]
	)
	if terrain_ray:
		var collider = terrain_ray["collider"]
		var normal = terrain_ray["normal"]
		if (
			Tools.is_terrain(collider)
			and (normal.angle_to(Vector3.DOWN) > MAX_CLIMB_ANGLE or can_grab_flat_terrain_surface())
		):
			var terrain_point = terrain_ray["position"]
			if InteractionHandlers.is_position_climbable(get_tree(), terrain_point):
				set_thing_colliding_with_hand(collider)
				return _thing_colliding_with_hand

	return null


func set_thing_colliding_with_hand(object: Node):
	_thing_colliding_with_hand = object


func get_rid(object) -> RID:
	return RID(object)


func set_hand_debug_marker(material: Material) -> void:
	hand_marker.material_override = material


func is_grasping_heavy_thing() -> bool:
	var currently_held_thing = get_held_thing()
	return (
		Tools.is_static(currently_held_thing)
		or InteractionHandlers.is_climbable(currently_held_thing)
	)


func is_colliding_with_thing() -> bool:
	return _thing_colliding_with_hand != null


func get_id_of_thing_colliding_with_hand() -> int:
	if is_instance_valid(_thing_colliding_with_hand):
		return _thing_colliding_with_hand.get_rid().get_id()
	return -1


func get_id_of_held_thing() -> int:
	var held_thing = get_held_thing()
	if is_instance_valid(held_thing):
		return held_thing.get_rid().get_id()
	return -1
