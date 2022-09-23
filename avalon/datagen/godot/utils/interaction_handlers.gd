extends Reference

class_name InteractionHandlers

const GRAB_JOINT_NAME = "grab_joint"


static func is_grabbable(object: Node) -> bool:
	return (
		object
		and is_instance_valid(object)
		and object.has_method("is_grabbable")
		and object.is_grabbable() == true
	)


static func is_climbable(object: Node) -> bool:
	return (
		(
			object
			and is_instance_valid(object)
			and object.has_method("is_climbable")
			and object.is_climbable() == true
		)
		or Tools.is_terrain(object)
		or Tools.is_tree(object)
	)


static func is_throwable(object: Node) -> bool:
	return (
		object
		and is_instance_valid(object)
		and object.has_method("is_throwable")
		and object.is_throwable() == true
	)


# TODO you can stil push objects even if they aren't pushable
static func is_pushable(object: Node) -> bool:
	return (
		object
		and is_instance_valid(object)
		and object.has_method("is_pushable")
		and object.is_pushable() == true
	)


static func is_edible(object: Node) -> bool:
	if not object or not object.has_method("is_edible"):
		return false
	HARD.assert(object.has_method("eat"), "%s claims it is_edibile but has no eat method" % object)
	return object.is_edible() == true


static func attempt_grab(hand: RigidBody, object: Node) -> Node:
	if not is_grabbable(object):
		return null
	HARD.assert(
		object.has_method("grab"), "%s claims it is_grabbable but has no grab method" % object
	)

	var grabbed = object.grab(hand)

	if HARD.mode():
		prints("grabbed %s" % grabbed, "" if grabbed == object else "instead of %s" % object)

	if grabbed != null:
		# note: only grabbable should ever need to add a joint
		add_joint(hand, grabbed)

	return grabbed


static func attempt_climb(hand: RigidBody, object: Node) -> Node:
	if (
		not is_climbable(object)
		or not is_position_climbable(hand.get_tree(), hand.global_transform.origin)
	):
		return null

	# TODO temporary fix to let you climb terrain without providing `climb` or `is_climbable`
	if Tools.is_terrain(object) or Tools.is_tree(object):
		if HARD.mode():
			print("climbed %s" % object)
		return object

	HARD.assert(
		object.has_method("climb"), "%s claims it is_climbable but has no climb method" % object
	)

	if HARD.mode():
		print("climbed %s" % object)

	return object.climb(hand)


static func attempt_throw(object: Node, throw_impulse: Vector3) -> void:
	if not is_throwable(object) and not object.has_method("throw"):
		return
	HARD.assert(object.has_method("throw"), "%s has no throw method" % object)

	# release the object than apply the impulse
	attempt_release(object)

	if HARD.mode():
		print("threw %s" % object)

	object.throw(throw_impulse)


# TODO you can stil push objects even if they aren't pushable
static func attempt_push(object: Node, push_impulse: Vector3, push_offset: Vector3) -> void:
	if not is_pushable(object) or not object.has_method("push"):
		return
	HARD.assert(object.has_method("push"), "%s has no push method" % object)

	if HARD.mode():
		print("pushed %s" % object)

	object.push(push_impulse, push_offset)


static func attempt_release(object: Node) -> void:
	# note: will only release a joint if it exists
	release_joint(object)

	if not object or not object.has_method("release"):
		return
	HARD.assert(object.has_method("release"), "%s has no release method" % object)

	if HARD.mode():
		print("released %s" % object)

	object.release()


static func attempt_hold(hand: RigidBody, object: Node) -> Node:
	if not Tools.is_static(object):
		object.sleeping = false

	# assume we can keep holding the object if we don't explicitly provide a hold method
	if not object or not object.has_method("hold"):
		return object
	HARD.assert(object.has_method("hold"), "%s has no hold method" % object)

	if HARD.mode():
		print("held %s" % object)

	return object.hold(hand)


static func attempt_eat(object: Node) -> float:
	if not is_edible(object) and not object.has_method("eat"):
		return 0.0
	HARD.assert(object.has_method("eat"), "%s has no eat method" % object)

	# release the object than apply the impulse
	attempt_release(object)

	if HARD.mode():
		print("ate %s" % object)

	return object.eat()


static func is_interactable_with_hand(object: Node) -> bool:
	return is_climbable(object) or is_grabbable(object)


static func is_able_to_collide_while_grabbing(object: Node) -> bool:
	return object.get_parent().get_class() == "Entity"


static func has_joint(thing: Node) -> bool:
	return (
		is_instance_valid(thing)
		and thing.has_node(GRAB_JOINT_NAME)
		and is_instance_valid(thing.get_node(GRAB_JOINT_NAME))
	)


static func add_joint(hand: Node, thing: Node) -> void:
	if has_joint(thing):
		return

	thing.sleeping = false
	thing.gravity_scale = 0.0

	var grab_joint = Generic6DOFJoint.new()
	grab_joint.name = GRAB_JOINT_NAME
	thing.add_child(grab_joint, true)
	grab_joint.global_transform.origin = hand.global_transform.origin
	grab_joint.set("nodes/node_a", thing.get_path())
	grab_joint.set("nodes/node_b", hand.get_path())
	grab_joint.set("collision/exclude_nodes", false)
	grab_joint.set("linear_limit_x/damping", 0.0)
	grab_joint.set("linear_limit_y/damping", 0.0)
	grab_joint.set("linear_limit_z/damping", 0.0)


static func release_joint(held_thing: Node) -> void:
	if not has_joint(held_thing):
		return
	var grab_joint: Generic6DOFJoint = held_thing.get_node(GRAB_JOINT_NAME)
	held_thing.remove_child(grab_joint)
	grab_joint.queue_free()
	held_thing.gravity_scale = 1.0


static func is_position_climbable(scene_tree: SceneTree, position: Vector3) -> bool:
	var terrain_manager = scene_tree.root.find_node("TerrainManager", true, false)
	HARD.assert(
		terrain_manager != null, "terrain_manager missing from scene or has unexpected name"
	)
	return terrain_manager and terrain_manager.is_position_climbable(position)
