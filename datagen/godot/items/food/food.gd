extends Item

class_name Food

export var energy = 1.0
export var pluck_velocity_threshold = 1.0

var _instantiation_time: float


func _ready():
	_instantiation_time = OS.get_ticks_msec()
	add_to_group("food")


func is_edible() -> bool:
	return true


func is_throwable() -> bool:
	return true


func is_grabbable() -> bool:
	return true


func is_pushable() -> bool:
	return true


# Attempt to eat this food. WARNING: Will free the food if it is fully consumed
func eat() -> float:
	if not is_edible():
		return 0.0
	hide()
	return energy


func is_on_plant() -> bool:
	return has_node("%s_stem_joint" % name)


func pluck():
	# TODO can this be lower
	var is_initial_stem_adjustment = _instantiation_time + 1000 > OS.get_ticks_msec()
	if is_initial_stem_adjustment:
		if HARD.mode():
			print("initial adjustment ignored: %s at speed %f" % [name, linear_velocity.length()])
		return
	if HARD.mode():
		print("plucked %s at speed %f" % [name, linear_velocity.length()])
	var stem = get_node("%s_stem_joint" % name)
	if stem != null:
		stem.queue_free()


func _physics_process(delta):
	._physics_process(delta)
	if is_on_plant() and linear_velocity.length() > pluck_velocity_threshold:
		pluck()


func _validate_openable():
	HARD.assert(
		has_node("open_mesh") and has_node("open_collision_shape"),
		"called _open helper on unopenable food"
	)


func _is_open() -> bool:
	var open_mesh: MeshInstance = $open_mesh
	return open_mesh.visible


func _open():
	_validate_openable()
	var open_mesh: MeshInstance = $open_mesh

	if open_mesh.visible:
		return

	var open_collision_shape: CollisionShape = $open_collision_shape
	var closed_mesh: MeshInstance = $closed_mesh
	var closed_collision_shape: CollisionShape = $closed_collision_shape

	closed_collision_shape.disabled = true
	open_collision_shape.disabled = false
	open_mesh.visible = true
	closed_collision_shape.disabled = true
	closed_mesh.visible = false


func _is_impact_velocity_sufficient(body: Node, velocity_threshold: float) -> bool:
	var other_velocity = body.linear_velocity if body is RigidBody else Vector3.ZERO
	var impact_magnitude = (other_velocity - linear_velocity).length()
	var is_sufficient = impact_magnitude >= velocity_threshold
	if HARD.mode():
		var verb = "opened" if is_sufficient else "was too slow to open"
		print("%s impact with %s %s %s" % [self, body, verb, impact_magnitude])
	return is_sufficient
