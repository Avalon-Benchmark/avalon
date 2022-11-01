extends Item

class_name Food

export var initial_stem_adjustement_seconds := 0.0

export var energy = 1.0
export var pluck_velocity_threshold = 1.0

export var _instantiation_time: float
export var _is_interacting := false


func _ready():
	if _instantiation_time == null:
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


func eat() -> float:
	if not is_edible():
		return 0.0
	hide()
	return energy


func is_on_plant() -> bool:
	return has_node("%s_stem_joint" % name)


# TODO: I cannot seem to reproduce the issues that led to this buffer period in the current codebase,
#       but given impending data collection I'll leave it in and disabled
#       so levels can be regenerated serverside with overridden defaults
func _is_initial_stem_adjustment() -> bool:
	if initial_stem_adjustement_seconds == 0.0 or _is_interacting:
		return false
	var end_adjustment_at_msec = _instantiation_time + (1000.0 * initial_stem_adjustement_seconds)
	return OS.get_ticks_msec() < end_adjustment_at_msec


func pluck():
	if _is_initial_stem_adjustment():
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
	HARD.assert(has_node("open_mesh"), "called _open helper on unopenable food")


func _on_body_entered(body: Node):
	._on_body_entered(body)
	if _is_interacting or Tools.is_tree(body):
		return
	_is_interacting = true


func hold(physical_hand: RigidBody) -> Node:
	_is_interacting = true
	return .hold(physical_hand)


func _is_open() -> bool:
	var open_mesh: MeshInstance = $open_mesh
	return open_mesh.visible


func _open():
	_validate_openable()
	var open_mesh: MeshInstance = $open_mesh

	if open_mesh.visible:
		return

	var closed_mesh: MeshInstance = $closed_mesh
	open_mesh.visible = true
	closed_mesh.visible = false


func _is_impact_velocity_sufficient(body: Node, velocity_threshold: float, action: String = "open") -> bool:
	var impact_magnitude = _calculate_impact_magnitude(body)
	var is_sufficient = impact_magnitude >= velocity_threshold
	if HARD.mode():
		var verb = action + "ed" if is_sufficient else "was too slow to " + action
		print("%s impact with %s %s %s" % [self, body, verb, impact_magnitude])
	return is_sufficient
