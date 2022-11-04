extends Entity

class_name Door

export var is_latched: bool = true
export var locks: Dictionary = {}


func _ready():
	var door_body = find_node("body")
	HARD.assert(door_body, "Can't find door body in %s" % self)

	if is_latched:
		latch()
	else:
		unlatch()


func is_locked():
	return locks.values().count(true) > 0


func latch():
	is_latched = true


func unlatch():
	if is_locked():
		return false
	else:
		is_latched = false
		return true


func lock(lock_id: int):
	locks[lock_id] = true
	if is_locked():
		_color_handle(Color(1.0, 0.0, 0.0, 1.0))


func unlock(lock_id: int):
	locks[lock_id] = false
	if not is_locked():
		_color_handle(Color(0.0, 1.0, 0.0, 1.0))


func _color_handle(_color: Color):
	pass
#	var door_body = self.find_node("body", true, false)
#	var i = 0
#	while true:
#		var handle_mesh = door_body.find_node("handle_%s_mesh" % i, true, false)
#		if handle_mesh == null:
#			break
#		var material = handle_mesh.get_surface_material(0).duplicate()
#		material.set_shader_param("color1", color)
#		handle_mesh.set_surface_material(0, material)
#		i += 1
