extends Reference

class_name Tools


static func dump(object, indent := "    ", level := 0, terminator := "\n", indent_line := true) -> void:
	if indent_line:
		printraw(indent.repeat(level))
	match typeof(object):
		TYPE_BOOL, TYPE_INT, TYPE_REAL, TYPE_VECTOR2, TYPE_VECTOR3:
			printraw(object)
		TYPE_STRING:
			printraw('"%s"' % object)
		TYPE_OBJECT:
			printraw("{\n")
			for prop in object.get_property_list():
				if prop.usage != PROPERTY_USAGE_SCRIPT_VARIABLE:
					continue
				if prop.name.begins_with("_"):
					continue
				var k = prop.name
				var v = object.get(k)
				dump(k, indent, level + 1, ": ")
				dump(v, indent, level + 1, ",\n", false)
			printraw(indent.repeat(level))
			printraw("}")
		TYPE_DICTIONARY:
			if object:
				printraw("{\n")
				for k in object:
					var v = object[k]
					dump(k, indent, level + 1, ": ")
					dump(v, indent, level + 1, ",\n", false)
				printraw(indent.repeat(level))
				printraw("}")
			else:
				printraw("{}")
		TYPE_ARRAY:
			if object:
				printraw("[\n")
				for v in object:
					dump(v, indent, level + 1, ",\n")
				printraw(indent.repeat(level))
				printraw("]")
			else:
				printraw("[]")
		_:
			printraw("<%s>" % object)
	printraw(terminator)


static func dict_filter_keys(dict) -> Array:
	HARD.assert(dict is Dictionary)
	var list := []
	for k in dict:
		if dict[k]:
			list.append(k)
	return list


static func dict_merge(d1, d2) -> Dictionary:
	HARD.assert(d1 is Dictionary)
	HARD.assert(d2 is Dictionary)
	var merged := {}
	for k in d1:
		merged[k] = d1[k]
	for k in d2:
		merged[k] = d2[k]
	return merged


static func node_all_children(node: Node, type_filter: String = "") -> Array:
	var children := []

	for child in node.get_children():
		var child_type: String = child.get_class()
		if child_type.begins_with(type_filter):
			children.append(child)
		if child is Node:
			children += node_all_children(child, type_filter)

	return children


static func path_list(path: String, include_dir := true, sort := true, skip_hidden := true) -> PoolStringArray:
	var diter := Directory.new()
	var error := OK
	var files := []

	path = path.rstrip("/")

	error = diter.open(path)
	HARD.assert(error == OK, 'cannot open directory: "%s"', path)

	error = diter.list_dir_begin(true, skip_hidden)
	HARD.assert(error == OK, 'cannot list directory: "%s"', path)

	var file_name := diter.get_next()
	while file_name != "":
		if include_dir:
			files.append(path + "/" + file_name)
		else:
			files.append(file_name)
		file_name = diter.get_next()

	if sort:
		files.sort()

	return PoolStringArray(files)


static func read_json_from_path(file_path: String) -> Dictionary:
	var file = File.new()
	var file_error = file.open(file_path, File.READ)
	HARD.assert(file_error == OK, "cannot open json file at: %s", [file_path])

	var text = file.get_as_text()
	var json = JSON.parse(text)
	file.close()

	HARD.assert(
		json.error == OK,
		"cannot read json file at: %s; parse error: %s",
		[file_path, json.error_string]
	)

	return json.result


static func write_json_to_path(file_path: String, data: Dictionary) -> void:
	var file = File.new()
	var file_error = file.open(file_path, File.WRITE)
	HARD.assert(file_error == OK, "cannot open file for writing json at: %s", [file_path])
	file.store_string(JSON.print(data))


static func string_hash(string: String, type := HashingContext.HASH_MD5) -> int:
	var error := OK
	var bytes := string.to_utf8()
	var hash_buf := StreamPeerBuffer.new()
	var hash_ctx := HashingContext.new()
	error = hash_ctx.start(type)
	HARD.assert(error == OK)
	error = hash_ctx.update(bytes)
	HARD.assert(error == OK)
	hash_buf.data_array = hash_ctx.finish()
	var hash_int := 0
	var hash_len := 8 * hash_buf.get_available_bytes()
	while hash_len > 0:
		var j := 0
		var k := 0
		for i in [64, 32, 16, 8]:
			if hash_len >= i:
				j = i
				break
		match j:
			64:
				k = hash_buf.get_64()
			32:
				k = hash_buf.get_32()
			16:
				k = hash_buf.get_16()
			8:
				k = hash_buf.get_8()
		hash_int ^= k
		hash_len -= j
	return hash_int


static func record_hash(bytes: PoolByteArray) -> PoolByteArray:
	var error := OK
	var hash_ctx := HashingContext.new()
	error = hash_ctx.start(HashingContext.HASH_MD5)
	HARD.assert(error == OK)
	error = hash_ctx.update(bytes)
	HARD.assert(error == OK)
	var hash_buf := hash_ctx.finish()
	return hash_buf


static func vec3_box_clamp(vector: Vector3, vmin: Vector3, vmax: Vector3) -> Vector3:
	for i in range(3):
		vector[i] = clamp(vector[i], vmin[i], vmax[i])
	return vector


static func vec3_sphere_clamp(vector: Vector3, max_length: float) -> Vector3:
	if vector.length() > max_length:
		return vector.normalized() * max_length
	else:
		return vector


static func vec3_range_lerp(value: Vector3, istart, istop, ostart, ostop) -> Vector3:
	if (
		typeof(istart) == TYPE_VECTOR3
		and typeof(istop) == TYPE_VECTOR3
		and typeof(ostart) == TYPE_VECTOR3
		and typeof(ostop) == TYPE_VECTOR3
	):
		return Vector3(
			range_lerp(value.x, istart.x, istop.x, ostart.x, ostop.x),
			range_lerp(value.y, istart.y, istop.y, ostart.y, ostop.y),
			range_lerp(value.z, istart.z, istop.z, ostart.z, ostop.z)
		)
	else:
		return Vector3(
			range_lerp(value.x, istart, istop, ostart, ostop),
			range_lerp(value.y, istart, istop, ostart, ostop),
			range_lerp(value.z, istart, istop, ostart, ostop)
		)


static func normalize(value: Vector3, value_min: float, value_max: float) -> Vector3:
	# normalizes to be in a range of -1 to 1
	return vec3_range_lerp(value, value_min, value_max, -1, 1)


static func update_file_path(path: String) -> String:
	if OS.get_name() == "OSX":
		path = path.replace("/dev/shm/", "/tmp/")
		path = path.replace("/mnt/private/data/", "/tmp/")
	elif OS.get_name() == "Windows":
		path = path.replace("/dev/shm/", "C:/local/temp/")
		path = path.replace("/mnt/private/data/", "C:/local/temp/")
		path = path.replace("/tmp/", "C:/local/temp/")
	return path


static func file_create(current_path: String) -> String:
	var path = update_file_path(current_path)
	var path_dir = path.rsplit("/", true, 1)[0]

	var directory = Directory.new()
	if not directory.file_exists(path):
		var error := Directory.new().make_dir_recursive(path_dir)
		HARD.assert(error == OK, 'cannot create directory: "%s"', path_dir)
	if not directory.file_exists(path):
		var file = File.new()
		var error = file.open(path, File.WRITE)
		file.close()
		HARD.assert(error == OK, 'cannot create file: "%s"', path)
	return path


# TODO: where should these two functions live, exactly?
#	maybe `PlayerHelper`


static func get_delta_quaternion(old: Transform, new: Transform) -> Quat:
	return new.basis.get_rotation_quat() * old.basis.get_rotation_quat().inverse()


static func get_move_delta_position(basis: Basis, delta_position: Vector3) -> Vector3:
	# remove pitch rotation so you don't move slower when you look down or up
	var adjusted_basis = basis.rotated(basis.x, -basis.get_euler().x)
	var position = adjusted_basis * delta_position
	position.y = 0
	return position


static func get_new_hand_position(
	current_hand_position: Vector3,
	relative_delta_hand_position: Vector3,
	old_head_transform: Transform,
	new_head_transfrom: Transform,
	arm_length: float
) -> Vector3:
	var delta_head_quaternion = get_delta_quaternion(old_head_transform, new_head_transfrom)
	var prev_hand_position_relative_to_head = current_hand_position - old_head_transform.origin
	var new_hand_position_relative_to_head_after_head_rotation = delta_head_quaternion.xform(
		prev_hand_position_relative_to_head
	)

	var head_position = new_head_transfrom.origin

	var new_hand_position_relative_to_head = (
		new_hand_position_relative_to_head_after_head_rotation
		+ new_head_transfrom.basis * relative_delta_hand_position
	)

	var new_hand_position = Vector3.ZERO

	# constrain the hand to a sphere around the head with radius = `arm_length`
	if new_hand_position_relative_to_head.length() >= arm_length:
		# TODO this is slightly broken, it moves your hand in y and x instead of just z
		new_hand_position = (
			new_hand_position_relative_to_head_after_head_rotation.normalized() * arm_length
			+ head_position
		)
	else:
		new_hand_position = new_hand_position_relative_to_head + head_position

	return new_hand_position


static func get_new_hand_position_while_climbing(
	current_hand_position: Vector3,
	relative_delta_hand_position: Vector3,
	old_head_transform: Transform,
	new_head_transfrom: Transform,
	arm_length: float
) -> Vector3:
	var prev_hand_position_relative_to_head = current_hand_position - old_head_transform.origin
	var head_position = new_head_transfrom.origin
	var new_hand_position_relative_to_head = (
		prev_hand_position_relative_to_head
		+ new_head_transfrom.basis * relative_delta_hand_position
	)
	# constrain the hand to a sphere around the head with radius = `arm_length`
	if new_hand_position_relative_to_head.length() >= arm_length:
		new_hand_position_relative_to_head = (
			new_hand_position_relative_to_head.normalized()
			* arm_length
		)

	return new_hand_position_relative_to_head + head_position


#	NOTE: The following does not work
#	physical_head.angular_velocity = (
#		(target_head.global_transform.basis.get_rotation_quat() * physical_head.global_transform.basis.get_rotation_quat().inverse()).get_euler()
#		/ delta
#	)
#	so we stole this implementation from somewhere on the internet: https://www.reddit.com/r/godot/comments/q1lawy/basis_and_angular_velocity_question/
static func calc_angular_velocity_from_basis(from_basis: Basis, to_basis: Basis) -> Vector3:
	var q1 = from_basis.get_rotation_quat()
	var q2 = to_basis.get_rotation_quat()

	# Quaternion that transforms q1 into q2
	var qt = q2 * q1.inverse()

	# deal with the discontinuity between 0 and 2pi
	if qt.w < 0:
		qt = Quat(-qt.x, -qt.y, -qt.z, -qt.w)

	# Angle from quaternion
	var angle = 2 * acos(clamp(qt.w, -1, 1))

	# Prevent divide by zero
	if angle < 0.0001:
		return Vector3.ZERO

	# Axis from quaternion
	var axis = Vector3(qt.x, qt.y, qt.z) / sqrt(1 - qt.w * qt.w)

	return axis * angle


static func vec_rad2deg(angle: Vector3) -> Vector3:
	return angle * 180 / PI


static func vec_deg2rad(angle: Vector3) -> Vector3:
	return angle * PI / 180


static func is_terrain(node: Node) -> bool:
	return is_instance_valid(node) and node.get_parent().name == "terrain_collision_meshes"


static func is_tree(node: Node) -> bool:
	if not is_instance_valid(node):
		return false
	return (
		node.get_parent().name == "tree_collision_meshes"
		or node.name.begins_with("fruit_tree_normal")
	)


static func is_static(node: Node) -> bool:
	return (
		node is StaticBody
		or node is RigidBody and node.mode == RigidBody.MODE_STATIC
		or is_terrain(node)
	)


static func is_physical_player(node: Node) -> bool:
	return is_instance_valid(node) and node.get_parent().name == "physical_player"


static func clamp_delta_rotation(current_rotation: float, delta_rotation: float, max_rotation: float):
	var clamped_rotation = clamp(delta_rotation + current_rotation, -max_rotation, max_rotation)
	return clamped_rotation - current_rotation


# from https://github.com/binogure-studio/godot-uuid/blob/master/uuid.gd
static func _get_random_int_for_uuid():
	# Randomize every time to minimize the risk of collisions
	randomize()
	return randi() % 256


static func uuidv4():
	# 16 random bytes with the bytes on index 6 and 8 modified
	var bytes = [
		# low
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		#
		# mid
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		#
		# hi
		((_get_random_int_for_uuid()) & 0x0f) | 0x40,
		_get_random_int_for_uuid(),
		#
		# clock
		((_get_random_int_for_uuid()) & 0x3f) | 0x80,
		_get_random_int_for_uuid(),
		#
		# clock
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
		_get_random_int_for_uuid(),
	]
	return "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x" % bytes


static func set_owner_of_subtree(
	owner: Node, current_children: Array, is_saving_scene_subnodes: bool = true
) -> void:
	for _node in current_children:
		var node: Node = _node
		node.set_owner(owner)
		var is_packed_scene_root = not node.filename.empty()
		if is_saving_scene_subnodes:
			# Prevents 2x instantiation of subnodes
			node.filename = ""
		elif is_packed_scene_root:
			continue
		set_owner_of_subtree(owner, node.get_children(), is_saving_scene_subnodes)
