extends Reference

class_name ClassBuilder

# TODO: ProjectSettings need to be manually updated by the Godot editor

const _class_dict = {}


static func get_object_from_json(json_object, config_path: String):
	match typeof(json_object):
		TYPE_DICTIONARY:
			if json_object.has("$type"):
				var blank = new_object_from_type(json_object["$type"])
				if typeof(blank) == TYPE_VECTOR2:
					blank.x = json_object["x"]
					blank.y = json_object["y"]
					return blank
				elif typeof(blank) == TYPE_VECTOR3:
					blank.x = json_object["x"]
					blank.y = json_object["y"]
					blank.z = json_object["z"]
					return blank
				for k in json_object:
					if k == "$type":
						continue
					blank.set(k, get_object_from_json(json_object[k], config_path + "/" + k))
				blank._config_path = config_path
				if blank.has_method("_post_init"):
					blank._post_init()
				return blank
			else:
				var dict = {}
				for k in json_object:
					dict[k] = get_object_from_json(json_object[k], config_path + "/" + k)
				return dict
		TYPE_ARRAY:
			var list = []
			var i = 0
			for v in json_object:
				list.append(get_object_from_json(v, config_path + "/" + str(i)))
				i += 1
			return list
		TYPE_REAL:
			if int(json_object) == json_object:
				return int(json_object)
			return json_object
		TYPE_BOOL, TYPE_STRING, TYPE_NIL:
			return json_object
		_:
			return HARD.stop("can't handle this json object")


static func new_object_from_type(name: String):
	_init_class_dict()
	match name:
		"Vector2":
			return Vector2()
		"Vector3":
			return Vector3()
	return _class_dict[name].new()


static func new_class_instance(spec):
	return get_class_resource(spec).new()


static func get_class_resource(spec):
	_init_class_dict()
	var _class_dict_copy = _class_dict.duplicate()
	var script_path: String = spec.script.resource_path
	var file_name = script_path.split("/")[-1]
	var klass = file_name.replace("Spec.gd", "")
	return _class_dict[klass]


static func _init_class_dict():
	if not _class_dict:
		for entry in ProjectSettings.get_setting("_global_script_classes"):
			_class_dict[entry["class"]] = ResourceLoader.load(entry["path"])


# TODO: this should be automatic
static func clear():
	_class_dict.clear()
