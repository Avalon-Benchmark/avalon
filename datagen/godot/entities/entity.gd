extends Spatial

class_name Entity

export var entity_id: int = -1


func _ready():
	HARD.assert(
		(self.scale - Vector3.ONE).length() <= 0.00001,
		"Scaling objects like this is not supported. See docs"
	)
	HARD.assert(
		self.entity_id != -1,
		"Must set item ids (%s). Used for debugging. If you dont care just set to 0" % self
	)


func get_class() -> String:
	return "Entity"


func to_dict() -> Dictionary:
	var data = {}
	data["name"] = name
	data["id"] = entity_id
	data["class"] = get_class()
	data["script"] = get_script().resource_path
	data["position"] = translation
	data["rotation"] = rotation_degrees
	return data
