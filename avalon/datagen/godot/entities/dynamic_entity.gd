extends RigidBody

class_name DynamicEntity

export var entity_id: int = -1


func _ready():
	HARD.assert(
		self.entity_id != -1,
		"Must set item ids. Used for debugging. If you dont care just set to 0"
	)
	if HARD.mode() and _scale_perturbation() >= _acceptable_scale_perturbation():
		var failure_desc = "%s.scale.length() == %s" % [self, _scale_perturbation()]
		var message = (
			"Exiting due to unsupported direct scale setting (%s). " % failure_desc
			+ "Items should have scale set via safe_scale. "
			+ "If this error occurred while loading a snapshot, "
			+ "the item may have had it's transform perturbed during physics."
		)
		HARD.stop(message)


func get_class() -> String:
	return "DynamicEntity"


func to_dict() -> Dictionary:
	var data = {}
	data["name"] = name
	data["id"] = entity_id
	data["class"] = get_class()
	data["script"] = get_script().resource_path
	data["position"] = translation
	data["velocity"] = linear_velocity
	data["rotation"] = rotation_degrees
	return data


func _scale_perturbation() -> float:
	return (self.scale - Vector3.ONE).length()


func _acceptable_scale_perturbation() -> float:
	return 0.00001
