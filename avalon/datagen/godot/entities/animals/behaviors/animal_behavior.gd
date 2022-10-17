extends Node

class_name AnimalBehavior


func do(_animal: Animal, _delta: float) -> Vector3:
	HARD.assert(
		false,
		"AnimalBehaviors must implement do, which should return the observed_velocity from a animal.controller method"
	)
	return Vector3.ZERO


func reset() -> void:
	pass


func describe():
	return script_name()


func script_name():
	var path: Array = get_script().resource_path.split("/")
	var file = path[len(path) - 1]
	return file.split(".")[0]
