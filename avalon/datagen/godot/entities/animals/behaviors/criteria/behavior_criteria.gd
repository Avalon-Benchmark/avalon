extends Node

class_name BehaviorCriteria


func is_matched_by(_animal: Animal) -> bool:
	HARD.assert(false, "AnimalBehaviorCriteria must implement is_matched_by")
	return false


static func all_match(animal: Animal, criteria: Array) -> bool:
	for c in criteria:
		var is_matched = c.is_matched_by(animal)
		if not is_matched:
			return false
	return true


func describe():
	return script_name()


func script_name():
	var path: Array = get_script().resource_path.split("/")
	var file = path[len(path) - 1]
	return file.split(".")[0]
