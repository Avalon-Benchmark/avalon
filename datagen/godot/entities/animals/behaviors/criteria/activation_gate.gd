extends BehaviorCriteria

class_name ActivationGate

var is_active: bool = false

var active_criteria: Array
var initial_criteria: Array


func _init(_activation_criteria: Array, _initial_criteria: Array = []):
	active_criteria = _activation_criteria
	initial_criteria = _initial_criteria


func is_matched_by(animal: Animal) -> bool:
	var is_start_or_continue_matched = BehaviorCriteria.all_match(animal, active_criteria)

	if is_active or not is_start_or_continue_matched:
		is_active = is_start_or_continue_matched
		return is_active

	is_active = BehaviorCriteria.all_match(animal, initial_criteria)
	return is_active
