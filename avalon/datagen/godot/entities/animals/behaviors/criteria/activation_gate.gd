extends BehaviorCriteria

class_name ActivationGate

export var is_active: bool = false

var active_criteria: Array
var initial_criteria: Array


func get_logic_nodes() -> Array:
	var criteria = []
	criteria.append_array(active_criteria)
	criteria.append_array(initial_criteria)
	return criteria


func init(_activation_criteria: Array, _initial_criteria: Array = []) -> BehaviorCriteria:
	active_criteria = _activation_criteria
	initial_criteria = _initial_criteria
	return self


func _ready():
	active_criteria = LogicNodes.prefer_persisted_array(self, "active", active_criteria)
	initial_criteria = LogicNodes.prefer_persisted_array(self, "initial", initial_criteria)


func is_matched_by(animal: Animal) -> bool:
	var is_start_or_continue_matched = BehaviorCriteria.all_match(animal, active_criteria)

	if is_active or not is_start_or_continue_matched:
		is_active = is_start_or_continue_matched
		return is_active

	is_active = BehaviorCriteria.all_match(animal, initial_criteria)
	return is_active
