extends SpecBase

class_name ControlledNodeSpec

var spawn_point_name: String


func get_node() -> ControlledNode:
	return HARD.assert(false, "Must be overridden")


func get_input_collector() -> InputCollector:
	return HARD.assert(false, "Must be overridden")
