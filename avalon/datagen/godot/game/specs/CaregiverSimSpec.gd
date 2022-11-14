extends AvalonSimSpec

class_name CaregiverSimSpec

var caregiver: ControlledNodeSpec


func get_controlled_node_specs() -> Array:
	var nodes = .get_controlled_node_specs()
	nodes.append(caregiver)
	return nodes
