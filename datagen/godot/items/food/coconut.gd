extends Food

class_name Coconut

# TODO make higher after testing
export var open_on_impact_velocity_threshold := 5.0


func _ready():
	_validate_openable()


func is_edible() -> bool:
	return _is_open()


func _on_body_entered(body: Node):
	if not is_edible() and _is_impact_velocity_sufficient(body, open_on_impact_velocity_threshold):
		_open()
	._on_body_entered(body)
