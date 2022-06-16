extends Weapon

class_name Stick


func _ready():
	angular_damp_factor = 0.5


func get_inflicted_damage(impact_velocity: Vector3) -> float:
	if not InteractionHandlers.has_joint(self):
		return 0.0
	return .get_inflicted_damage(impact_velocity)
