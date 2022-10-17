extends Weapon

class_name Rock

export var throw_velocity_threshold := 1.0
export var natural_fall_velocity_threshold := 10.0

export var relative_impact_velocity_threshold := 0.5

export var is_thrown := false


func is_mid_air() -> bool:
	return not is_held and ground_contact_count == 0


func release() -> void:
	.release()
	is_thrown = is_mid_air()


func _ready():
	max_damping = 2.0
	angular_damp_factor = 1.0
	linear_damp_factor = 1.0
	ground_contact_damp_factor = 1.0


func _on_body_entered(body: Node):
	._on_body_entered(body)
	if is_thrown:
		is_thrown = is_mid_air()


func is_dangerous():
	if not is_mid_air():
		return false
	var speed = linear_velocity.length()
	return (
		(is_thrown and speed >= throw_velocity_threshold)
		or (speed >= natural_fall_velocity_threshold)
	)


func get_inflicted_damage(target_velocity: Vector3) -> float:
	if not is_dangerous():
		return 0.0

	var impact_magnitude = _calculate_impact_magnitude(target_velocity)
	if impact_magnitude < relative_impact_velocity_threshold:
		if HARD.mode():
			var impact_desc = "(%s < %s)" % [impact_magnitude, relative_impact_velocity_threshold]
			var absolute = "(full velocity: %s)" % linear_velocity.length()
			print("%s's relative velocity %s too slow %s" % [self, impact_desc, absolute])
		return 0.0

	return .get_inflicted_damage(target_velocity)
