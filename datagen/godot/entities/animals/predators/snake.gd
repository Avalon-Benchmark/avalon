extends Predator
class_name Snake

export var inactive_avoid_speed := Vector2(1.5, 1.5)
export var inactive_avoid_steps := 4
export var inactive_rest_steps := 8

export var active_within_threshold := 3.0

export var active_chase_speed := Vector2(4, 1.5)
export var active_chase_steps := 8
export var active_rest_steps := 4

var activation_criteria: ActivationGate


func _ready():
	var in_detection_zone := PlayerWithinDetectionZone.new()
	var initial_activation_criteria := [
		in_detection_zone,
		PlayerWithinThreshold.new(active_within_threshold),
	]
	activation_criteria = ActivationGate.new([], initial_activation_criteria)

	inactive_behavior = ConditionalBehavior.new(
		[in_detection_zone],
		HopInDirection.new(
			AWAY_FROM_PLAYER, inactive_avoid_speed, inactive_avoid_steps, inactive_rest_steps
		)
	)

	active_behavior = PursueAndAttackPlayer.new(
		HopInDirection.new(
			TOWARDS_PLAYER, active_chase_speed, active_chase_steps, active_rest_steps
		)
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_chase_steps, inactive_avoid_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
