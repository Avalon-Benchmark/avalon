extends Animal
class_name Deer

export var inactive_avoid_speed := Vector2(1.5, 1.25)
export var inactive_avoid_steps := 4
export var inactive_rest_steps := 8

export var active_within_threshold := 6.0

export var active_flee_speed := Vector2(5.5, 2.25)
# TODO rename "active_rest_steps" to "active_rest_frames." It takes a rabbit ~ 3 frames to hop
export var active_flee_steps := 3
export var active_rest_steps := 4

var activation_criteria: ActivationGate


func _ready():
	var in_detection_zone := PlayerWithinDetectionZone.new()
	var initial_activation_criteria := [PlayerWithinThreshold.new(active_within_threshold)]
	activation_criteria = ActivationGate.new([in_detection_zone], initial_activation_criteria)

	inactive_behavior = ConditionalBehavior.new(
		[in_detection_zone],
		HopInDirection.new(
			AWAY_FROM_PLAYER, inactive_avoid_speed, inactive_avoid_steps, inactive_rest_steps
		)
	)

	active_behavior = HopInDirection.new(
		AWAY_FROM_PLAYER, active_flee_speed, active_flee_steps, active_rest_steps
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_flee_steps, active_flee_speed / 2
	)


func select_next_behavior() -> AnimalBehavior:
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
