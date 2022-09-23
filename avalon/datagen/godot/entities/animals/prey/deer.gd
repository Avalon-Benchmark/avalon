extends Animal
class_name Deer

export var inactive_avoid_speed := Vector2(1.5, 1.25)
export var inactive_avoid_hops := 4
export var inactive_rest_frames := 8

export var active_within_threshold := 6.0

export var active_flee_speed := Vector2(5.5, 2.25)
export var active_flee_hops := 3
export var active_rest_frames := 4

var activation_criteria: ActivationGate


func _ready():
	var in_detection_zone := PlayerInDetectionRadius.new()
	var initial_activation_criteria := [PlayerWithinThreshold.new(active_within_threshold)]
	activation_criteria = ActivationGate.new([in_detection_zone], initial_activation_criteria)

	inactive_behavior = ConditionalBehavior.new(
		[in_detection_zone],
		HopInDirection.new(
			AWAY_FROM_PLAYER, inactive_avoid_speed, inactive_avoid_hops, inactive_rest_frames
		)
	)

	active_behavior = HopInDirection.new(
		AWAY_FROM_PLAYER, active_flee_speed, active_flee_hops, active_rest_frames
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_flee_hops, active_flee_speed / 2
	)


func select_next_behavior() -> AnimalBehavior:
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
