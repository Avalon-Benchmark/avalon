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
	var continued_activation_criteria := [PlayerInDetectionRadius.new()]
	var initial_activation_criteria := [PlayerWithinThreshold.new().init(active_within_threshold)]
	activation_criteria = load_or_init(
		"activation_criteria",
		ActivationGate.new().init(continued_activation_criteria, initial_activation_criteria)
	)

	set_inactive(
		ConditionalBehavior.new().init(
			[PlayerInDetectionRadius.new()],
			HopInDirection.new().init(
				AWAY_FROM_PLAYER, inactive_avoid_speed, inactive_avoid_hops, inactive_rest_frames
			)
		)
	)

	set_active(
		HopInDirection.new().init(
			AWAY_FROM_PLAYER, active_flee_speed, active_flee_hops, active_rest_frames
		)
	)

	set_avoid_ocean(
		AvoidOcean.new().init(_rng_key("avoid_ocean"), active_flee_hops, active_flee_speed / 2)
	)


func select_next_behavior() -> AnimalBehavior:
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
