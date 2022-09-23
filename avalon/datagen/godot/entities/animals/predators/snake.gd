extends Predator
class_name Snake

export var inactive_avoid_speed := Vector2(1.5, 1.5)
export var inactive_avoid_hops := 4
export var inactive_rest_frames := 8

export var active_within_threshold := 3.0

export var active_chase_speed := Vector2(4, 1.5)
export var active_chase_hops := 8
export var active_rest_frames := 4

var activation_criteria: ActivationGate


func _ready():
	var player_in_detection_radius := PlayerInDetectionRadius.new()
	var initial_activation_criteria := [
		player_in_detection_radius,
		PlayerWithinThreshold.new(active_within_threshold),
	]
	activation_criteria = ActivationGate.new([], initial_activation_criteria)

	inactive_behavior = ConditionalBehavior.new(
		[player_in_detection_radius],
		HopInDirection.new(
			AWAY_FROM_PLAYER, inactive_avoid_speed, inactive_avoid_hops, inactive_rest_frames
		)
	)

	active_behavior = PursueAndAttackPlayer.new(
		HopInDirection.new(
			TOWARDS_PLAYER, active_chase_speed, active_chase_hops, active_rest_frames
		)
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_chase_hops, inactive_avoid_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
