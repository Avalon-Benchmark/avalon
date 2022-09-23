extends Predator
class_name Bear

export var inactive_speed := Vector2(2, 2)
export var inactive_movement_frequency: float = 1.0 / 8
export var inactive_movement_hops := 1

export var active_chase_speed := Vector2(4, 1.5)
export var active_chase_hops := 6
export var active_rest_frames := 4

export var climbing_speed := 12
export var climbing_escape_hop_speed := Vector2(6, 12)
export var required_grounded_movement_distance := 2
export var climb_after_hops_without_progress := 3


func _ready():
	inactive_behavior = ClimbWhenNecessary.new(
		climbing_speed,
		climbing_escape_hop_speed,
		HopRandomly.new(
			_rng_key("inactive"),
			inactive_movement_frequency,
			inactive_speed,
			inactive_movement_hops
		)
	)

	var stay_grounded_when_making_headway = AbleToMakeHeadway.new(
		required_grounded_movement_distance, climb_after_hops_without_progress
	)

	active_behavior = PursueAndAttackPlayer.new(
		ClimbWhenNecessary.new(
			climbing_speed,
			climbing_escape_hop_speed,
			HopInDirection.new(
				TOWARDS_PLAYER, active_chase_speed, active_chase_hops, active_rest_frames
			),
			stay_grounded_when_making_headway
		)
	)

	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_chase_hops, inactive_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if is_player_in_detection_radius:
		return active_behavior
	return inactive_behavior
