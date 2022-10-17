extends Animal
class_name Rabbit

export var inactive_speed := Vector2(4, 1.25)
export var inactive_movement_frequency: float = 1.0 / 8
export var inactive_movement_hops := 4

export var active_flee_speed := Vector2(6, 2.25)
export var active_flee_hops := 4
export var active_rest_frames := 3


func _ready():
	set_inactive(
		HopRandomly.new().init(
			_rng_key("inactive"),
			inactive_movement_frequency,
			inactive_speed,
			inactive_movement_hops
		)
	)
	set_active(
		HopInDirection.new().init(
			AWAY_FROM_PLAYER, active_flee_speed, active_flee_hops, active_rest_frames
		)
	)
	set_avoid_ocean(
		AvoidOcean.new().init(_rng_key("avoid_ocean"), active_flee_hops, inactive_speed)
	)


func select_next_behavior() -> AnimalBehavior:
	if _is_player_in_detection_radius:
		return active_behavior
	return inactive_behavior
