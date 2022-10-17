extends Animal
class_name Turtle

export var inactive_speed := Vector2(2, 0.75)
export var inactive_hops_per_side := 8
export var inactive_frames_per_turn := 8
export var inactive_turn_angle := 30

export var active_flee_speed := Vector2(2.5, 1.0)
export var active_flee_hops := 6
export var active_rest_frames := 4


func _ready():
	set_inactive(
		HopInCircle.new().init(
			inactive_speed, inactive_hops_per_side, inactive_frames_per_turn, inactive_turn_angle
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
