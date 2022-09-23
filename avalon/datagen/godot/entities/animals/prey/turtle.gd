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
	inactive_behavior = HopInCircle.new(
		inactive_speed, inactive_hops_per_side, inactive_frames_per_turn, inactive_turn_angle
	)
	active_behavior = HopInDirection.new(
		AWAY_FROM_PLAYER, active_flee_speed, active_flee_hops, active_rest_frames
	)
	avoid_ocean_behavior = AvoidOcean.new(_rng_key("avoid_ocean"), active_flee_hops, inactive_speed)


func select_next_behavior() -> AnimalBehavior:
	if is_player_in_detection_radius:
		return active_behavior

	return inactive_behavior
