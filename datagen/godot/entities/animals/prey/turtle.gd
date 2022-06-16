extends Animal
class_name Turtle

export var inactive_speed := Vector2(2, 0.75)
export var inactive_steps_per_side := 8
export var inactive_steps_per_turn := 8
export var inactive_turn_angle := 30

export var active_flee_speed := Vector2(2.5, 1.0)
export var active_flee_steps := 6
export var active_rest_steps := 4

var in_detection_zone: PlayerWithinDetectionZone


func _ready():
	in_detection_zone = PlayerWithinDetectionZone.new()
	inactive_behavior = HopInCircle.new(
		inactive_speed, inactive_steps_per_side, inactive_steps_per_turn, inactive_turn_angle
	)
	active_behavior = HopInDirection.new(
		AWAY_FROM_PLAYER, active_flee_speed, active_flee_steps, active_rest_steps
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_flee_steps, inactive_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if in_detection_zone.is_matched_by(self):
		return active_behavior

	return inactive_behavior
