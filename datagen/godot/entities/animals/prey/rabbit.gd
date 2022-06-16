extends Animal
class_name Rabbit

export var inactive_speed := Vector2(4, 1.25)
export var inactive_movement_frequency: float = 1.0 / 8
export var inactive_movement_steps := 4

export var active_flee_speed := Vector2(6, 2.25)
export var active_flee_steps := 4
# TODO rename "active_rest_steps" to "active_rest_frames." It takes a rabbit ~ 3 frames to hop
export var active_rest_steps := 3

var in_detection_zone: PlayerWithinDetectionZone


func _ready():
	in_detection_zone = PlayerWithinDetectionZone.new()
	inactive_behavior = HopRandomly.new(
		_rng_key("inactive"), inactive_movement_frequency, inactive_speed, inactive_movement_steps
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
