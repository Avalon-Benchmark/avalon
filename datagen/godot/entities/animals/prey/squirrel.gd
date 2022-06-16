extends Animal
class_name Squirrel

export var inactive_speed := Vector2(2, 2)
export var inactive_movement_frequency: float = 1.0 / 8
export var inactive_movement_steps := 4

export var active_flee_speed := Vector2(5, 2)
# TODO rename "active_rest_steps" to "active_rest_frames." It takes a rabbit ~ 3 frames to hop
export var active_flee_steps := 6
export var active_rest_steps := 4

export var climbing_speed := 12
export var climbing_escape_hop_speed := Vector2(6, 8)
export var required_grounded_movement_distance := 2
export var climb_after_hops_without_progress := 3

var in_detection_zone: PlayerWithinDetectionZone


func _ready():
	in_detection_zone = PlayerWithinDetectionZone.new()

	inactive_behavior = ClimbWhenNecessary.new(
		climbing_speed,
		climbing_escape_hop_speed,
		HopRandomly.new(
			_rng_key("inactive"),
			inactive_movement_frequency,
			inactive_speed,
			inactive_movement_steps
		)
	)

	var stay_grounded_when_making_headway = AbleToMakeHeadway.new(
		required_grounded_movement_distance, climb_after_hops_without_progress
	)

	active_behavior = ClimbWhenNecessary.new(
		climbing_speed,
		climbing_escape_hop_speed,
		HopInDirection.new(
			AWAY_FROM_PLAYER, active_flee_speed, active_flee_steps, active_rest_steps
		),
		stay_grounded_when_making_headway
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_flee_steps, inactive_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if in_detection_zone.is_matched_by(self):
		return active_behavior
	return inactive_behavior
