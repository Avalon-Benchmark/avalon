extends Animal
class_name Mouse

export var inactive_speed := Vector2(3.25, 1)
export var inactive_movement_frequency: float = 1.0 / 10
export var inactive_movement_steps := 8

export var active_flee_speed := Vector2(6.5, 1)
# TODO rename "active_rest_steps" to "active_rest_frames." It takes a rabbit ~ 3 frames to hop
export var active_flee_steps := 6
export var active_rest_steps := 3

export var climbing_speed := 12
export var climbing_escape_hop_speed := Vector2(6, 8)
export var required_grounded_movement_distance := 2
export var climb_after_hops_without_progress := 3

var activation_criteria: ActivationGate


func _ready():
	var active_criteria := [PlayerWithinDetectionZone.new()]
	var initial_criteria = [CanSeePlayer.new()]
	# TODO does not deactivate when sight is lost because it would never actually flee
	activation_criteria = ActivationGate.new(active_criteria, initial_criteria)

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
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
