extends Predator
class_name Jaguar

export var movement_detection_threshold := 0.1
export var give_up_after_stillness_steps := 16

export var inactive_speed := Vector2(3, 2)
export var inactive_hops_per_side := 4
export var inactive_frames_per_turn := 4
export var inactive_turn_angle := 45

export var active_chase_speed := Vector2(4.5, 2)
export var active_turn_speed := 2.5
export var active_chase_hops := 6
export var active_rest_frames := 3

export var climbing_speed := 12
export var climbing_escape_hop_speed := Vector2(6, 8)
export var required_grounded_movement_distance := 2
export var climb_after_hops_without_progress := 3

var activation_criteria: ActivationGate


func _ready():
	var continued_activation_criteria = [
		PlayerInDetectionRadius.new(),
		NoticesPlayerMoving.new().init(movement_detection_threshold, give_up_after_stillness_steps),
	]
	var initial_activation_criteria = [CanSeePlayer.new()]
	activation_criteria = load_or_init(
		"activation_criteria",
		ActivationGate.new().init(continued_activation_criteria, initial_activation_criteria)
	)

	set_inactive(
		ClimbWhenNecessary.new().init(
			climbing_speed,
			climbing_escape_hop_speed,
			HopInCircle.new().init(
				inactive_speed,
				inactive_hops_per_side,
				inactive_frames_per_turn,
				inactive_turn_angle
			)
		)
	)

	var stay_grounded_when_making_headway = AbleToMakeHeadway.new().init(
		required_grounded_movement_distance, climb_after_hops_without_progress
	)

	set_active(
		PursueAndAttackPlayer.new().init(
			ClimbWhenNecessary.new().init(
				climbing_speed,
				climbing_escape_hop_speed,
				HopInDirection.new().init(
					TOWARDS_PLAYER,
					active_chase_speed,
					active_chase_hops,
					active_rest_frames,
					active_turn_speed
				),
				stay_grounded_when_making_headway
			)
		)
	)
	set_avoid_ocean(
		AvoidOcean.new().init(_rng_key("avoid_ocean"), active_chase_hops, inactive_speed)
	)


func select_next_behavior() -> AnimalBehavior:
	if activation_criteria.is_matched_by(self):
		return active_behavior
	return inactive_behavior
