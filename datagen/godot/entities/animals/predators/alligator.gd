extends Predator
class_name Alligator

export var player := Vector2(2, 2)

export var out_of_reach_height = 4
export var required_movement_distance = 2
export var give_up_after_hops_without_progress = 32

export var inactive_speed := Vector2(2, 2)
export var inactive_movement_frequency: float = 1.0 / 16
export var inactive_movement_steps := 4

export var active_chase_speed := Vector2(2, 2)
export var active_chase_steps := 6
export var active_rest_steps := 4

var activation_criteria: Array


func _ready():
	activation_criteria = [
		PlayerWithinDetectionZone.new(),
		PlayerReachableByGround.new(
			out_of_reach_height, required_movement_distance, give_up_after_hops_without_progress
		),
	]
	inactive_behavior = HopRandomly.new(
		_rng_key("inactive"), inactive_movement_frequency, inactive_speed, inactive_movement_steps
	)
	active_behavior = PursueAndAttackPlayer.new(
		HopInDirection.new(
			TOWARDS_PLAYER, active_chase_speed, active_chase_steps, active_rest_steps
		)
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"), active_chase_steps, inactive_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if BehaviorCriteria.all_match(self, activation_criteria):
		return active_behavior
	return inactive_behavior
