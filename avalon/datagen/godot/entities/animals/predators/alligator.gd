extends Predator
class_name Alligator

export var player := Vector2(2, 2)

export var out_of_reach_height = 4
export var required_movement_distance = 2
export var give_up_after_hops_without_progress = 32

export var inactive_speed := Vector2(2, 2)
export var inactive_movement_frequency: float = 1.0 / 16
export var inactive_movement_hops := 4

export var active_chase_speed := Vector2(2, 2)
export var active_chase_hops := 6
export var active_rest_frames := 4

var reachable_by_ground: PlayerReachableByGround


func _ready():
	reachable_by_ground = load_or_init(
		"reachable_by_ground",
		PlayerReachableByGround.new().init(
			out_of_reach_height, required_movement_distance, give_up_after_hops_without_progress
		)
	)
	set_inactive(
		HopRandomly.new().init(
			_rng_key("inactive"),
			inactive_movement_frequency,
			inactive_speed,
			inactive_movement_hops
		)
	)
	set_active(
		PursueAndAttackPlayer.new().init(
			HopInDirection.new().init(
				TOWARDS_PLAYER, active_chase_speed, active_chase_hops, active_rest_frames
			)
		)
	)
	set_avoid_ocean(
		AvoidOcean.new().init(_rng_key("avoid_ocean"), active_chase_hops, inactive_speed)
	)


func select_next_behavior() -> AnimalBehavior:
	if _is_player_in_detection_radius and reachable_by_ground.is_matched_by(self):
		return active_behavior
	return inactive_behavior
