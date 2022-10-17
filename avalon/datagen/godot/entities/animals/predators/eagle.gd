extends Predator
class_name Eagle

export var noticed_by_player_switch_steps := 10

export var inactive_speed := 3.0
export var inactive_turn_frequency: float = 1.0 / 8

export var active_chase_full_speed := 3.75
export var active_chase_turn_speed := 1.5
export var active_chase_turn_rotation_speed := 2.0

var noticed_by_player: NoticedByPlayer


func _ready():
	noticed_by_player = load_or_init(
		"noticed_by_player", NoticedByPlayer.new().init(noticed_by_player_switch_steps)
	)
	noticed_by_player.connect_visibility_notifier($visibility_notifier)
	set_inactive(
		FlyRandomly.new().init(_rng_key("inactive"), inactive_turn_frequency, inactive_speed)
	)
	set_active(
		PursueAndAttackPlayer.new().init(
			FlyInDirection.new().init(
				TOWARDS_PLAYER,
				active_chase_full_speed,
				active_chase_turn_speed,
				active_chase_turn_rotation_speed
			)
		)
	)
	set_avoid_ocean(
		AvoidOcean.new().init(
			_rng_key("avoid_ocean"),
			AvoidOcean.FLY_STEPS,
			inactive_speed,
			active_chase_turn_rotation_speed
		)
	)


func select_next_behavior() -> AnimalBehavior:
	var is_noticed = noticed_by_player.is_matched_by(self)
	if (not is_noticed) and _is_player_in_detection_radius:
		return active_behavior
	return inactive_behavior
