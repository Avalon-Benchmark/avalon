extends Predator
class_name Bee

export var territory_radius := 15
export var retreat_within_radius := 10

export var inactive_full_speed := 2.0
export var inactive_turn_speed := 0.5
export var inactive_turn_frequency: float = 1.0 / 16

export var active_chase_full_speed := 2.75
export var active_chase_turn_speed := 1.0
export var active_chase_turn_rotation_speed := 1.5

var return_to_territory: ReturnToTerritoryBehavior


func _ready():
	return_to_territory = load_or_init(
		"return_to_territory",
		ReturnToTerritoryBehavior.new().init(
			FlyInDirection.new().init(TOWARDS_TERRITORY, inactive_full_speed, inactive_turn_speed),
			global_transform.origin,
			territory_radius,
			retreat_within_radius
		)
	)
	set_inactive(
		FlyRandomly.new().init(_rng_key("inactive"), inactive_turn_frequency, inactive_full_speed)
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
			inactive_full_speed,
			active_chase_turn_rotation_speed
		)
	)


func select_next_behavior() -> AnimalBehavior:
	if active_behavior.is_resting_after_attack(self):
		if HARD.mode():
			print("%s died from attacking" % self)
		_die()

	if _is_player_in_detection_radius:
		return active_behavior

	if return_to_territory.is_returning_to_territory(self):
		return return_to_territory

	return inactive_behavior
