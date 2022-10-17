extends Animal
class_name Pigeon

export var territory_radius := 15
export var retreat_within_radius := 10

export var inactive_speed := 2.0
export var inactive_turn_frequency: float = 1.0 / 16

export var active_flee_full_speed := 2.75
export var active_flee_turn_speed := 1.25
export var active_flee_turn_rotation_speed := 1.5

var return_to_territory: ReturnToTerritoryBehavior


func _ready():
	return_to_territory = load_or_init(
		"return_to_territory",
		ReturnToTerritoryBehavior.new().init(
			FlyInDirection.new().init(TOWARDS_TERRITORY, inactive_speed),
			global_transform.origin,
			territory_radius,
			retreat_within_radius
		)
	)
	set_inactive(
		FlyRandomly.new().init(_rng_key("inactive"), inactive_turn_frequency, inactive_speed)
	)
	set_active(
		FlyInDirection.new().init(
			AWAY_FROM_PLAYER,
			active_flee_full_speed,
			active_flee_turn_speed,
			active_flee_turn_rotation_speed
		)
	)
	set_avoid_ocean(
		AvoidOcean.new().init(
			_rng_key("avoid_ocean"),
			AvoidOcean.FLY_STEPS,
			active_flee_full_speed,
			active_flee_turn_speed
		)
	)


func select_next_behavior() -> AnimalBehavior:
	if _is_player_in_detection_radius:
		return active_behavior
	if return_to_territory.is_returning_to_territory(self):
		return return_to_territory
	return inactive_behavior
