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

var in_detection_zone: PlayerWithinDetectionZone
var return_to_territory: ReturnToTerritoryBehavior


func _ready():
	in_detection_zone = PlayerWithinDetectionZone.new()
	return_to_territory = ReturnToTerritoryBehavior.new(
		FlyInDirection.new(TOWARDS_TERRITORY, inactive_full_speed, inactive_turn_speed),
		global_transform.origin,
		territory_radius,
		retreat_within_radius
	)
	inactive_behavior = FlyRandomly.new(
		_rng_key("inactive"), inactive_turn_frequency, inactive_full_speed
	)
	active_behavior = PursueAndAttackPlayer.new(
		FlyInDirection.new(
			TOWARDS_PLAYER,
			active_chase_full_speed,
			active_chase_turn_speed,
			active_chase_turn_rotation_speed
		)
	)
	avoid_ocean_behavior = AvoidOcean.new(
		_rng_key("avoid_ocean"),
		AvoidOcean.FLY_STEPS,
		inactive_full_speed,
		active_chase_turn_rotation_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if active_behavior.is_resting_after_attack(self):
		if HARD.mode():
			print("%s died from attacking" % self)
		_die()

	if in_detection_zone.is_matched_by(self):
		return active_behavior

	if return_to_territory.is_returning_to_territory(self):
		return return_to_territory

	return inactive_behavior
