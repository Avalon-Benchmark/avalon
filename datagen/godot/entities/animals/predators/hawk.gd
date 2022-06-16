extends Predator
class_name Hawk

export var territory_radius := 15
export var retreat_within_radius := 10

export var inactive_speed := 2.0
export var inactive_steps_per_side := 8
export var inactive_steps_per_turn := 8
export var inactive_turn_angle := 30

export var active_chase_full_speed := 3.25
export var active_chase_turn_speed := 1.5
export var active_chase_turn_rotation_speed := 1.5

var in_detection_zone: PlayerWithinDetectionZone
var return_to_territory: ReturnToTerritoryBehavior


func _ready():
	in_detection_zone = PlayerWithinDetectionZone.new()
	return_to_territory = ReturnToTerritoryBehavior.new(
		FlyInDirection.new(TOWARDS_TERRITORY, inactive_speed),
		global_transform.origin,
		territory_radius,
		retreat_within_radius
	)

	inactive_behavior = FlyInCircles.new(
		inactive_speed, inactive_steps_per_side, inactive_steps_per_turn, inactive_turn_angle
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
		inactive_speed,
		active_chase_turn_rotation_speed
	)


func select_next_behavior() -> AnimalBehavior:
	if return_to_territory.is_returning_to_territory(self):
		return return_to_territory

	if (
		in_detection_zone.is_matched_by(self)
		and is_player_in_territory()
		and not should_give_respite()
	):
		return active_behavior

	return inactive_behavior


func should_give_respite():
	var pursue = active_behavior as PursueAndAttackPlayer
	return pursue.is_resting_after_attack(self) and not pursue.is_knock_back_sensible(self)


func is_player_in_territory():
	return return_to_territory.is_within_territory(self.get_player_position())
