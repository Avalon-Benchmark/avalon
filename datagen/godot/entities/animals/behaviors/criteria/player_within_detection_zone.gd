# TODO consider doing away with detection zone signal and just using distance
# TODO also consider deleting this class and inlining it everywhere.
# Original intent was to validate radius in the scene, but we don't do that anymore.
extends BehaviorCriteria

class_name PlayerWithinDetectionZone


func is_matched_by(animal: Animal) -> bool:
	return animal.is_player_in_detection_radius or is_player_still_below_climber(animal)


# prevent stuttering while climbing away from player
func is_player_still_below_climber(animal: Animal) -> bool:
	if not animal.is_climbing():
		return false

	var player_pos = animal.get_player_position()
	var animal_pos = animal.global_transform.origin
	var two_d_distance = Vector2(animal_pos.x, animal_pos.z).distance_to(
		Vector2(player_pos.x, player_pos.z)
	)
	return two_d_distance <= animal.player_detection_radius
