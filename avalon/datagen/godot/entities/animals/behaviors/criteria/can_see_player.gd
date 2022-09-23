extends BehaviorCriteria

class_name CanSeePlayer


func is_matched_by(animal: Animal) -> bool:
	return animal.is_point_in_front_of(animal.get_player_position())
