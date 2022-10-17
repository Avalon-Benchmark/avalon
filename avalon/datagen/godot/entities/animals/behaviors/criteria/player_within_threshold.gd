extends BehaviorCriteria

class_name PlayerWithinThreshold

export var distance_threshold: float


func init(_threshold: float) -> BehaviorCriteria:
	distance_threshold = _threshold
	return self


func is_matched_by(animal: Animal) -> bool:
	return animal.distance_to_player() <= distance_threshold
