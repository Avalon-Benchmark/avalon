extends BehaviorCriteria

class_name PlayerWithinThreshold

var distance_threshold: float


func _init(_threshold: float):
	distance_threshold = _threshold


func is_matched_by(animal: Animal) -> bool:
	return animal.distance_to_player() <= distance_threshold
