# Equivalent of animal._is_player_in_detection_radius.
# Composed with higher-order behaviors like ActivationGate and ConditionalBehavior to avoid duplication
extends BehaviorCriteria

class_name PlayerInDetectionRadius


func is_matched_by(animal: Animal) -> bool:
	return animal._is_player_in_detection_radius
