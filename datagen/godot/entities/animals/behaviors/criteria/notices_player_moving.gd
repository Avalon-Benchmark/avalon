extends BehaviorCriteria

class_name NoticesPlayerMoving

var last_player_position: Vector3 = Vector3.ZERO

var movement_detection_threshold: float = 0.01
var give_up_after_hops: int
var player_still_steps := 0


func _init(_movement_detection_threshold: float, _give_up_after_hops: int):
	movement_detection_threshold = _movement_detection_threshold
	give_up_after_hops = _give_up_after_hops


func is_matched_by(animal: Animal) -> bool:
	var player_position = animal.get_player_position()
	if player_position == Vector3.INF:
		return false

	var player_delta = (player_position - last_player_position).length()
	last_player_position = player_position

	if player_delta > movement_detection_threshold:
		player_still_steps = 0
		return true

	player_still_steps += 1
	return player_still_steps < give_up_after_hops
