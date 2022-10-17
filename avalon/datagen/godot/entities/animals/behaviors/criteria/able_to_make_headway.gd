# Whether this animal can "make headway" with their current behavior,
# based on the heuristic "has traveled at least required_movement_distance before the given give_up_hops"
extends BehaviorCriteria

class_name AbleToMakeHeadway

export var required_movement_distance: int
export var give_up_after_hops: int

export var recent_position: Vector3
export var give_up_step := -1
export var is_able_to_make_headway: bool = true
export var is_last_step_counted: bool = false


func init(_required_movement_distance: int, _give_up_after_hops: int) -> BehaviorCriteria:
	required_movement_distance = _required_movement_distance
	give_up_after_hops = _give_up_after_hops
	return self


func wait_to_consider_giving_up(animal: Animal) -> bool:
	var is_on_floor = animal.controller.is_on_floor()
	if not is_last_step_counted and is_on_floor:
		if give_up_step == -1:
			finish_reset(animal)
			return true
		give_up_step = (give_up_step + 1) % give_up_after_hops
		is_last_step_counted = true
		return give_up_step != 0

	if not is_on_floor:
		is_last_step_counted = false

	return true


func is_matched_by(animal: Animal) -> bool:
	var is_waiting = wait_to_consider_giving_up(animal)

	if is_waiting:
		return is_able_to_make_headway

	is_able_to_make_headway = is_meeting_movement_requirement(animal)
	return is_able_to_make_headway


func is_meeting_movement_requirement(animal: Animal) -> bool:
	var current_position = animal.global_transform.origin

	var travel_distance = current_position.distance_to(recent_position)
	recent_position = current_position
	return travel_distance >= required_movement_distance


func finish_reset(animal: Animal):
	give_up_step = 0
	recent_position = animal.global_transform.origin


func reset():
	give_up_step = -1
	is_able_to_make_headway = true
	is_last_step_counted = false
