# Used by grounded predators to determine whether they should give up trying to chase the player
extends AbleToMakeHeadway

class_name PlayerReachableByGround

export var out_of_reach_height: int


func init(
	_out_of_reach_height: int, _required_movement_distance: int, _give_up_after_hops: int = NAN
) -> BehaviorCriteria:
	HARD.assert(not is_nan(_give_up_after_hops), "must supply give_up_after_hops")
	out_of_reach_height = _out_of_reach_height
	return .init(_required_movement_distance, _give_up_after_hops)


func is_matched_by(predator: Animal) -> bool:
	var is_ready_to_evaluate_progress = wait_to_consider_giving_up(predator)
	if not is_ready_to_evaluate_progress:
		return is_able_to_make_headway

	if is_able_to_make_headway:
		is_able_to_make_headway = (
			is_meeting_movement_requirement(predator)
			or _is_player_at_or_below_elevation(predator)
		)
	else:
		var is_player_within_vertical_reach_again = _is_player_at_or_below_elevation(predator)
		is_able_to_make_headway = is_player_within_vertical_reach_again

	return is_able_to_make_headway


func _is_player_at_or_below_elevation(predator: Animal):
	var current_elevation = predator.global_transform.origin.y
	return predator.get_player_position().y <= current_elevation + out_of_reach_height
