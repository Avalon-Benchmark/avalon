extends AbleToMakeHeadway

class_name PlayerReachableByGround

var out_of_reach_height: int


func _init(_out_of_reach_height: int, _required_movement_distance: int, _give_up_after_hops: int).(
	_required_movement_distance, _give_up_after_hops
):
	out_of_reach_height = _out_of_reach_height


# used by grounded predators that consider giving up every step
# TODO consider better/more robust mechanism for saying "The player is now climbing," etc
func is_matched_by(predator: Animal) -> bool:
	var is_ready = wait_to_consider_giving_up(predator)
	if not is_ready:
		return is_able_to_make_headway

	if is_able_to_make_headway:
		is_able_to_make_headway = (
			is_meeting_movement_requirement(predator)
			or _is_player_at_or_below_elevation(predator)
		)
	else:
		# If we have already given up, don't reconsider until player comes down to our level again
		is_able_to_make_headway = _is_player_at_or_below_elevation(predator)

	return is_able_to_make_headway


func _is_player_at_or_below_elevation(predator: Animal):
	var current_elevation = predator.global_transform.origin.y
	return predator.get_player_position().y <= current_elevation + out_of_reach_height
