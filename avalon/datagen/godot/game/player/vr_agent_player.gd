extends VRPlayer

class_name VRAgentPlayer


# TODO: need to remove this clamping or else it doesn't work for playback purpose
func get_head_distance_from_feet(
	head_vertical_delta_position: float, curr_head_distance_from_feet: float
) -> float:
	return clamp(
		curr_head_distance_from_feet + head_vertical_delta_position,
		min_head_position_off_of_floor,
		height + extra_height_margin
	)
