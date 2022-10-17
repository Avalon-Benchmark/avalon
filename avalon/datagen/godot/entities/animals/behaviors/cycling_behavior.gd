extends AnimalBehavior

class_name CyclingBehavior

export var act_steps: int
export var rest_steps: int

export var current_step := 0


func init_super(_act_steps: int, _rest_steps: int):
	self.act_steps = _act_steps
	self.rest_steps = _rest_steps
	return self


func step_behavior_cycle_forward():
	# run for run steps, rest for rest_steps
	var is_cycle_complete = current_step == get_total_cycle_steps()
	if is_cycle_complete:
		current_step = 1
	else:
		current_step += 1


func get_total_cycle_steps():
	return rest_steps + act_steps


func is_able_to_act() -> bool:
	return current_step < (act_steps + 1)


func reset():
	current_step = 0
