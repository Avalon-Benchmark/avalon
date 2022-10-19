extends CyclingBehavior

class_name HopInCircle

export var hop_speed: Vector2

export var loop_angle_increment: float
export var loop_steps_per_side: float


func init(_hop_speed: Vector2, _act_steps: int, _rest_steps: int, _loop_angle_increment: float) -> CyclingBehavior:
	.init_super(_act_steps, _rest_steps)
	hop_speed = _hop_speed
	loop_angle_increment = _loop_angle_increment
	return self


func do(animal: Animal, delta: float) -> Vector3:
	var step = Vector3.ZERO
	if animal.is_mid_hop():
		return animal.controller.move(animal.get_ongoing_movement(), delta)

	step_behavior_cycle_forward()

	if is_able_to_act():
		step = animal.forward_hop(hop_speed)
	else:
		animal.rotate_y(deg2rad(loop_angle_increment / rest_steps))

	return animal.controller.hop(step, delta)
