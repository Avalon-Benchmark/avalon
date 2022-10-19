extends CyclingBehavior

class_name FlyInCircles

export var fly_speed: float
export var loop_angle_increment: float


func init(_fly_speed: float, _glide_steps: int, _turn_steps: int, _loop_angle_increment: float) -> CyclingBehavior:
	.init_super(_glide_steps, _turn_steps)
	fly_speed = _fly_speed
	loop_angle_increment = _loop_angle_increment
	return self


func do(animal: Animal, delta: float) -> Vector3:
	step_behavior_cycle_forward()
	animal.transform = FlyRandomly.seek_desired_altitude(animal, delta)

	var velocity = animal.get_local_forward() * fly_speed

	var is_turning = not is_able_to_act()
	if not is_turning:
		animal.rotate_y(deg2rad(loop_angle_increment / self.rest_steps))
		velocity = velocity / 2

	return animal.controller.move(velocity, delta)
