# Used by all animals to avoid hopping or flying into the ocean.
# Takes priority over all other behaviors,
# redirecting the animal back inland as best it can for the given movement_steps.
extends CyclingBehavior

class_name AvoidOcean

enum { LEFT = -1, RIGHT = 1 }

const FLY_STEPS = 50

var rng_key: String
var left_or_right_dist: ChoicesDistribution
var turn_rotation_speed: float
var speed: Vector2

var turn_direction: int

var initially_seen_ocean := Vector3.INF
var is_turn_complete := false

var total_steps_away_from_seeing_ocean := 8
var last_seen_ocean_step := -1


func _init(_rng_key: String, movement_steps: int, _speed, _turn_rotation_speed: float = 1).(
	movement_steps, 1
):
	rng_key = _rng_key
	turn_rotation_speed = _turn_rotation_speed
	left_or_right_dist = ChoicesDistribution.new([LEFT, RIGHT], [0.5, 0.5])
	speed = _speed if _speed is Vector2 else Vector2(_speed, 0)


func do(animal, delta: float) -> Vector3:
	if animal.is_climbing():
		animal.stop_climbing()

	if not is_already_avoiding():
		turn_direction = left_or_right_dist.new_value(rng_key)
		step_behavior_cycle_forward()

	var is_still_heading_towards_ocean = animal.controller.is_heading_towards_ocean()
	last_seen_ocean_step = 0 if is_still_heading_towards_ocean else last_seen_ocean_step + 1
	var is_turning_away = (
		is_still_heading_towards_ocean
		or last_seen_ocean_step < total_steps_away_from_seeing_ocean
	)
	if is_turning_away:
		animal.rotate_y(turn_direction * deg2rad(45) / total_steps_away_from_seeing_ocean)

	if is_turning_away or animal.is_mid_hop():
		var velocity = Vector3.ZERO
		if not animal.is_flying():
			velocity = animal.get_ongoing_movement()
			if is_turning_away:
				# Just stop mid air like there's a wall
				velocity = Vector3(0, animal.get_ongoing_movement().y, 0)
		else:
			# slow down significantly
			velocity = animal.get_local_forward() * speed.length() / 3
		return animal.controller.move(velocity, delta, Vector3.DOWN)

	step_behavior_cycle_forward()

	if animal.is_grounded():
		return animal.controller.hop(animal.forward_hop(speed), delta)

	animal.transform = FlyRandomly.seek_desired_altitude(animal, delta)
	var fly_speed = animal.get_local_forward() * speed.length()
	return animal.controller.move(fly_speed, delta, Vector3.DOWN)


func is_already_avoiding() -> bool:
	return current_step != 0 && is_able_to_act()


func reset():
	.reset()
	is_turn_complete = false
	last_seen_ocean_step = -1
