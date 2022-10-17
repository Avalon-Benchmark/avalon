extends AnimalBehavior

class_name FlyInDirection

export var direction: int  # -1 towards, 1 away
export var full_speed: float
export var turn_speed: float
export var turn_rotation_speed: float

export var readiness_threshold = deg2rad(45)

export var fixed_point := Vector3.INF


func init(
	_direction: int,
	_full_speed: float,
	_turn_speed: float = 1.0,
	_turn_rotation_speed: float = 1.0,
	_fixed_point = null
) -> AnimalBehavior:
	direction = _direction
	full_speed = _full_speed
	turn_speed = _turn_speed
	turn_rotation_speed = _turn_rotation_speed
	if _fixed_point:
		fixed_point = _fixed_point
	return self


func target_position(animal: Animal) -> Vector3:
	if is_fixed_point_set():
		return fixed_point
	var vertical_correction = Vector3(0, -0.5 if direction == Animal.AWAY_FROM_PLAYER else 0.5, 0)
	return animal.get_player_position() + vertical_correction


func go_up_when_stuck(animal: Animal, target: Vector3) -> Vector3:
	if animal.is_moving():
		return target
	var animal_pos = animal.global_transform.origin
	var pos_2d = Vector3(target.x, animal_pos.y, target.z)
	var distance_2d = animal_pos.distance_to(pos_2d)
	var is_close_enough = not is_nan(distance_2d) and distance_2d < 2
	if not is_close_enough:
		# angle up to get over presumed obstacle
		target.y = animal_pos.y + (-2 if direction == Animal.AWAY_FROM_PLAYER else 2)
	return target


func do(animal: Animal, delta: float) -> Vector3:
	var position = target_position(animal)
	if direction == Animal.TOWARDS_TERRITORY:
		position.y = stay_around_coasting_altitude(animal, position.y)

	position = go_up_when_stuck(animal, position)

	var remaining_rotation = animal.face(direction, position, turn_rotation_speed * delta)
	var is_going_right_direction = remaining_rotation <= readiness_threshold

	var velocity = animal.get_local_forward()
	if is_going_right_direction:
		velocity *= full_speed
	else:
		velocity *= turn_speed

	return animal.controller.move(velocity, delta)


func stay_around_coasting_altitude(animal: Animal, target_altitude: float) -> float:
	var current_altitude = animal.global_transform.origin.y
	var is_going_above_desired_altitude = (
		target_altitude > current_altitude
		and animal.controller.is_above_desired_altitude()
	)
	if animal.controller.is_at_desired_altitude() or is_going_above_desired_altitude:
		return current_altitude
	return target_altitude


func describe():
	var dir = ""
	match direction:
		Animal.TOWARDS_PLAYER:
			dir = "towards player"
		Animal.TOWARDS_TERRITORY:
			dir = "towards territory"
		Animal.AWAY_FROM_PLAYER:
			dir = "away from player"
	return "fly %s" % dir


func is_fixed_point_set() -> bool:
	return fixed_point != Vector3.INF
