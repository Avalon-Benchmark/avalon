extends CyclingBehavior

class_name HopInDirection

var direction: int  # -1 towards, 1 away, 2 towards territoriy
var hop_speed: Vector2

var fixed_point := Vector3.INF  # Optional

var readiness_threshold = deg2rad(15)
var turn_speed: float

var last_remaining_rotation := INF


# NOTE It takes a rabbit roughly 3 frames to hop
func _init(
	_direction: int,
	_hop_speed: Vector2,
	_act_steps: int,
	_rest_frames: int,
	_turn_speed: float = 2.0,
	_fixed_point = null
).(_act_steps, _rest_frames):
	direction = _direction
	hop_speed = _hop_speed
	turn_speed = _turn_speed
	if _fixed_point:
		fixed_point = _fixed_point


func turn_until_ready_to_hop(animal: Animal, position: Vector3, delta: float) -> bool:
	var remaining_rotation = animal.face(direction, position, turn_speed * delta)
	var is_stuck = abs(remaining_rotation - last_remaining_rotation) < deg2rad(1)
	last_remaining_rotation = remaining_rotation
	return is_stuck or remaining_rotation <= readiness_threshold


func target_position(animal: Animal) -> Vector3:
	return fixed_point if fixed_point != Vector3.INF else animal.get_player_position()


func get_direction_velocity_multiplier() -> int:
	return -1 if direction < 0 else 1


func do(animal: Animal, delta: float) -> Vector3:
	var position = target_position(animal)
	if animal.is_mid_hop():
		return animal.controller.move(animal.get_ongoing_movement(), delta, Vector3.DOWN)

	var is_ready_to_hop = turn_until_ready_to_hop(animal, position, delta)

	# Don't run until facing sufficiently away
	if not is_ready_to_hop:
		return animal.controller.move(Vector3.ZERO, delta)

	step_behavior_cycle_forward()

	if not is_able_to_act():
		return animal.controller.move(Vector3.ZERO, delta)

	var is_sidestep_ok = true
	var step_velocity = (
		hop_speed.x
		* get_direction_velocity_multiplier()
		* animal.get_movement_direction_towards(position, is_sidestep_ok)
	)
	step_velocity.y = hop_speed.y

	return animal.controller.hop(step_velocity, delta)


func reset():
	.reset()
	last_remaining_rotation = INF


func get_name():
	var dir = ""
	match direction:
		Animal.TOWARDS_PLAYER:
			dir = "towards player"
		Animal.TOWARDS_TERRITORY:
			dir = "towards territory"
		Animal.AWAY_FROM_PLAYER:
			dir = "away from player"
	return "hop %s" % dir
